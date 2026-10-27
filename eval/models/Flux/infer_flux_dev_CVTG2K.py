# pylint: disable=invalid-name
# Preserve public package and command names.
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

# NCCL default is 10 min. Resume runs leave some ranks generating while others
# skip; the next dist.barrier() then times out and aborts area 5. Two hours
# covers a full leftover Flux 1024 / 50-step pass.
NCCL_TIMEOUT = timedelta(hours=2)
# Flux 1024x1024 x 50 steps cannot reasonably run 16-wide on one GPU. A leftover
# batch of that size is also what pushes a generate-heavy rank past the NCCL
# timeout while skip-heavy ranks wait at the next split barrier.
BATCH_SIZE = 1


# Fixed seeds so all ranks sample consistently.
random.seed(42)
torch.manual_seed(42)


if __package__:
    from .flux_eval_common import (
        build_parser as build_parser,
        create_image_gallery as create_image_gallery,
        load_json as load_json,
        make_seed as make_seed,
        load_model as load_model,
    )
else:
    from flux_eval_common import (
        build_parser as build_parser,
        create_image_gallery as create_image_gallery,
        load_json as load_json,
        make_seed as make_seed,
        load_model as load_model,
    )


def image_ready(img_path):
    return os.path.isfile(img_path) and os.path.getsize(img_path) > 0


def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=NCCL_TIMEOUT)
    return local_rank


class PromptDatasetCVTG2K(Dataset):
    def __init__(self, benchmark, area, output_dir):
        self.output_dir = output_dir
        self.data = []

        json_path = f"TextCrafter/CVTG-2K/{benchmark}/{area}.json"
        json_data = load_json(json_path)
        if not json_data:
            raise ValueError(f"Failed to load data from {json_path}")

        data_list = json_data.get("data_list", [])
        data_list.sort(key=lambda x: x["index"])

        # Drop already-written images BEFORE DistributedSampler so remaining
        # work is rebalanced across ranks on resume. Sampling the full list
        # first leaves skip-heavy ranks idle at the next split barrier while
        # generate-heavy ranks are still in pipe(), which NCCL then kills.
        for item in data_list:
            index = item.get("index")
            prompt = item.get("prompt")
            if index is None or not prompt:
                continue
            img_path = os.path.join(output_dir, f"{index}.png")
            if image_ready(img_path):
                continue
            self.data.append({"prompt": prompt, "index": index})

        print(
            f"Loaded {len(data_list)} CVTG2K samples "
            f"(benchmark: {benchmark}, area: {area}); "
            f"{len(self.data)} missing"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "index": item["index"],
        }


def custom_collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    indices = [item["index"] for item in batch]
    return prompts, indices


def main():
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    print(f"Total number of processes: {world_size}")

    parser = build_parser('cvtg2k')
    negative_prompt = None
    args = parser.parse_args()

    width, height = int(args.width), int(args.height)
    # Use the passed negative prompt when provided; otherwise no negative prompt.
    if (
        args.negative_prompt is not None
        and args.negative_prompt.strip() != ""
        and args.negative_prompt != "None"
    ):
        negative_prompt = args.negative_prompt
    neg_kwargs = {}
    if negative_prompt is not None:
        neg_kwargs = {"negative_prompt": negative_prompt, "true_cfg_scale": 4.0}

    use_lora = True
    ckpt_path = args.ckpt_path
    if (not ckpt_path) or (ckpt_path.strip() == "") or (not os.path.exists(ckpt_path)):
        use_lora = False
        ckpt_path = None
    pipe = load_model(local_rank, args.model_path, ckpt_path, use_lora)
    output_model_name = f"{args.model_name}_EN"

    for area in range(2, 6):
        for benchmark in ("CVTG", "CVTG-Style"):
            # Output layout: <model>/<benchmark>/<area>, e.g. Flux_EN/CVTG/2
            output_dir = os.path.join(
                args.gallery_output_dir, output_model_name, benchmark, str(area)
            )
            os.makedirs(output_dir, exist_ok=True)
            dist.barrier()

            # Only missing samples are loaded; an empty dataset must not be passed to DistributedSampler.
            dataset = PromptDatasetCVTG2K(benchmark, area, output_dir)
            if len(dataset) == 0:
                print(f"Skip {benchmark} area {area}: all images already exist")
            else:
                sampler = DistributedSampler(dataset, shuffle=False)
                dataloader = DataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    sampler=sampler,
                    collate_fn=custom_collate_fn,
                )

                for _batch_infer_idx, (batch_prompts, batch_indices) in tqdm(
                    enumerate(dataloader),
                    desc=f"Processing {benchmark} area {area} (rank {local_rank})",
                ):
                    # Re-stat each image right before generate. Another concurrent
                    # run of this script may have written the file after Dataset
                    # was constructed (or even after this batch was loaded).
                    pending_prompts = []
                    pending_indices = []
                    for prompt, idx in zip(batch_prompts, batch_indices):
                        img_path = os.path.join(output_dir, f"{idx}.png")
                        if image_ready(img_path):
                            print(f"image_path:{img_path} exists thus skip")
                            continue
                        pending_prompts.append(prompt)
                        pending_indices.append(idx)

                    if not pending_prompts:
                        continue

                    with torch.no_grad():
                        generators = [
                            torch.Generator(device=f"cuda:{local_rank}").manual_seed(
                                make_seed(idx, 1)
                            )
                            for idx in pending_indices
                        ]
                        batch_images = pipe(
                            pending_prompts,
                            height=height,
                            width=width,
                            guidance_scale=3.5,
                            num_inference_steps=50,
                            max_sequence_length=512,
                            generator=generators,
                            **neg_kwargs,
                        ).images

                    for img, idx in zip(batch_images, pending_indices):
                        img_path = os.path.join(output_dir, f"{idx}.png")
                        if image_ready(img_path):
                            print(f"image_path:{img_path} exists thus skip save")
                            continue
                        img.save(img_path, "PNG")

            print(f"Finish {benchmark} area {area} (rank {local_rank})!")
            # Keep ranks in lockstep between splits so a slow generate cannot
            # overlap the next split's barrier with a 10-min-default timeout.
            dist.barrier()

    # Synchronize all ranks then tear down the process group so torchrun exits
    # cleanly. Without this, NCCL shutdown can deadlock and block the outer loop.
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


__all__ = [
    'PromptDatasetCVTG2K',
    'build_parser',
    'create_image_gallery',
    'custom_collate_fn',
    'image_ready',
    'load_json',
    'load_model',
    'main',
    'make_seed',
    'setup_distributed',
]
