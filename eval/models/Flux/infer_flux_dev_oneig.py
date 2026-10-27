import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random

import megfile
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm


# Fixed seeds so all ranks sample consistently.
random.seed(42)
torch.manual_seed(42)


if __package__:
    from .flux_eval_common import (
        build_parser as build_parser,
        create_image_gallery as create_image_gallery,
        load_json as load_json,
        make_seed as make_seed,
        setup_distributed as setup_distributed,
        load_model as load_model,
        custom_collate_fn as custom_collate_fn,
    )
else:
    from flux_eval_common import (
        build_parser as build_parser,
        create_image_gallery as create_image_gallery,
        load_json as load_json,
        make_seed as make_seed,
        setup_distributed as setup_distributed,
        load_model as load_model,
        custom_collate_fn as custom_collate_fn,
    )


class PromptDatasetOneIG(Dataset):
    def __init__(self, data, output_dir=None):
        self.output_dir = output_dir
        self.data = []

        if isinstance(data, dict):
            data = list(data.values())
        elif not isinstance(data, list):
            raise ValueError("Unsupported data type")

        exitsnums = 0
        for item in data:
            if item.get("category") == "Text_Rendering":
                self.data.append(item)

        if self.output_dir:
            filtered_data = []
            for item in self.data:
                image_path = os.path.join(self.output_dir, f"{item['id']}.webp")
                if not os.path.exists(image_path):
                    filtered_data.append(item)
                else:
                    exitsnums += 1
                    print(f"image_path:{image_path} exists thus skip")

            self.data = filtered_data

        print(f"Loaded {len(self.data)} Text_Rendering samples,exisiting {exitsnums} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item.get("prompt_en") or item.get("prompt_cn"),
            "id": item["id"],
        }


def main():
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    print(f"Total number of processes: {world_size}")
    parser = build_parser('oneig')
    negative_prompt = None
    width, height = 1024, 1024
    args = parser.parse_args()

    # CLI width/height override the default resolution.
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

    if args.mode == "EN":
        load_json_file = "OneIG-Benchmark/OneIG-Bench/OneIG-Bench.json"
    else:
        load_json_file = "OneIG-Benchmark/OneIG-Bench/OneIG-Bench-ZH.json"
    batch_size = 16

    output_dir = os.path.join(args.gallery_output_dir, args.model_name + f"_{args.mode}")
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()
    use_lora = True
    ckpt_path = args.ckpt_path
    if (not ckpt_path) or (ckpt_path.strip() == "") or (not os.path.exists(ckpt_path)):
        use_lora = False
        ckpt_path = None
    pipe = load_model(local_rank, args.model_path, ckpt_path, use_lora)

    ds = load_json(load_json_file)
    dataset = PromptDatasetOneIG(ds, output_dir=output_dir)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, collate_fn=custom_collate_fn
    )
    for _batch_infer_idx, (batch_prompts, batch_ids) in tqdm(
        enumerate(dataloader), desc=f"Processing batches OneIG-{args.mode}"
    ):
        prompt_images = [[] for _ in range(len(batch_prompts))]
        # Run 4 passes over the batch; prompt_images[j] collects the 4 images for prompt j.
        for i in range(4):
            print(f"gen prompt idx :{i}")
            with torch.no_grad():
                generators = [
                    torch.Generator(device=f"cuda:{local_rank}").manual_seed(make_seed(pid, i))
                    for pid in batch_ids
                ]
                batch_images = pipe(
                    batch_prompts,
                    height=height,
                    width=width,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    generator=generators,
                    **neg_kwargs,
                ).images
            for i, img in enumerate(batch_images):
                prompt_images[i].append(img)
        print(f"len(prompt_images):{len(prompt_images)}")

        # Save each prompt's 4 images as one 2x2 gallery named {id}.webp.
        for i in range(len(batch_prompts)):
            current_images = prompt_images[i]
            if len(current_images) < 4:
                print(
                    f"Warning: prompt {i} has only {len(current_images)} images (need 4); padding with the last image."
                )
                while len(current_images) < 4:
                    current_images.append(current_images[-1])
            else:
                print(f"Got 4 images, saving gallery")
            gallery = create_image_gallery(current_images, rows=2, cols=2)
            file_path = megfile.smart_path_join(output_dir, f"{batch_ids[i]}.webp")
            with megfile.smart_open(file_path, "wb") as f:
                gallery.save(f, "WebP")

    print(f"finish OneIG-{args.mode} !")

    # Synchronize all ranks then tear down the process group so torchrun exits
    # cleanly. Without this, NCCL shutdown can deadlock and block the outer loop.
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


__all__ = [
    'PromptDatasetOneIG',
    'build_parser',
    'create_image_gallery',
    'custom_collate_fn',
    'load_json',
    'load_model',
    'main',
    'make_seed',
    'setup_distributed',
]
