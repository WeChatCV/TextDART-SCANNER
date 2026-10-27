import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["DIFFUSERS_DISABLE_NATIVE_ATTENTION"] = "1"
import random
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_resume import is_complete_image


random.seed(42)
torch.manual_seed(42)


if __package__:
    from .qwen_eval_common import (
        build_parser as build_parser,
        create_image_gallery as create_image_gallery,
        load_model as load_model,
        load_json as load_json,
        make_seed as make_seed,
        setup_distributed as setup_distributed,
        create_rank0_progress as create_rank0_progress,
        update_rank0_progress as update_rank0_progress,
        custom_collate_fn as custom_collate_fn,
    )
else:
    from qwen_eval_common import (
        build_parser as build_parser,
        create_image_gallery as create_image_gallery,
        load_model as load_model,
        load_json as load_json,
        make_seed as make_seed,
        setup_distributed as setup_distributed,
        create_rank0_progress as create_rank0_progress,
        update_rank0_progress as update_rank0_progress,
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

        # Resume: skip samples whose gallery image already exists and is non-empty.
        if self.output_dir:
            filtered_data = []
            for item in self.data:
                image_path = os.path.join(self.output_dir, f"{item['id']}.webp")
                if not is_complete_image(image_path):
                    filtered_data.append(item)
                else:
                    exitsnums += 1
                    print(f"image_path:{image_path} exists thus skip")
            self.data = filtered_data

        print(f"Loaded {len(self.data)} Text_Rendering samples, existing {exitsnums} samples")

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
    _positive_magic = {"EN": "Ultra HD, 4K, cinematic composition.",
        "ZH": "\u8d85\u6e05\uff0c4K\uff0c\u7535\u5f71\u7ea7\u6784\u56fe"}
    negative_prompt = " "
    aspect_ratios = {
        "1:1": (1328, 1328),  # Recommended Qwen-Image resolutions.
        "16:9": (1664, 928),
        "9:16": (928, 1664),
    }
    width, height = aspect_ratios["1:1"]
    args = parser.parse_args()

    # CLI width/height override the default aspect-ratio resolution.
    width, height = int(args.width), int(args.height)

    # Use the passed negative prompt when provided; otherwise keep the built-in default.
    if (
        args.negative_prompt is not None
        and args.negative_prompt.strip() != ""
        and args.negative_prompt != "None"
    ):
        negative_prompt = args.negative_prompt

    print(f"negative_prompt:{negative_prompt}")
    if args.mode == "EN":
        load_json_file = "OneIG-Benchmark/OneIG-Bench/OneIG-Bench.json"
    else:
        load_json_file = "OneIG-Benchmark/OneIG-Bench/OneIG-Bench-ZH.json"
    batch_size = 1
    output_dir = os.path.join(args.gallery_output_dir, args.model_name + f"_{args.mode}")
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()
    use_lora = True
    ckpt_path = args.ckpt_path
    if (not ckpt_path) or (ckpt_path.strip() == "") or (not os.path.exists(ckpt_path)):
        use_lora = False
        ckpt_path = None
    ds = load_json(load_json_file)
    dataset = PromptDatasetOneIG(ds, output_dir=output_dir)
    if local_rank == 0:
        print(f"Resume filter: pending={len(dataset)} output={output_dir}")
    dist.barrier()
    if len(dataset) == 0:
        if local_rank == 0:
            print(f"Nothing to generate; all complete images already exist. -> {output_dir}")
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return
    pipe = load_model(local_rank, args.model_path, ckpt_path, use_lora)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, collate_fn=custom_collate_fn
    )
    progress_bar = create_rank0_progress(
        local_rank,
        len(dataset),
        f"{args.model_name} | OneIG | {args.mode}",
    )
    for _batch_infer_idx, (batch_prompts, batch_ids) in enumerate(dataloader):
        enhanced_prompts = batch_prompts
        prompt_images = [[] for _ in range(len(batch_prompts))]
        # OneIG expects 4 samples per prompt, saved together as one 2x2 gallery.
        for i in range(4):
            with torch.no_grad():
                generators = [
                    torch.Generator(device=f"cuda:{local_rank}").manual_seed(make_seed(pid, i))
                    for pid in batch_ids
                ]
                batch_images = pipe(
                    prompt=enhanced_prompts,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=50,
                    true_cfg_scale=4.0,
                    generator=generators,
                ).images

            for j, img in enumerate(batch_images):
                prompt_images[j].append(img)

        for i in range(len(batch_prompts)):
            current_images = prompt_images[i]
            if len(current_images) < 4:
                if local_rank == 0:
                    tqdm.write(
                        (
                            f"Warning: prompt {batch_ids[i]} generated {len(current_images)}/4 images; padding "
                            f"before saving."
                        )
                    )
                while len(current_images) < 4:
                    current_images.append(current_images[-1])
            gallery = create_image_gallery(current_images, rows=2, cols=2)
            file_path = os.path.join(output_dir, f"{batch_ids[i]}.webp")
            with open(file_path, "wb") as f:
                gallery.save(f, "WebP")
        update_rank0_progress(progress_bar, local_rank, len(batch_ids), len(dataset))

    if progress_bar is not None:
        progress_bar.close()
    if local_rank == 0:
        print(f"Done: {args.model_name} | OneIG | {args.mode} -> {output_dir}")

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
    'create_rank0_progress',
    'custom_collate_fn',
    'load_json',
    'load_model',
    'main',
    'make_seed',
    'setup_distributed',
    'update_rank0_progress',
]
