# pylint: disable=invalid-name
# Preserve public package and command names.
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["DIFFUSERS_DISABLE_NATIVE_ATTENTION"] = "1"
import random
import sys

import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_resume import is_complete_image


if __package__:
    from .qwen_eval_common import (
        build_parser as build_parser,
        create_image_gallery as create_image_gallery,
        load_json as load_json,
        make_seed as make_seed,
        setup_distributed as setup_distributed,
        create_rank0_progress as create_rank0_progress,
        update_rank0_progress as update_rank0_progress,
    )
else:
    from qwen_eval_common import (
        build_parser as build_parser,
        create_image_gallery as create_image_gallery,
        load_json as load_json,
        make_seed as make_seed,
        setup_distributed as setup_distributed,
        create_rank0_progress as create_rank0_progress,
        update_rank0_progress as update_rank0_progress,
    )


def load_model(local_rank, model_path, ckpt_path=None, use_lora=True):
    torch_dtype = torch.bfloat16
    device = f"cuda:{local_rank}"

    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    ).to(device)

    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    if ckpt_path is not None and os.path.exists(ckpt_path) and use_lora:
        # Load the trained LoRA adapter from its directory. ckpt_path should point
        # at a PEFT adapter directory containing adapter_config.json +
        # adapter_model.safetensors (e.g. ".../checkpoints/checkpoint-300"). A path
        # to the adapter_model.safetensors file is also accepted and resolved to its
        # parent directory.
        adapter_dir = ckpt_path
        if os.path.isfile(adapter_dir):
            adapter_dir = os.path.dirname(adapter_dir)
        adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
        if not os.path.isfile(adapter_config_path):
            raise FileNotFoundError(
                f"Expected a PEFT adapter directory containing adapter_config.json, "
                f"but none was found. ckpt_path={ckpt_path!r}, resolved "
                f"adapter_dir={adapter_dir!r}."
            )
        print(f"Loading LoRA adapter from: {adapter_dir}")
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, adapter_dir)
        pipe.transformer.eval()

    return pipe


random.seed(42)
torch.manual_seed(42)


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

        # Resume: keep only samples whose image is missing or empty.
        for item in data_list:
            index = item.get("index")
            prompt = item.get("prompt")
            if index is not None and prompt:
                image_path = os.path.join(output_dir, f"{index}.png")
                if not is_complete_image(image_path):
                    self.data.append({"prompt": prompt, "index": index})

        print(f"Loaded {len(self.data)} CVTG2K samples (benchmark: {benchmark}, area: {area})")

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
    _positive_magic = {"EN": "Ultra HD, 4K, cinematic composition.",
        "ZH": "\u8d85\u6e05\uff0c4K\uff0c\u7535\u5f71\u7ea7\u6784\u56fe"}
    negative_prompt = " "
    _aspect_ratios = {
        "1:1": (1328, 1328),  # Recommended Qwen-Image resolutions.
        "16:9": (1664, 928),
        "9:16": (928, 1664),
    }
    args = parser.parse_args()

    width, height = int(args.width), int(args.height)
    # Use the passed negative prompt when provided; otherwise keep the built-in default.
    if (
        args.negative_prompt is not None
        and args.negative_prompt.strip() != ""
        and args.negative_prompt != "None"
    ):
        negative_prompt = args.negative_prompt

    use_lora = True
    ckpt_path = args.ckpt_path
    if (not ckpt_path) or (ckpt_path.strip() == "") or (not os.path.exists(ckpt_path)):
        use_lora = False
        ckpt_path = None
    pipe = None
    batch_size = 1
    output_model_name = f"{args.model_name}_EN"
    for area in range(2, 6):
        for benchmark in ("CVTG", "CVTG-Style"):
            # Output layout: <gallery_output_dir>/<model>_EN/<benchmark>/<area>
            output_dir = os.path.join(
                args.gallery_output_dir, output_model_name, benchmark, str(area)
            )
            if local_rank == 0:
                os.makedirs(output_dir, exist_ok=True)
            dist.barrier()

            dataset = PromptDatasetCVTG2K(benchmark, area, output_dir)
            if local_rank == 0:
                print(
                    f"Resume filter: pending={len(dataset)} "
                    f"{benchmark} area={area} output={output_dir}"
                )
            dist.barrier()
            if len(dataset) == 0:
                if local_rank == 0:
                    print(f"Nothing pending for {benchmark} area {area}, skip.")
                continue
            if pipe is None:
                pipe = load_model(local_rank, args.model_path, ckpt_path, use_lora)
            sampler = DistributedSampler(dataset, shuffle=False)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, sampler=sampler, collate_fn=custom_collate_fn
            )
            progress_bar = create_rank0_progress(
                local_rank,
                len(dataset),
                f"{output_model_name} | CVTG2K | {benchmark} area {area}",
            )
            for _batch_infer_idx, (batch_prompts, batch_indices) in enumerate(dataloader):
                enhanced_prompts = batch_prompts
                with torch.no_grad():
                    generators = [
                        torch.Generator(device=f"cuda:{local_rank}").manual_seed(make_seed(idx, 1))
                        for idx in batch_indices
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

                for img, idx in zip(batch_images, batch_indices):
                    img_path = os.path.join(output_dir, f"{idx}.png")
                    img.save(img_path, "PNG")
                update_rank0_progress(progress_bar, local_rank, len(batch_indices), len(dataset))

            if progress_bar is not None:
                progress_bar.close()
            if local_rank == 0:
                print(
                    f"Done: {output_model_name} | CVTG2K | {benchmark} area {area} -> {output_dir}"
                )

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
    'create_rank0_progress',
    'custom_collate_fn',
    'load_json',
    'load_model',
    'main',
    'make_seed',
    'setup_distributed',
    'update_rank0_progress',
]
