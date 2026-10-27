"""Shared Qwenimage evaluation helpers."""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline
from peft import PeftModel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

if __package__:
    from .image_resume import is_complete_image
else:
    from image_resume import is_complete_image



if __package__:
    from ..generation_common import assemble_gallery, JsonlPromptDataset, custom_collate_fn, load_json, make_seed
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from generation_common import assemble_gallery, JsonlPromptDataset, custom_collate_fn, load_json, make_seed


def create_image_gallery(images, rows=2, cols=2, target_size=(1024, 1024)):
    return assemble_gallery(images, rows, cols, target_size)


def load_model(local_rank, model_path, ckpt_path=None, use_lora=True):
    torch_dtype = torch.bfloat16
    device = f"cuda:{local_rank}"

    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    ).to(device)

    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    if ckpt_path is not None and use_lora:
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


def setup_distributed():
    import datetime

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def create_rank0_progress(local_rank, total, desc):
    if local_rank != 0:
        return None
    return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=True, unit="sample")


def update_rank0_progress(progress_bar, local_rank, count, total):
    if local_rank == 0 and progress_bar is not None:
        world_size = dist.get_world_size()
        progress_bar.update(count * world_size)


class PromptDatasetJSONL(JsonlPromptDataset):
    """Load pending prompts using this backbone's image completion check."""

    def __init__(self, jsonl_path, output_dir):
        super().__init__(jsonl_path, output_dir, is_complete_image)


_DEFAULT_ARGUMENTS = {'--mode': {'type': str,
            'choices': ['ZH', 'EN'],
            'default': 'EN',
            'help': 'Dataset mode: ZH (Chinese) or EN (English)'},
 '--model_name': {'type': str,
                  'default': 'Qwenimage',
                  'help': 'Name of the model used for output directory'},
 '--model_path': {'type': str,
                  'default': 'Qwen/Qwen-Image',
                  'help': 'Base model path or Hugging Face model id'},
 '--ckpt_path': {'type': str,
                 'default': None,
                 'help': 'Path to PEFT LoRA weights directory'},
 '--gallery_output_dir': {'type': str,
                          'default': 'GenTextEval/eval_results',
                          'help': 'Root directory for saved images'},
 '--negative_prompt': {'type': str,
                       'default': None,
                       'help': 'Negative prompt. If omitted (or empty/None), the '
                               'built-in default is used.'},
 '--width': {'type': int,
             'default': 1328,
             'help': 'Width of the generated image.'},
 '--height': {'type': int,
              'default': 1328,
              'help': 'Height of the generated image.'}}

_ARGUMENT_ORDER = {'gentexteval': ['--mode',
                 '--model_name',
                 '--model_path',
                 '--ckpt_path',
                 '--gallery_output_dir',
                 '--negative_prompt',
                 '--width',
                 '--height'],
 'oneig': ['--mode',
           '--model_name',
           '--model_path',
           '--ckpt_path',
           '--gallery_output_dir',
           '--negative_prompt',
           '--width',
           '--height'],
 'longtext': ['--mode',
              '--model_name',
              '--model_path',
              '--ckpt_path',
              '--gallery_output_dir',
              '--negative_prompt',
              '--width',
              '--height'],
 'cvtg2k': ['--width',
            '--height',
            '--model_name',
            '--model_path',
            '--ckpt_path',
            '--gallery_output_dir',
            '--negative_prompt']}

_ARGUMENT_OVERRIDES = {'oneig': {'--gallery_output_dir': {'default': 'OneIG-Benchmark/images/text'}},
 'longtext': {'--gallery_output_dir': {'default': 'X-Omni/eval_results'}},
 'cvtg2k': {'--width': {'type': str, 'help': 'Width of the image'},
            '--height': {'type': str, 'help': 'Height of the image'},
            '--gallery_output_dir': {'default': 'TextCrafter/CVTG-2K/eval_results',
                                     'help': 'Root output directory'}}}


def build_parser(benchmark):
    """Build the CLI with benchmark-specific defaults."""
    parser = argparse.ArgumentParser()
    overrides = _ARGUMENT_OVERRIDES.get(benchmark, {})
    for flag in _ARGUMENT_ORDER[benchmark]:
        kwargs = {**_DEFAULT_ARGUMENTS[flag], **overrides.get(flag, {})}
        parser.add_argument(flag, **kwargs)
    return parser


def run_jsonl_benchmark(benchmark):
    paths = {
        "longtext": ("X-Omni/textbench/text_prompts.jsonl", "X-Omni/textbench/text_prompts_zh.jsonl"),
        "gentexteval": ("GenTextEval/GenTextEval_en.jsonl", "GenTextEval/GenTextEval_zh.jsonl"),
    }[benchmark]
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    print(f"Total number of processes: {world_size}")
    parser = build_parser(benchmark)
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

    if args.mode == "EN":
        jsonl_path = paths[0]
    else:
        jsonl_path = paths[1]
    batch_size = 1

    output_dir = os.path.join(args.gallery_output_dir, f"{args.model_name}_{args.mode}")
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()
    use_lora = True
    ckpt_path = args.ckpt_path
    if (not ckpt_path) or (ckpt_path.strip() == "") or (not os.path.exists(ckpt_path)):
        use_lora = False
        ckpt_path = None
    dataset = PromptDatasetJSONL(jsonl_path, output_dir)
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
        f"{args.model_name} | {benchmark} | {args.mode}",
    )
    # Each prompt is sampled 4 times and saved as {prompt_id}_{repeat_id}.png.
    for _batch_infer_idx, (batch_prompts, batch_ids) in enumerate(dataloader):
        enhanced_prompts = batch_prompts
        for repeat_id in range(1, 5):
            with torch.no_grad():
                generators = [
                    torch.Generator(device=f"cuda:{local_rank}").manual_seed(
                        make_seed(pid, repeat_id)
                    )
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

            for img, prompt_id in zip(batch_images, batch_ids):
                img_path = os.path.join(output_dir, f"{prompt_id}_{repeat_id}.png")
                img.save(img_path, "PNG")
        update_rank0_progress(progress_bar, local_rank, len(batch_ids), len(dataset))

    if progress_bar is not None:
        progress_bar.close()
    if local_rank == 0:
        print(f"Done: {args.model_name} | {benchmark} | {args.mode} -> {output_dir}")

    # Synchronize all ranks then tear down the process group so torchrun exits
    # cleanly. Without this, NCCL shutdown can deadlock and block the outer loop.
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


__all__ = ["load_json", "make_seed", "custom_collate_fn", "PromptDatasetJSONL", "load_model",
           "create_image_gallery", "setup_distributed", "build_parser", "run_jsonl_benchmark"]


LONGTEXT_EXPORTS = (
    'PromptDatasetJSONL',
    'build_parser',
    'create_image_gallery',
    'create_rank0_progress',
    'custom_collate_fn',
    'load_json',
    'load_model',
    'main',
    'make_seed',
    'run_jsonl_benchmark',
    'setup_distributed',
    'update_rank0_progress',
)


GENTEXTEVAL_EXPORTS = (
    'PromptDatasetJSONL',
    'build_parser',
    'create_image_gallery',
    'create_rank0_progress',
    'custom_collate_fn',
    'load_json',
    'load_model',
    'main',
    'make_seed',
    'run_jsonl_benchmark',
    'setup_distributed',
    'update_rank0_progress',
)
