"""Shared Flux evaluation helpers."""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import FluxPipeline
from peft import PeftModel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm



if __package__:
    from ..generation_common import assemble_gallery, JsonlPromptDataset, custom_collate_fn, load_json, make_seed
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from generation_common import assemble_gallery, JsonlPromptDataset, custom_collate_fn, load_json, make_seed


def create_image_gallery(images, rows=2, cols=2):
    return assemble_gallery(images, rows, cols)


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


class PromptDatasetJSONL(JsonlPromptDataset):
    """Load pending prompts using this backbone's image completion check."""

    def __init__(self, jsonl_path, output_dir):
        super().__init__(jsonl_path, output_dir, os.path.exists)


def load_model(local_rank, model_path, ckpt_path=None, use_lora=True):
    pipeline = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(f"cuda:{local_rank}")
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not local_rank == 0,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
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
        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, adapter_dir)
    pipeline.transformer.eval()
    return pipeline


_DEFAULT_ARGUMENTS = {'--mode': {'type': str,
            'choices': ['ZH', 'EN'],
            'default': 'EN',
            'help': 'Dataset language (ZH/EN)'},
 '--model_name': {'type': str,
                  'default': 'Flux',
                  'help': 'Model name (used as the output subdirectory)'},
 '--model_path': {'type': str,
                  'default': 'black-forest-labs/FLUX.1-dev',
                  'help': 'Base model path or HF id'},
 '--ckpt_path': {'type': str,
                 'default': None,
                 'help': 'Path to PEFT LoRA weights directory'},
 '--gallery_output_dir': {'type': str,
                          'default': 'X-Omni/eval_results',
                          'help': 'Root directory for saved images'},
 '--negative_prompt': {'type': str,
                       'default': None,
                       'help': 'Negative prompt. If omitted (or empty/None), no '
                               'true-CFG negative prompt is used.'},
 '--width': {'type': int,
             'default': 1024,
             'help': 'Width of the generated image.'},
 '--height': {'type': int,
              'default': 1024,
              'help': 'Height of the generated image.'}}

_ARGUMENT_ORDER = {'longtext': ['--mode',
              '--model_name',
              '--model_path',
              '--ckpt_path',
              '--gallery_output_dir',
              '--negative_prompt',
              '--width',
              '--height'],
 'cvtg2k': ['--model_name',
            '--model_path',
            '--ckpt_path',
            '--gallery_output_dir',
            '--negative_prompt',
            '--width',
            '--height'],
 'oneig': ['--mode',
           '--model_name',
           '--ckpt_path',
           '--model_path',
           '--gallery_output_dir',
           '--negative_prompt',
           '--width',
           '--height'],
 'gentexteval': ['--mode',
                 '--model_name',
                 '--model_path',
                 '--ckpt_path',
                 '--gallery_output_dir',
                 '--negative_prompt',
                 '--width',
                 '--height']}

_ARGUMENT_OVERRIDES = {'cvtg2k': {'--model_name': {'help': 'Model name, used in the output path'},
            '--gallery_output_dir': {'default': 'TextCrafter/CVTG-2K/eval_results',
                                     'help': 'Output root directory'}},
 'oneig': {'--mode': {'help': 'Dataset mode: ZH (Chinese) or EN (English)'},
           '--model_name': {'help': 'Name of the model used for output directory'},
           '--gallery_output_dir': {'default': 'OneIG-Benchmark/images/text/',
                                    'help': 'Root directory for saved galleries'}},
 'gentexteval': {'--gallery_output_dir': {'default': 'GenTextEval/eval_results'}}}


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
        jsonl_path = paths[0]
    else:
        jsonl_path = paths[1]

    output_dir = os.path.join(args.gallery_output_dir, f"{args.model_name}_{args.mode}")
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()

    use_lora = True
    ckpt_path = args.ckpt_path
    if (not ckpt_path) or (ckpt_path.strip() == "") or (not os.path.exists(ckpt_path)):
        use_lora = False
        ckpt_path = None
    pipe = load_model(local_rank, args.model_path, ckpt_path, use_lora)
    dataset = PromptDatasetJSONL(jsonl_path, output_dir)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler, collate_fn=custom_collate_fn)

    # Each prompt is generated 4 times and saved as {prompt_id}_{repeat_id}.png.
    for batch_infer_idx, (batch_prompts, batch_ids) in tqdm(
        enumerate(dataloader), desc=f"Processing batches {args.mode}"
    ):
        for repeat_id in range(1, 5):
            with torch.no_grad():
                generators = [
                    torch.Generator(device=f"cuda:{local_rank}").manual_seed(
                        make_seed(pid, repeat_id)
                    )
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

            for img, prompt_id in zip(batch_images, batch_ids):
                img_path = os.path.join(output_dir, f"{prompt_id}_{repeat_id}.png")
                img.save(img_path, "PNG")

        print(f"Batch {batch_infer_idx} done ({args.mode})")

    print(f"{args.mode} inference finished, images saved to: {output_dir}")

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
    'custom_collate_fn',
    'load_json',
    'load_model',
    'main',
    'make_seed',
    'run_jsonl_benchmark',
    'setup_distributed',
)


GENTEXTEVAL_EXPORTS = (
    'PromptDatasetJSONL',
    'build_parser',
    'custom_collate_fn',
    'load_json',
    'load_model',
    'main',
    'make_seed',
    'run_jsonl_benchmark',
    'setup_distributed',
)
