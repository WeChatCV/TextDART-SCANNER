import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["DIFFUSERS_DISABLE_NATIVE_ATTENTION"] = "1"
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


random.seed(42)
torch.manual_seed(42)


if __package__:
    from . import qwen_eval_common as _common
else:
    import qwen_eval_common as _common

__all__ = _common.GENTEXTEVAL_EXPORTS


def __getattr__(name):
    return getattr(_common, name)


def main():
    _common.run_jsonl_benchmark('gentexteval')


if __name__ == "__main__":
    main()
