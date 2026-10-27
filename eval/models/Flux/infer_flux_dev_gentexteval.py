import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random

import torch


# Fixed seeds so all ranks sample consistently.
random.seed(42)
torch.manual_seed(42)


if __package__:
    from . import flux_eval_common as _common
else:
    import flux_eval_common as _common

__all__ = _common.GENTEXTEVAL_EXPORTS


def __getattr__(name):
    return getattr(_common, name)


def main():
    _common.run_jsonl_benchmark('gentexteval')


if __name__ == "__main__":
    main()
