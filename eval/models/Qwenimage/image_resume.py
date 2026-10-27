"""Resume helpers for Qwen-Image eval generation.

A path counts as done only when it is a non-empty file. Zero-byte leftovers
from crashed writes must be regenerated.
"""

import os


def is_complete_image(path):
    """Return True if ``path`` exists on disk and has at least one byte."""
    if not os.path.isfile(path):
        return False
    return os.path.getsize(path) > 0
