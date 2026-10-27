"""Shared prompt loading and deterministic seed utilities."""

import json
import os
import zlib

from PIL import Image
from torch.utils.data import Dataset


def load_json(file_path):
    """
    Load a JSON file and parse its contents into a Python object.

    :param file_path: Path to the JSON file.
    :return: The parsed object, or None if loading fails.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: failed to parse JSON in {file_path}")
        return None
    except (OSError, UnicodeError, TypeError, ValueError, RecursionError) as e:
        print(f"Unexpected error: {e}")
        return None


def make_seed(prompt_id, repeat_id):
    # Deterministic per-(prompt, repeat) seed: the same prompt + repeat always maps
    # to the same latent regardless of model, GPU count, batch size or resumed runs.
    return zlib.crc32(f"{prompt_id}_{repeat_id}".encode("utf-8")) & 0xFFFFFFFF


def custom_collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    ids = [item["id"] for item in batch]
    return prompts, ids


class JsonlPromptDataset(Dataset):
    def __init__(self, jsonl_path, output_dir, complete_image):
        self.output_dir = output_dir
        self.data = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    prompt = item.get("prompt")
                    prompt_id = item.get("prompt_id")

                    if prompt is None or prompt_id is None:
                        continue

                    # Skip prompts whose 4 images have all been generated already.
                    all_exist = True
                    for repeat_id in range(1, 5):
                        img_path = os.path.join(output_dir, f"{prompt_id}_{repeat_id}.png")
                        if not complete_image(img_path):
                            all_exist = False
                            break
                    if not all_exist:
                        self.data.append({"prompt": prompt, "id": prompt_id})
                except json.JSONDecodeError:
                    print(f"Warning: skipping malformed JSONL line: {line}")
                    continue

        print(f"Loaded {len(self.data)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "id": item["id"],
        }


def assemble_gallery(images, rows, cols, target_size=None):
    """Assemble a grid, optionally resizing each image before placement."""
    assert len(images) >= rows * cols, "Not enough images provided!"
    if target_size is not None:
        images = [img.resize(target_size, Image.LANCZOS) for img in images]
        img_width, img_height = target_size
    else:
        img_width, img_height = images[0].size
    gallery = Image.new("RGB", (cols * img_width, rows * img_height))
    for row in range(rows):
        for col in range(cols):
            gallery.paste(images[row * cols + col], (col * img_width, row * img_height))
    return gallery
