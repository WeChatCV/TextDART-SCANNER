#!/usr/bin/env python3
"""
Download the models used by the evaluation scripts.
Files go to $EVAL_MODEL_ROOT (default ~/.cache/textdart_eval).
This is an optional archive; the external upstream CVTG evaluator uses its
library-specific caches and does not automatically consume these files.

Models:
1. PaddleOCR models (CVTG OCR)
   - en_PP-OCRv3_det_infer
   - en_PP-OCRv4_rec_infer
   - ch_ppocr_mobile_v2.0_cls_infer
2. CLIP ViT-L/14 (loaded by the CVTG evaluation)
3. OpenCLIP ViT-L-14 weights (loaded by the CVTG evaluation)

Qwen/Qwen2.5-VL-7B-Instruct (LongText + OneIG OCR) is downloaded automatically
by transformers from $QWEN25VL_PATH (default: HF model id) and is not handled here.
"""

import http.client
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

# ============ Config ============ #

# Root directory for downloaded models
MODEL_ROOT = os.environ.get("EVAL_MODEL_ROOT", os.path.expanduser("~/.cache/textdart_eval"))

# PaddleOCR model directory
PADDLEOCR_DIR = os.path.join(MODEL_ROOT, "paddleocr_rewards")

# Optional HuggingFace endpoint override (empty string = huggingface.co)
HF_MIRROR = os.environ.get("HF_ENDPOINT", "")

# ============ PaddleOCR model URLs ============ #

PADDLEOCR_MODELS = {
    "whl/det/en/en_PP-OCRv3_det_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
    "whl/rec/en/en_PP-OCRv4_rec_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
    "whl/cls/ch_ppocr_mobile_v2.0_cls_infer": (
        "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
    ),
}

# ============ OpenCLIP model ============ #

# OpenCLIP ViT-L-14 weights (used for the Aesthetic Score), hosted on HuggingFace
OPENCLIP_MODEL_URL = (
    "https://huggingface.co/timm/vit_large_patch14_clip_224.openai/resolve/main/open_clip_pytorch_model.bin"
)
OPENCLIP_SAVE_DIR = os.path.join(MODEL_ROOT, "openai_clip_timm")
OPENCLIP_SAVE_PATH = os.path.join(OPENCLIP_SAVE_DIR, "open_clip_pytorch_model.bin")


def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def print_step(msg):
    print(f"\n  → {msg}")


def download_file(url, save_path, desc=""):
    """Download a file (resumable via wget, falling back to urllib). Returns True on success."""
    if os.path.exists(save_path):
        print(f"    ✅ Exists, skipping: {save_path}")
        return True

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"    ⬇️  Downloading: {desc or url}")
    print(f"       Saving to: {save_path}")

    try:
        subprocess.run(["wget", "-c", "-q", "--show-progress", "-O", save_path, url], check=True)
        print(f"    ✅ Downloaded: {save_path}")
        return True
    except subprocess.CalledProcessError:
        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"    ✅ Downloaded: {save_path}")
            return True
        except (OSError, ValueError, http.client.HTTPException) as e:
            print(f"    ❌ Download failed: {e}")
            return False


def download_and_extract_tar(url, extract_to, target_subdir):
    """Download a tar archive and extract its contents into extract_to/target_subdir."""
    target_path = os.path.join(extract_to, target_subdir)

    if os.path.exists(target_path) and os.listdir(target_path):
        print(f"    ✅ Exists, skipping: {target_path}")
        return True

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    tar_filename = url.split("/")[-1]
    tar_path = os.path.join("/tmp", tar_filename)

    print(f"    ⬇️  Downloading: {tar_filename}")
    try:
        subprocess.run(["wget", "-c", "-q", "--show-progress", "-O", tar_path, url], check=True)
    except subprocess.CalledProcessError:
        try:
            urllib.request.urlretrieve(url, tar_path)
        except (OSError, ValueError, http.client.HTTPException) as e:
            print(f"    ❌ Download failed: {e}")
            return False

    print(f"    📦 Extracting: {tar_filename} → {target_path}")
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            tmp_extract = os.path.join("/tmp", "paddleocr_extract")
            if os.path.exists(tmp_extract):
                shutil.rmtree(tmp_extract)
            tar.extractall(tmp_extract)

            # Use the single top-level directory of the archive if there is one
            extracted_items = os.listdir(tmp_extract)
            if len(extracted_items) == 1 and os.path.isdir(
                os.path.join(tmp_extract, extracted_items[0])
            ):
                src_dir = os.path.join(tmp_extract, extracted_items[0])
            else:
                src_dir = tmp_extract

            os.makedirs(target_path, exist_ok=True)
            for item in os.listdir(src_dir):
                src = os.path.join(src_dir, item)
                dst = os.path.join(target_path, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)

            shutil.rmtree(tmp_extract, ignore_errors=True)

        os.remove(tar_path)
        print(f"    ✅ Extracted: {target_path}")
        return True
    except (OSError, tarfile.TarError, shutil.Error, ValueError) as e:
        print(f"    ❌ Extraction failed: {e}")
        return False


def download_clip_model():
    """Download the CLIP ViT-L/14 checkpoint (same file the `clip` package fetches)."""
    clip_path = os.path.join(MODEL_ROOT, "ViT-L-14.pt")
    if os.path.exists(clip_path):
        print(f"    ✅ Exists, skipping: {clip_path}")
        return True

    clip_url = (
        "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a"
        "03836/ViT-L-14.pt"
    )
    return download_file(clip_url, clip_path, desc="CLIP ViT-L/14")


def main():
    print_header("Evaluation model downloader")
    print(f"  Model root: {MODEL_ROOT}")
    print(f"  HF endpoint: {HF_MIRROR or 'not set'}")

    os.makedirs(MODEL_ROOT, exist_ok=True)

    success_count = 0
    fail_count = 0

    # ============ 1. PaddleOCR models ============ #
    print_header("1/3 PaddleOCR models (CVTG)")
    for subdir, url in PADDLEOCR_MODELS.items():
        if download_and_extract_tar(url, PADDLEOCR_DIR, subdir):
            success_count += 1
        else:
            fail_count += 1

    # ============ 2. CLIP ViT-L/14 ============ #
    print_header("2/3 CLIP ViT-L/14 (CVTG CLIPScore)")
    if download_clip_model():
        success_count += 1
    else:
        fail_count += 1

    # ============ 3. OpenCLIP weights ============ #
    print_header("3/3 OpenCLIP ViT-L-14 weights (CVTG Aesthetic Score)")
    openclip_url = OPENCLIP_MODEL_URL
    if HF_MIRROR:
        openclip_url = openclip_url.replace("https://huggingface.co", HF_MIRROR)
    if download_file(openclip_url, OPENCLIP_SAVE_PATH, desc="OpenCLIP ViT-L-14"):
        success_count += 1
    else:
        fail_count += 1

    # ============ Summary ============ #
    print_header("Download summary")
    print(f"  ✅ Succeeded: {success_count}")
    print(f"  ❌ Failed: {fail_count}")
    print()
    print("  Model locations:")
    print(f"    PaddleOCR:     {PADDLEOCR_DIR}/")
    print(f"    CLIP ViT-L/14: {MODEL_ROOT}/ViT-L-14.pt")
    print(f"    OpenCLIP:      {OPENCLIP_SAVE_PATH}")
    print()

    if fail_count > 0:
        print("  ⚠️  Some downloads failed; check your network and re-run this script")
        sys.exit(1)


if __name__ == "__main__":
    main()
