import argparse
import json
from pathlib import Path


def validate_cached_splits(train_dir: Path, test_dir: Path) -> None:
    if train_dir.resolve() == test_dir.resolve():
        raise ValueError("training and validation caches must be different directories")
    manifests = []
    for split, directory in (("train", train_dir), ("test", test_dir)):
        path = directory / "split_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run prepare_dataset.sh before training")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("split") != split or not manifest.get("source_files"):
            raise ValueError(f"expected a non-empty {split} source manifest at {path}")
        if not (directory / "dataset_info.json").is_file():
            raise FileNotFoundError(f"dataset_info.json missing in {directory}")
        manifests.append(manifest)
    train_sources, test_sources = (
        {str(Path(f).resolve()) for f in manifest["source_files"]} for manifest in manifests
    )
    if train_sources & test_sources:
        raise ValueError("training and validation caches contain overlapping source files")
    for key in ("model", "image_max_token_num"):
        if manifests[0].get(key) != manifests[1].get(key):
            raise ValueError(f"training and validation caches have different {key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("test_dir", type=Path)
    args = parser.parse_args()
    validate_cached_splits(args.train_dir, args.test_dir)
