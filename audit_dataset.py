import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_hash(path):
   
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(root):
    root = Path(root)
    if not root.is_dir():
        sys.exit(f"Folder not found: {root}")

    class_counts = Counter()
    corrupt = []
    sizes = Counter()
    hashes = defaultdict(list)

    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not class_dirs:
        sys.exit("No class subfolders found. Check the folder layout.")

    for class_dir in class_dirs:
        for path in class_dir.rglob("*"):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                with Image.open(path) as img:
                    img.verify()  # raises an error if the file is broken
                with Image.open(path) as img:
                    sizes[img.size] += 1
            except Exception:
                corrupt.append(path)
                continue
            class_counts[class_dir.name] += 1
            hashes[file_hash(path)].append(path)

    total = sum(class_counts.values())
    print(f"\nDataset: {root}")
    print(f"Classes: {len(class_counts)}   Images: {total}\n")

    print("Images per class:")
    for name in sorted(class_counts):
        print(f"  {name:<40} {class_counts[name]:>6}")

    counts = list(class_counts.values())
    if counts and max(counts) > 5 * min(counts):
        print("\nWARNING: classes are very imbalanced "
              f"(largest {max(counts)}, smallest {min(counts)}).")

    print("\nMost common image sizes (width x height):")
    for (w, h), n in sizes.most_common(5):
        print(f"  {w} x {h}: {n}")

    duplicates = [p for p in hashes.values() if len(p) > 1]
    extra = sum(len(p) - 1 for p in duplicates)
    print(f"\nExact duplicate files: {extra} extra copies "
          f"in {len(duplicates)} groups")

    print(f"Corrupt/unreadable files: {len(corrupt)}")
    for p in corrupt[:10]:
        print(f"  {p}")

    print("\nNote: this only finds IDENTICAL files. Near-duplicates "
          "(same leaf, slightly different photo) need a later step.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python audit_dataset.py path/to/dataset")
    audit(sys.argv[1])
