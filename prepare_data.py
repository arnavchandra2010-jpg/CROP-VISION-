
import argparse
import csv
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))

# Table used to count the 1-bits in a byte (for comparing hashes quickly).
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def clean_name(name):
    """'Potato___Early_blight' -> 'potato_early_blight'"""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return name


def hash_to_int(h):
    """Turn an 8x8 ImageHash into one 64-bit integer."""
    bits = np.packbits(h.hash.flatten().astype(np.uint8))
    return int.from_bytes(bits.tobytes(), "big")


def eight_hashes(path):
    """Perceptual hash of the image in all 8 orientations
    (4 rotations x with/without mirror). This is what lets us catch
    rotated and flipped copies of the same leaf photo."""
    with Image.open(path) as img:
        img = img.convert("L")
        out = []
        for mirror in (False, True):
            base = img.transpose(Image.FLIP_LEFT_RIGHT) if mirror else img
            for angle in (0, 90, 180, 270):
                rotated = base.rotate(angle, expand=True) if angle else base
                out.append(hash_to_int(imagehash.phash(rotated)))
    return out


def popcount64(x):
    """Count 1-bits in each uint64 of an array."""
    b = x.view(np.uint8).reshape(*x.shape, 8)
    return _POPCOUNT[b].sum(axis=-1)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def group_near_duplicates(paths, threshold):
    """Return a list of group ids (one per path). Same id = near-duplicates."""
    n = len(paths)
    H = np.zeros((n, 8), dtype=np.uint64)
    keep = np.ones(n, dtype=bool)
    for i, p in enumerate(paths):
        try:
            H[i] = np.array(eight_hashes(p), dtype=np.uint64)
        except Exception as e:  # unreadable file
            print(f"  skipping unreadable file: {p} ({e})")
            keep[i] = False

    uf = UnionFind(n)
    for i in range(n):
        if not keep[i]:
            continue
        # distance from image i (as-is) to every image in all 8 orientations
        d = popcount64(H[i + 1:] ^ H[i, 0]).min(axis=1) if i + 1 < n else []
        for off in np.nonzero(np.asarray(d) <= threshold)[0]:
            j = i + 1 + int(off)
            if keep[j]:
                uf.union(i, j)
    return [uf.find(i) for i in range(n)], keep


def split_groups(groups, rng):
    """groups: dict group_id -> list of indices. Returns dict index -> split name.
    Fills train, then val, then test so each gets close to its target share."""
    ids = list(groups)
    rng.shuffle(ids)
    total = sum(len(groups[g]) for g in ids)
    targets = {name: frac * total for name, frac in SPLITS}
    counts = {name: 0 for name, _ in SPLITS}
    result = {}
    for g in ids:
        # give the group to the split that is furthest below its target
        name = max(counts, key=lambda s: targets[s] - counts[s])
        for idx in groups[g]:
            result[idx] = name
        counts[name] += len(groups[g])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="raw dataset folder (class subfolders)")
    ap.add_argument("--dst", required=True, help="new folder to create")
    ap.add_argument("--include", default="", help="keep only classes containing this text")
    ap.add_argument("--threshold", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    if not src.is_dir():
        sys.exit(f"Source folder not found: {src}")
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"Destination is not empty: {dst}\nChoose a new folder so nothing is overwritten.")

    class_dirs = sorted(d for d in src.iterdir() if d.is_dir()
                        and a.include.lower() in d.name.lower())
    if not class_dirs:
        sys.exit("No matching class folders found.")

    rng = random.Random(a.seed)
    rows = []  # (src_path, class, group, split)
    summary = {}

    for class_dir in class_dirs:
        cname = clean_name(class_dir.name)
        paths = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        print(f"{cname}: {len(paths)} images - checking for near-duplicates...")
        gid, keep = group_near_duplicates(paths, a.threshold)

        groups = defaultdict(list)
        for i, g in enumerate(gid):
            if keep[i]:
                groups[g].append(i)
        split_of = split_groups(groups, rng)

        multi = sum(len(v) for v in groups.values() if len(v) > 1)
        summary[cname] = {"images": len(split_of), "groups": len(groups),
                          "in_dup_groups": multi,
                          **{s: 0 for s, _ in SPLITS}}
        for i, split in split_of.items():
            summary[cname][split] += 1
            rows.append((paths[i], cname, f"{cname}_{gid[i]}", split))

    # copy files
    print("\nCopying files...")
    manifest = []
    used = set()
    for path, cname, g, split in rows:
        out_dir = dst / split / cname
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / path.name
        k = 1
        while out in used or out.exists():
            out = out_dir / f"{path.stem}_{k}{path.suffix}"
            k += 1
        used.add(out)
        shutil.copy2(path, out)
        manifest.append((str(out.relative_to(dst)), cname, g, split))

    with open(dst / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "group_id", "split"])
        w.writerows(manifest)

    print(f"\n{'class':<32}{'images':>7}{'groups':>8}{'in dup groups':>15}"
          f"{'train':>7}{'val':>6}{'test':>6}")
    for c, s in summary.items():
        print(f"{c:<32}{s['images']:>7}{s['groups']:>8}{s['in_dup_groups']:>15}"
              f"{s['train']:>7}{s['val']:>6}{s['test']:>6}")
    print(f"\nDone. Clean dataset written to: {dst}")
    print("Remember: this 'test' split is still lab-style data. "
          "Your real-field photos will be a separate test set.")


if __name__ == "__main__":
    main()
