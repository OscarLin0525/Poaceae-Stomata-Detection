#!/usr/bin/env python3
"""Create fixed validation/test splits with sampled train percentages.

The script rebuilds 1%, 5%, 10%, and 20% folders for BARLEY/WHEAT/RICE.
Validation and test sets are fixed across all percentage folders; only the
training subset changes. Existing percentage folders are moved into a timestamped
backup directory before new folders are written.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PCTS = (1, 5, 10, 20)
SPECIES = ("BARLEY", "WHEAT", "RICE")


@dataclass(frozen=True)
class Pair:
    stem: str
    image: Path
    label: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild species splits with fixed val/test and sampled train percentages."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset root containing BARLEY/WHEAT/RICE.",
    )
    parser.add_argument("--species", nargs="*", default=list(SPECIES), help="Species folders to process.")
    parser.add_argument("--train-pcts", nargs="*", type=int, default=list(PCTS), help="Train percentages to create.")
    parser.add_argument("--val-pct", type=float, default=5.0, help="Fixed validation percentage of all paired data.")
    parser.add_argument(
        "--max-train-pct",
        type=float,
        default=20.0,
        help="Largest train pool percentage. Test is fixed after holding this pool and val out.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument(
        "--backup-name",
        type=str,
        default="",
        help="Optional backup folder name under each species. Default: backup_fixed_eval_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Delete/rebuild percentage folders without moving old contents to backup.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned counts without writing files.",
    )
    return parser.parse_args()


def image_dirs_for_species(species_dir: Path, train_pcts: Sequence[int]) -> Iterable[Path]:
    percent_dirs = [species_dir / f"{pct}%" for pct in train_pcts if (species_dir / f"{pct}%").exists()]
    if percent_dirs:
        for pct_dir in percent_dirs:
            for split in ("train", "val", "test"):
                yield pct_dir / "images" / split
        return

    for split in ("train", "val", "test"):
        yield species_dir / "images" / split


def label_for_image(species_dir: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(species_dir)
    parts = list(rel.parts)
    try:
        image_idx = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Image path is not under an images directory: {image_path}") from exc
    parts[image_idx] = "labels"
    parts[-1] = f"{image_path.stem}.txt"
    return species_dir.joinpath(*parts)


def collect_pairs(species_dir: Path, train_pcts: Sequence[int]) -> List[Pair]:
    by_stem: Dict[str, Pair] = {}
    missing: List[Path] = []
    for img_dir in image_dirs_for_species(species_dir, train_pcts):
        if not img_dir.exists():
            continue
        for image_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            label_path = label_for_image(species_dir, image_path)
            if not label_path.exists():
                missing.append(image_path)
                continue
            by_stem.setdefault(image_path.stem, Pair(image_path.stem, image_path, label_path))
    if missing:
        preview = "\n".join(str(p) for p in missing[:8])
        raise FileNotFoundError(f"{species_dir.name}: {len(missing)} images are missing labels:\n{preview}")
    return [by_stem[k] for k in sorted(by_stem)]


def count_from_pct(total: int, pct: float) -> int:
    if pct <= 0.0:
        return 0
    return max(1, int(round(total * pct / 100.0)))


def copy_pairs(pairs: Sequence[Pair], out_dir: Path, split: str) -> None:
    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        shutil.copy2(pair.image, img_out / pair.image.name)
        shutil.copy2(pair.label, lbl_out / f"{pair.stem}.txt")


def backup_existing(species_dir: Path, pct_names: Sequence[str], backup_name: str, no_backup: bool) -> Path | None:
    existing = [species_dir / name for name in pct_names if (species_dir / name).exists()]
    if not existing:
        return None
    if no_backup:
        for path in existing:
            shutil.rmtree(path)
        return None
    backup_root = species_dir / backup_name
    backup_root.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), str(backup_root / path.name))
    return backup_root


def remap_pair_after_backup(pair: Pair, species_dir: Path, backup_root: Path | None, pct_names: Sequence[str]) -> Pair:
    if backup_root is None:
        return pair
    image = pair.image
    label = pair.label
    for name in pct_names:
        old_root = species_dir / name
        new_root = backup_root / name
        try:
            image_rel = image.relative_to(old_root)
            image = new_root / image_rel
        except ValueError:
            pass
        try:
            label_rel = label.relative_to(old_root)
            label = new_root / label_rel
        except ValueError:
            pass
    return Pair(pair.stem, image, label)


def remap_pairs_after_backup(
    pairs: Sequence[Pair],
    species_dir: Path,
    backup_root: Path | None,
    pct_names: Sequence[str],
) -> List[Pair]:
    return [remap_pair_after_backup(pair, species_dir, backup_root, pct_names) for pair in pairs]


def write_manifest(
    species_dir: Path,
    species: str,
    rows: List[Dict[str, object]],
    split_members: Dict[str, Sequence[Pair]],
) -> None:
    manifest = species_dir / "fixed_eval_split_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["species", "folder", "split", "count", "stems"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    for split_name, pairs in split_members.items():
        path = species_dir / f"fixed_{split_name}_stems.txt"
        path.write_text("\n".join(pair.stem for pair in pairs) + "\n", encoding="utf-8")


def rebuild_species(
    root: Path,
    species: str,
    train_pcts: Sequence[int],
    val_pct: float,
    max_train_pct: float,
    seed: int,
    backup_name: str,
    no_backup: bool,
    dry_run: bool,
) -> Dict[str, int]:
    species_dir = root / species
    pairs = collect_pairs(species_dir, train_pcts)
    total = len(pairs)
    if total == 0:
        raise RuntimeError(f"No paired data found for {species_dir}")

    max_train_count = count_from_pct(total, max_train_pct)
    val_count = count_from_pct(total, val_pct)
    if max_train_count + val_count >= total:
        raise ValueError(
            f"{species}: max_train_count + val_count must be smaller than total "
            f"({max_train_count} + {val_count} >= {total})"
        )

    rng = random.Random(f"{seed}:{species}")
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    train_pool = shuffled[:max_train_count]
    val_pairs = shuffled[max_train_count : max_train_count + val_count]
    test_pairs = shuffled[max_train_count + val_count :]

    pct_names = [f"{pct}%" for pct in train_pcts]
    if dry_run:
        print(
            f"{species}: total={total}, train_pool={len(train_pool)}, "
            f"val={len(val_pairs)}, test={len(test_pairs)}"
        )
        for pct in train_pcts:
            print(f"  {pct}% train={min(count_from_pct(total, pct), len(train_pool))}")
        return {"total": total, "val": len(val_pairs), "test": len(test_pairs)}

    backup_root = backup_existing(species_dir, pct_names, backup_name, no_backup)
    if backup_root is not None:
        print(f"{species}: old percentage folders moved to {backup_root}")
        train_pool = remap_pairs_after_backup(train_pool, species_dir, backup_root, pct_names)
        val_pairs = remap_pairs_after_backup(val_pairs, species_dir, backup_root, pct_names)
        test_pairs = remap_pairs_after_backup(test_pairs, species_dir, backup_root, pct_names)

    rows: List[Dict[str, object]] = []
    split_members: Dict[str, Sequence[Pair]] = {
        "val": val_pairs,
        "test": test_pairs,
        "train_pool_20pct": train_pool,
    }
    for pct in train_pcts:
        out_dir = species_dir / f"{pct}%"
        train_count = min(count_from_pct(total, pct), len(train_pool))
        train_pairs = train_pool[:train_count]
        copy_pairs(train_pairs, out_dir, "train")
        copy_pairs(val_pairs, out_dir, "val")
        copy_pairs(test_pairs, out_dir, "test")
        for split, split_pairs in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)):
            rows.append(
                {
                    "species": species,
                    "folder": f"{pct}%",
                    "split": split,
                    "count": len(split_pairs),
                    "stems": " ".join(pair.stem for pair in split_pairs),
                }
            )
        split_members[f"train_{pct}pct"] = train_pairs
        print(f"{species} {pct}%: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    write_manifest(species_dir, species, rows, split_members)
    return {"total": total, "val": len(val_pairs), "test": len(test_pairs)}


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    train_pcts = sorted(set(int(p) for p in args.train_pcts))
    backup_name = args.backup_name.strip() or f"backup_fixed_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"root={root}")
    print(f"train_pcts={train_pcts} val_pct={args.val_pct} max_train_pct={args.max_train_pct} seed={args.seed}")
    for species in args.species:
        rebuild_species(
            root=root,
            species=str(species).upper(),
            train_pcts=train_pcts,
            val_pct=float(args.val_pct),
            max_train_pct=float(args.max_train_pct),
            seed=int(args.seed),
            backup_name=backup_name,
            no_backup=bool(args.no_backup),
            dry_run=bool(args.dry_run),
        )


if __name__ == "__main__":
    main()
