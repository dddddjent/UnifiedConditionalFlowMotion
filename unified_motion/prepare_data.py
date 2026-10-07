"""Import an existing prepared research dataset into the public manifest layout.

python -m unified_motion.prepare_data --source /path/to/prepared-export \
    --output data/paper --pair-end-offset 0

This imports precomputed features; it does not distribute or download dataset assets.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

from unified_motion.data import MotionRecord


def import_dataset(
    source: Path,
    output: Path,
    pair_end_offset: int,
    evaluator_metadata: Path | None,
    normalization_height: float,
) -> None:
    assert source.is_dir() and not output.exists(), "Choose a new output directory."
    assert math.isfinite(normalization_height) and normalization_height > 0
    if evaluator_metadata is not None:
        assert all((evaluator_metadata / f"{name}.npy").is_file() for name in ("mean", "std"))
    captions = json.loads((source / "all_caption_clean.json").read_text())
    split_root = source / "data_split_info"
    exports: dict[str, list[MotionRecord]] = {}
    motions, characters = set(), set()
    for split in ("train", "test"):
        records = []
        for identifier in (split_root / f"{split}_ids.txt").read_text().splitlines():
            identifier = identifier.strip()
            if not identifier:
                continue
            base, start, end = identifier.split("#")
            character = base.rsplit("__", 1)[1]
            entries = captions[identifier]
            descriptions = [
                text.strip() for field in ("manual", "gpt")
                for text in entries.get(field, []) if text.strip()
            ]
            assert descriptions, f"No captions for {identifier}."
            records.append(MotionRecord(
                id=identifier,
                motion=f"motions/{base}.npy",
                skeleton=f"skeletons/{character}.npy",
                source="smg" if character == "smg" else "mixamo",
                character=character,
                captions=descriptions,
                start=int(start),
                end=int(end),
            ))
            motions.add(base)
            characters.add(character)
        assert records and len({record.id for record in records}) == len(records)
        exports[split] = records
    assert {record.source for record in exports["train"]} == {"smg", "mixamo"}
    pairs = []
    known = {record.id for record in exports["test"]}
    for line in (split_root / "test_pair.txt").read_text().splitlines():
        if not line.strip():
            continue
        comma = line.find(",", line.find("#"))
        assert comma >= 0, f"Malformed pair: {line}"

        def pair_id(value: str) -> str:
            base, start, end = value.strip().split("#")
            return f"{base}#{start}#{int(end) + pair_end_offset}"

        first, second = pair_id(line[:comma]), pair_id(line[comma + 1:])
        assert first in known and second in known, (
            "Pair IDs must match test clips. Specify --pair-end-offset explicitly "
            "if the export uses a different end convention."
        )
        pairs.append({"source": first, "target": second})
    for directory in ("motions", "skeletons", "metadata", "splits"):
        (output / directory).mkdir(parents=True)
    for name in sorted(motions):
        shutil.copyfile(
            source / "renamed_feats" / f"{name}.npy", output / "motions" / f"{name}.npy"
        )
    for name in sorted(characters):
        shutil.copyfile(
            source / "char_feats" / f"{name}.npy", output / "skeletons" / f"{name}.npy"
        )
    statistics = (
        ("mean", "motion_mean"), ("std", "motion_std"),
        ("tpose_mean", "skeleton_mean"), ("tpose_std", "skeleton_std"),
    )
    for original, public in statistics:
        shutil.copyfile(
            source / "meta_data" / f"{original}.npy", output / "metadata" / f"{public}.npy"
        )
    if evaluator_metadata is not None:
        evaluator_output = output / "metadata" / "evaluator"
        evaluator_output.mkdir()
        for name in ("mean", "std"):
            shutil.copyfile(evaluator_metadata / f"{name}.npy", evaluator_output / f"{name}.npy")
    (output / "metadata" / "evaluation.json").write_text(
        json.dumps({"normalization_height": normalization_height}, indent=2) + "\n"
    )
    for split, records in exports.items():
        (output / "splits" / f"{split}.jsonl").write_text(
            "".join(json.dumps(asdict(record)) + "\n" for record in records)
        )
    (output / "splits" / "retarget-test.jsonl").write_text(
        "".join(json.dumps(pair) + "\n" for pair in pairs)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-end-offset", type=int, default=0)
    parser.add_argument("--evaluator-metadata", type=Path)
    parser.add_argument("--normalization-height", type=float, default=85.616624)
    args = parser.parse_args()
    import_dataset(
        args.source, args.output, args.pair_end_offset,
        args.evaluator_metadata, args.normalization_height,
    )


if __name__ == "__main__":
    main()
