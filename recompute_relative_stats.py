#!/usr/bin/env python
"""Create the place dataset copy with stats for PI0.5 relative actions."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import recompute_stats


DEFAULT_SOURCE = Path(
    "/home/zhangyuqi/zhangyuqi/G1-20260911/G1_data/pick_place_9.11/only_pick"
)
DEFAULT_OUTPUT = Path(
    "/home/zhangyuqi/zhangyuqi/G1-20260911/G1_data/pick_place_9.11/only_pick_relative_stats_50"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def copy_with_hardlinks(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = output / item.name
        if item.is_dir():
            copy_with_hardlinks(item, target)
        elif item.is_symlink():
            target.symlink_to(os.readlink(item))
        elif item.suffix.lower() in {".parquet", ".mp4", ".avi", ".mov", ".webm"}:
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)
        else:
            shutil.copy2(item, target)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {source}")
    if source == output:
        raise ValueError("Source and output must be different directories.")
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output}; pass --force to replace it")
        logging.info("Removing existing output: %s", output)
        shutil.rmtree(output)

    logging.info("Hard-linking dataset: %s -> %s", source, output)
    copy_with_hardlinks(source, output)

    dataset = LeRobotDataset(repo_id=None, root=output)
    logging.info("Recomputing relative stats (chunk_size=%d)", args.chunk_size)
    recompute_stats(
        dataset,
        skip_image_video=True,
        relative_action=True,
        relative_exclude_joints=["gripper"],
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
    )
    logging.info("Done: %s", output / "meta" / "stats.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
