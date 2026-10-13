#!/usr/bin/env python
"""Compare PI0.5 action chunks with dataset actions in an offline replay.

Errors are measured in the dataset's original action space, after model output
unnormalization and relative-to-absolute conversion.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import draccus
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import lerobot.policies.pi05.configuration_pi05  # noqa: F401 - registers PI0.5 with draccus
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import make_dataset
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.action_error_metrics import ActionErrorAccumulator
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.utils import init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True, help="Checkpoint pretrained_model directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("pi05_train_relative.yaml"),
        help="Training YAML used to locate and preprocess the replay dataset",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("offline_replay_metrics"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0, help="0 evaluates the complete dataset")
    parser.add_argument("--log-every", type=int, default=20, help="Log scalars every N replay batches")
    parser.add_argument("--heatmap-every", type=int, default=100, help="Upload/save heatmap every N batches")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic action noise")
    parser.add_argument("--no-swanlab", action="store_true", help="Disable SwanLab even if enabled in YAML")
    return parser.parse_args()


def load_config(path: Path) -> TrainPipelineConfig:
    with path.open(encoding="utf-8") as file:
        return draccus.load(TrainPipelineConfig, file)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    init_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.device = args.device
    cfg.policy = policy_cfg
    cfg.output_dir = args.output_dir
    cfg.job_name = f"{cfg.job_name}_offline_replay"
    cfg.resume = False
    cfg.wandb.run_id = None
    cfg.wandb.enable = cfg.wandb.enable and not args.no_swanlab

    dataset = make_dataset(cfg)
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(args.policy_path, config=policy_cfg).to(args.device).eval()
    replay_generator = torch.Generator(device=args.device)
    replay_generator.manual_seed(args.seed)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(args.policy_path),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=lerobot_collate_fn if dataset.meta.has_language_columns else None,
        persistent_workers=args.num_workers > 0,
    )
    logger = WandBLogger(cfg) if cfg.wandb.enable and cfg.wandb.project else None
    accumulator = ActionErrorAccumulator()

    total_batches = len(dataloader) if args.max_batches <= 0 else min(len(dataloader), args.max_batches)
    completed_batches = 0
    for batch_index, batch in enumerate(tqdm(dataloader, total=total_batches, desc="Offline replay"), start=1):
        if args.max_batches > 0 and batch_index > args.max_batches:
            break
        raw_target = batch["action"]
        action_is_pad = batch.get("action_is_pad")
        for camera_key in dataset.meta.camera_keys:
            if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                batch[camera_key] = batch[camera_key].float() / 255.0
        model_batch = preprocessor(batch)
        replay_noise = torch.randn(
            (raw_target.shape[0], policy_cfg.chunk_size, policy_cfg.max_action_dim),
            generator=replay_generator,
            device=args.device,
        )
        prediction = policy.predict_action_chunk(
            model_batch, noise=replay_noise, num_steps=args.num_inference_steps
        )
        prediction = postprocessor(prediction)
        accumulator.update(prediction, raw_target.to(prediction.device), action_is_pad)
        completed_batches = batch_index

        if args.log_every > 0 and batch_index % args.log_every == 0:
            accumulator.save(
                args.output_dir, prefix="replay", action_names=policy_cfg.action_feature_names
            )
            metrics = accumulator.scalar_dict()
            logging.info(
                "Replay batch %d: MAE=%.6f MSE=%.6f", batch_index, metrics["mae"], metrics["mse"]
            )
            if logger:
                logger.log_dict(metrics, batch_index, mode="replay")
        if args.heatmap_every > 0 and batch_index % args.heatmap_every == 0:
            heatmap = accumulator.save_heatmap(
                args.output_dir / "replay_mae_heatmap.png",
                action_names=policy_cfg.action_feature_names,
            )
            if logger:
                logger.log_image(heatmap, batch_index, mode="replay", key="mae_heatmap")

    if completed_batches == 0:
        raise RuntimeError("The replay dataset produced no batches")
    paths = accumulator.save(
        args.output_dir, prefix="replay", action_names=policy_cfg.action_feature_names
    )
    heatmap = accumulator.save_heatmap(
        args.output_dir / "replay_mae_heatmap.png", action_names=policy_cfg.action_feature_names
    )
    metrics = accumulator.scalar_dict()
    logging.info("Final offline replay metrics: MAE=%.6f MSE=%.6f", metrics["mae"], metrics["mse"])
    logging.info("Saved reports: %s", ", ".join(str(path) for path in [*paths.values(), heatmap]))
    if logger:
        logger.log_dict(metrics, completed_batches, mode="replay")
        logger.log_image(heatmap, completed_batches, mode="replay", key="mae_heatmap_final")
        logger.finish()


if __name__ == "__main__":
    main()
