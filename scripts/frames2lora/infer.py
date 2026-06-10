import argparse
import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from ctx_to_lora.data.video_manifest import dump_jsonl, load_video_manifest
from scripts.frames2lora.train_smolvlm_stage1 import (
    TrainArgs,
    build_stage1_model,
    generate_fixed_examples,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Frames2LoRA inference from a Stage 1 checkpoint."
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Directory containing train_args.json and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint filename under RUN_DIR/checkpoints/ or a direct path.",
    )
    parser.add_argument(
        "--preset",
        default="auto",
        choices=("auto", "500m", "2.2b"),
        help="Use built-in public checkpoint settings when train_args.json is absent.",
    )
    parser.add_argument("--manifest", required=True, help="JSONL video manifest.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--video-size-longest-edge", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def checkpoint_path(run_dir: Path | None, checkpoint: str) -> Path:
    path = Path(checkpoint)
    if path.exists():
        return path
    if run_dir is None:
        return path
    return run_dir / "checkpoints" / checkpoint


def checkpoint_label(path: Path) -> str:
    match = re.search(r"step-(\d+)\.pt$", path.name)
    if match:
        return str(int(match.group(1)))
    return path.stem


def infer_preset(checkpoint: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lower = checkpoint.lower()
    if "500m" in lower:
        return "500m"
    if "2.2b" in lower or "2p2b" in lower or "22b" in lower:
        return "2.2b"
    raise ValueError(
        "Could not infer checkpoint preset from filename. Pass --preset 500m or --preset 2.2b."
    )


def default_train_args(preset: str) -> TrainArgs:
    if preset == "500m":
        model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        video_load_backend = "auto"
    elif preset == "2.2b":
        model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        video_load_backend = "auto"
    else:
        raise ValueError(f"Unknown preset: {preset}")
    return TrainArgs(
        smolvlm_name_or_path=model_name,
        train_manifest="",
        val_manifest="",
        val_core_manifest=None,
        val_gen_manifest=None,
        output_dir="",
        per_device_batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=0,
        learning_rate=0.0,
        weight_decay=0.0,
        warmup_ratio=0.0,
        max_grad_norm=1.0,
        seed=42,
        log_every=1,
        eval_every=1,
        save_every=1,
        max_train_samples=None,
        max_val_samples=None,
        num_workers=0,
        lora_r=16,
        lora_dropout=0.0,
        target_modules=["down_proj"],
        latent_size=512,
        dropout_rate=0.0,
        n_latent_queries=8,
        num_blocks=9,
        num_self_attn_per_block=0,
        video_fps=None,
        max_frames=12,
        video_size_longest_edge=384,
        video_load_backend=video_load_backend,
        internalization_prompt="Internalize this video for later captioning.",
        kl_weight=0.0,
        kl_temperature=1.0,
        generation_max_new_tokens=128,
        legacy_sanity_manifests=[],
        legacy_sanity_max_samples_per_manifest=0,
        wandb_project="frames2lora",
        wandb_mode="disabled",
        wandb_group=None,
        wandb_run_name=None,
        wandb_notes=None,
        resume_checkpoint=None,
        resume_trainer_state=None,
        resume_global_step=None,
        resume_ignore_scheduler_state=False,
    )


def load_train_args(run_dir: Path | None, overrides: argparse.Namespace) -> TrainArgs:
    args_path = run_dir / "train_args.json" if run_dir is not None else None
    if args_path is None or not args_path.exists():
        train_args = default_train_args(infer_preset(overrides.checkpoint, overrides.preset))
        if overrides.video_size_longest_edge is not None:
            train_args.video_size_longest_edge = overrides.video_size_longest_edge
        if overrides.generation_max_new_tokens is not None:
            train_args.generation_max_new_tokens = overrides.generation_max_new_tokens
        if overrides.seed is not None:
            train_args.seed = overrides.seed
        return train_args

    with open(args_path) as f:
        raw_args: dict[str, Any] = json.load(f)

    raw_args.setdefault("video_size_longest_edge", None)
    raw_args.setdefault("resume_ignore_scheduler_state", False)

    if overrides.video_size_longest_edge is not None:
        raw_args["video_size_longest_edge"] = overrides.video_size_longest_edge
    if overrides.generation_max_new_tokens is not None:
        raw_args["generation_max_new_tokens"] = overrides.generation_max_new_tokens
    if overrides.seed is not None:
        raw_args["seed"] = overrides.seed

    allowed = {field.name for field in fields(TrainArgs)}
    filtered_args = {key: value for key, value in raw_args.items() if key in allowed}
    missing = sorted(allowed - filtered_args.keys())
    if missing:
        raise ValueError(f"train_args.json is missing required fields: {missing}")
    return TrainArgs(**filtered_args)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else None
    ckpt_path = checkpoint_path(run_dir, args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    train_args = load_train_args(run_dir, args)
    set_seed(train_args.seed)
    rows = load_video_manifest(args.manifest, max_samples=args.max_samples)

    accelerator = Accelerator(mixed_precision="bf16")
    model, raw_model, processor, tokenizer = build_stage1_model(train_args)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(accelerator.device)
    raw_model.eval()

    outputs = generate_fixed_examples(
        accelerator,
        model,
        raw_model,
        processor,
        tokenizer,
        rows,
        train_args,
    )
    for row in outputs:
        row["checkpoint"] = checkpoint_label(ckpt_path)

    dump_jsonl(args.output, outputs)
    print(f"[infer] wrote {len(outputs)} rows to {args.output}")


if __name__ == "__main__":
    main()
