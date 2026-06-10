import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
try:
    import wandb
except ModuleNotFoundError:
    class _MissingWandb:
        def init(self, *args, **kwargs):
            if kwargs.get("mode") != "disabled":
                raise ModuleNotFoundError(
                    "wandb is not installed. Install it or pass --wandb-mode disabled."
                )

        def log(self, *args, **kwargs):
            return None

        def finish(self, *args, **kwargs):
            return None

    wandb = _MissingWandb()
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PretrainedConfig,
    get_cosine_schedule_with_warmup,
)

from ctx_to_lora.configs import (
    AggregatorArguments,
    CtxEncoderArguments,
    HypernetArguments,
)
from ctx_to_lora.data.video_manifest import (
    DATA_ROOT,
    DEFAULT_INTERNALIZATION_PROMPT,
    dump_jsonl,
    load_video_manifest,
    normalize_manifest_row,
    resolve_video_path,
)
from ctx_to_lora.model_loading import get_lora_config
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel, get_hypernet_config
from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers
from ctx_to_lora.modeling.lora_merger import combine_lora
from scripts.frames2lora.train_smolvlm_online import (
    extract_l2l_fused_text_features,
    prepare_smolvlm_inputs,
    prepare_smolvlm_teacher_batch,
)


def debug_rank_log(accelerator: Accelerator | None, message: str) -> None:
    if os.environ.get("FRAMES2LORA_DEBUG_STARTUP", "").lower() not in {"1", "true", "yes"}:
        return
    rank = "na"
    if accelerator is not None:
        try:
            rank = str(accelerator.process_index)
        except Exception:  # pragma: no cover - defensive debug helper
            rank = "unknown"
    line = f"[debug rank={rank} pid={os.getpid()}] {message}"
    print(line, flush=True)
    try:
        with open(f"/tmp/frames2lora_startup_rank_{rank}_pid_{os.getpid()}.log", "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


@dataclass
class TrainArgs:
    smolvlm_name_or_path: str
    train_manifest: str
    val_manifest: str
    val_core_manifest: str | None
    val_gen_manifest: str | None
    output_dir: str
    per_device_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    seed: int
    log_every: int
    eval_every: int
    save_every: int
    max_train_samples: int | None
    max_val_samples: int | None
    num_workers: int
    lora_r: int
    lora_dropout: float
    target_modules: list[str]
    latent_size: int
    dropout_rate: float
    n_latent_queries: int
    num_blocks: int
    num_self_attn_per_block: int
    video_fps: float | None
    max_frames: int
    video_size_longest_edge: int | None
    video_load_backend: str
    internalization_prompt: str
    kl_weight: float
    kl_temperature: float
    generation_max_new_tokens: int
    legacy_sanity_manifests: list[str]
    legacy_sanity_max_samples_per_manifest: int
    wandb_project: str
    wandb_mode: str
    wandb_group: str | None
    wandb_run_name: str | None
    wandb_notes: str | None
    resume_checkpoint: str | None
    resume_trainer_state: str | None
    resume_global_step: int | None
    resume_ignore_scheduler_state: bool


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(
        description="Stage 1 SmolVLM caption pretraining with generated LoRA."
    )
    parser.add_argument(
        "--smolvlm-name-or-path",
        default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    )
    parser.add_argument(
        "--train-manifest",
        default=str(DATA_ROOT / "processed" / "finevideo" / "train.teacher_visual_smolvlm.jsonl"),
    )
    parser.add_argument(
        "--val-manifest",
        default=str(DATA_ROOT / "processed" / "finevideo" / "val.teacher_visual_smolvlm.jsonl"),
    )
    parser.add_argument(
        "--val-core-manifest",
        default="",
    )
    parser.add_argument(
        "--val-gen-manifest",
        default="",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=60000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", default="down_proj")
    parser.add_argument("--latent-size", type=int, default=512)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--n-latent-queries", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=9)
    parser.add_argument("--num-self-attn-per-block", type=int, default=0)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument(
        "--video-size-longest-edge",
        type=int,
        default=None,
        help="Optional SmolVLM processor resize longest edge for video frames.",
    )
    parser.add_argument(
        "--video-load-backend",
        default="auto",
        choices=("auto", "decord", "pyav", "opencv", "torchvision"),
        help="Video decode backend for processor.apply_chat_template.",
    )
    parser.add_argument(
        "--internalization-prompt",
        default=DEFAULT_INTERNALIZATION_PROMPT,
    )
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--generation-max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--legacy-sanity-manifest",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--legacy-sanity-max-samples-per-manifest",
        type=int,
        default=16,
    )
    parser.add_argument("--wandb-project", default="frames2lora-stage1")
    parser.add_argument(
        "--wandb-mode",
        default="auto",
        choices=("auto", "online", "offline", "disabled"),
    )
    parser.add_argument("--wandb-group", default="finevideo-stage1")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-notes", default=None)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-trainer-state", default="")
    parser.add_argument("--resume-global-step", type=int, default=None)
    parser.add_argument(
        "--resume-ignore-scheduler-state",
        action="store_true",
        help="Resume optimizer/global_step/RNG but rebuild the LR scheduler from step count instead of loading its saved state.",
    )
    parsed = parser.parse_args()

    output_dir = parsed.output_dir
    if not output_dir:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output_dir = str(DATA_ROOT / "runs" / f"{timestamp}-smolvlm-stage1")

    def maybe_existing(path: str) -> str | None:
        if path and os.path.exists(path):
            return path
        return None

    return TrainArgs(
        smolvlm_name_or_path=parsed.smolvlm_name_or_path,
        train_manifest=parsed.train_manifest,
        val_manifest=parsed.val_manifest,
        val_core_manifest=maybe_existing(parsed.val_core_manifest),
        val_gen_manifest=maybe_existing(parsed.val_gen_manifest),
        output_dir=output_dir,
        per_device_batch_size=parsed.per_device_batch_size,
        eval_batch_size=parsed.eval_batch_size,
        gradient_accumulation_steps=parsed.gradient_accumulation_steps,
        max_steps=parsed.max_steps,
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        warmup_ratio=parsed.warmup_ratio,
        max_grad_norm=parsed.max_grad_norm,
        seed=parsed.seed,
        log_every=parsed.log_every,
        eval_every=parsed.eval_every,
        save_every=parsed.save_every,
        max_train_samples=parsed.max_train_samples,
        max_val_samples=parsed.max_val_samples,
        num_workers=parsed.num_workers,
        lora_r=parsed.lora_r,
        lora_dropout=parsed.lora_dropout,
        target_modules=[x.strip() for x in parsed.target_modules.split(",") if x.strip()],
        latent_size=parsed.latent_size,
        dropout_rate=parsed.dropout_rate,
        n_latent_queries=parsed.n_latent_queries,
        num_blocks=parsed.num_blocks,
        num_self_attn_per_block=parsed.num_self_attn_per_block,
        video_fps=parsed.video_fps,
        max_frames=parsed.max_frames,
        video_size_longest_edge=parsed.video_size_longest_edge,
        video_load_backend=parsed.video_load_backend,
        internalization_prompt=parsed.internalization_prompt,
        kl_weight=parsed.kl_weight,
        kl_temperature=parsed.kl_temperature,
        generation_max_new_tokens=parsed.generation_max_new_tokens,
        legacy_sanity_manifests=parsed.legacy_sanity_manifest,
        legacy_sanity_max_samples_per_manifest=parsed.legacy_sanity_max_samples_per_manifest,
        wandb_project=parsed.wandb_project,
        wandb_mode=parsed.wandb_mode,
        wandb_group=parsed.wandb_group,
        wandb_run_name=parsed.wandb_run_name,
        wandb_notes=parsed.wandb_notes,
        resume_checkpoint=maybe_existing(parsed.resume_checkpoint),
        resume_trainer_state=maybe_existing(parsed.resume_trainer_state),
        resume_global_step=parsed.resume_global_step,
        resume_ignore_scheduler_state=parsed.resume_ignore_scheduler_state,
    )


def ensure_layout(output_dir: Path) -> None:
    for path in (
        DATA_ROOT,
        DATA_ROOT / "raw",
        DATA_ROOT / "processed",
        DATA_ROOT / "runs",
        output_dir,
        output_dir / "checkpoints",
        output_dir / "generations",
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_wandb_mode(requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    if os.environ.get("WANDB_DISABLED", "").lower() in {"true", "1", "yes"}:
        return "disabled"
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    return "offline"


def build_labels(tokenizer, prompt: str, target_text: str):
    prompt_messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    full_messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    )
    input_ids = torch.tensor(full_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[: len(prompt_ids)] = -100
    return input_ids, attention_mask, labels


class VideoCaptionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], *, internalization_prompt: str):
        self.rows = rows
        self.internalization_prompt = internalization_prompt

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "id": row["id"],
            "video_path": resolve_video_path(row["video_path"]),
            "prompt": row["prompt"],
            "target_text": row["target_text"],
            "dataset": row["dataset"],
            "task_type": row["task_type"],
            "split": row.get("split"),
            "metadata": row.get("metadata", {}),
            "internalization_prompt": self.internalization_prompt,
        }


class Stage1Collator:
    def __init__(self, tokenizer, video_fps: float | None, max_frames: int):
        self.tokenizer = tokenizer
        self.video_fps = video_fps
        self.max_frames = max_frames

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids, attention_masks, labels = [], [], []
        prompt_only_ids, prompt_only_masks = [], []
        video_messages = []
        teacher_prompt_messages = []
        teacher_full_messages = []

        for example in batch:
            inp_ids, attn_mask, lbl = build_labels(
                self.tokenizer,
                prompt=example["prompt"],
                target_text=example["target_text"],
            )
            prompt_ids = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": example["prompt"]}],
                    }
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
            )
            prompt_only_ids.append(torch.tensor(prompt_ids, dtype=torch.long))
            prompt_only_masks.append(torch.ones(len(prompt_ids), dtype=torch.long))
            input_ids.append(inp_ids)
            attention_masks.append(attn_mask)
            labels.append(lbl)
            video_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": example["video_path"]},
                            {
                                "type": "text",
                                "text": example["internalization_prompt"],
                            },
                        ],
                    }
                ]
            )
            teacher_prompt_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": example["video_path"]},
                            {"type": "text", "text": example["prompt"]},
                        ],
                    }
                ]
            )
            teacher_full_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": example["video_path"]},
                            {"type": "text", "text": example["prompt"]},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": example["target_text"]}],
                    },
                ]
            )

        batch_input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch_attention_mask = pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )
        batch_labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        batch_prompt_ids = pad_sequence(
            prompt_only_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch_prompt_masks = pad_sequence(
            prompt_only_masks,
            batch_first=True,
            padding_value=0,
        )

        return {
            "ids": [example["id"] for example in batch],
            "datasets": [example["dataset"] for example in batch],
            "task_types": [example["task_type"] for example in batch],
            "prompts": [example["prompt"] for example in batch],
            "targets": [example["target_text"] for example in batch],
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
            "prompt_only_input_ids": batch_prompt_ids,
            "prompt_only_attention_mask": batch_prompt_masks,
            "video_messages": video_messages,
            "teacher_prompt_messages": teacher_prompt_messages,
            "teacher_full_messages": teacher_full_messages,
            "video_fps": self.video_fps,
            "max_frames": self.max_frames,
        }


def move_text_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def build_stage1_model(args: TrainArgs, device: torch.device | None = None):
    processor = AutoProcessor.from_pretrained(
        args.smolvlm_name_or_path,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_model = AutoModelForImageTextToText.from_pretrained(
        args.smolvlm_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if device is not None:
        raw_model = raw_model.to(device)

    peft_config = get_lora_config(
        args.smolvlm_name_or_path,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    peft_model = PeftModel(raw_model, peft_config)
    for param in peft_model.parameters():
        param.requires_grad = False

    raw_model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(raw_model, "generation_config", None):
        raw_model.generation_config.pad_token_id = tokenizer.pad_token_id

    hypernet_args = HypernetArguments(
        latent_size=args.latent_size,
        dropout_rate=args.dropout_rate,
        per_rank_gen=True,
    )
    aggregator_args = AggregatorArguments(
        aggregator_type="perceiver",
        n_latent_queries=args.n_latent_queries,
        num_blocks=args.num_blocks,
        num_self_attn_per_block=args.num_self_attn_per_block,
    )
    ctx_encoder_args = CtxEncoderArguments(
        ctx_encoder_model_name_or_path="precomputed",
        ctx_encoder_type="per_layer_activations",
    )
    ctx_config = PretrainedConfig(hidden_size=raw_model.config.text_config.hidden_size)
    hypernet_config = get_hypernet_config(
        peft_model,
        ctx_config,
        hypernet_args,
        aggregator_args,
        ctx_encoder_args,
    )
    model = ModulatedPretrainedModel(
        peft_model,
        hypernet_config,
        ctx_encoder_args,
        use_sequence_packing=False,
    )
    model.train()
    model.base_model.eval()
    raw_model.eval()
    return model, raw_model, processor, tokenizer


def compute_ce_loss(logits: torch.Tensor, labels: torch.Tensor):
    shift_labels = nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    vocab_size = logits.shape[-1]
    token_losses = nn.functional.cross_entropy(
        logits.float().view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
    ).view(labels.shape[0], -1)
    token_mask = shift_labels.ne(-100)
    per_sample_loss = (token_losses * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
    return per_sample_loss.mean(), per_sample_loss, token_mask.sum(dim=1)


def compute_teacher_kl_loss(
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
    *,
    temperature: float,
):
    shared_vocab_size = min(student_logits.shape[-1], teacher_logits.shape[-1])
    student_shift_labels = nn.functional.pad(student_labels, (0, 1), value=-100)[
        ..., 1:
    ].contiguous()
    teacher_shift_labels = nn.functional.pad(teacher_labels, (0, 1), value=-100)[
        ..., 1:
    ].contiguous()
    student_shift_logits = student_logits[..., :shared_vocab_size]
    teacher_shift_logits = teacher_logits[..., :shared_vocab_size]

    per_sample_kl = []
    for sample_idx in range(student_shift_logits.shape[0]):
        student_mask = student_shift_labels[sample_idx].ne(-100)
        teacher_mask = teacher_shift_labels[sample_idx].ne(-100)
        student_sample_logits = student_shift_logits[sample_idx][student_mask]
        teacher_sample_logits = teacher_shift_logits[sample_idx][teacher_mask]
        if student_sample_logits.shape[0] == 0 or teacher_sample_logits.shape[0] == 0:
            per_sample_kl.append(student_logits.new_zeros(()))
            continue
        if student_sample_logits.shape[0] != teacher_sample_logits.shape[0]:
            min_len = min(student_sample_logits.shape[0], teacher_sample_logits.shape[0])
            student_sample_logits = student_sample_logits[:min_len]
            teacher_sample_logits = teacher_sample_logits[:min_len]
        teacher_logp = nn.functional.log_softmax(
            teacher_sample_logits.float() / temperature,
            dim=-1,
        )
        student_logp = nn.functional.log_softmax(
            student_sample_logits.float() / temperature,
            dim=-1,
        )
        teacher_p = teacher_logp.exp()
        kl_tokens = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)
        per_sample_kl.append(kl_tokens.mean())

    per_sample_kl = torch.stack(per_sample_kl)
    return per_sample_kl.mean() * (temperature**2), per_sample_kl * (temperature**2)


def extract_stage1_ctx_features(
    raw_model,
    processor,
    batch,
    device,
    num_target_layers: int,
    video_size_longest_edge: int | None = None,
    video_load_backend: str = "auto",
):
    vlm_inputs = prepare_smolvlm_inputs(
        processor,
        batch["video_messages"],
        device,
        video_fps=batch["video_fps"],
        max_frames=batch["max_frames"],
        video_size_longest_edge=video_size_longest_edge,
        video_load_backend=video_load_backend,
    )
    return extract_l2l_fused_text_features(
        raw_model,
        vlm_inputs,
        num_target_layers=num_target_layers,
    )


def set_model_train_state(model, raw_model) -> None:
    model.train()
    model.base_model.eval()
    raw_model.eval()


def disable_lora_hooks(model) -> None:
    model.reset()


def enable_lora_hooks(model) -> None:
    model.patch_lora_forward()


def _iter_exception_chain(exc: BaseException):
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _is_skippable_video_error(exc: BaseException) -> bool:
    for err in _iter_exception_chain(exc):
        if isinstance(err, UnicodeDecodeError):
            return True
        # PyAV/transformers occasionally raises bare IndexError for malformed videos.
        if isinstance(err, IndexError) and "tuple index out of range" in str(err).lower():
            return True
        # PyAV can also surface empty-decode failures as a plain ValueError.
        if isinstance(err, ValueError) and "need at least one array to stack" in str(err).lower():
            return True
        err_name = err.__class__.__name__.lower()
        err_mod = err.__class__.__module__.lower()
        msg = str(err).lower()
        if "decord" in err_mod or "decorderror" in err_name:
            return True
        if "av." in err_mod or "ffmpeg" in err_mod or "pyav" in err_mod:
            return True
        if "invalid total_num_frames" in msg:
            return True
        if "cannot open" in msg and "video" in msg:
            return True
        if "error opening input file" in msg:
            return True
        if "unable to handle eof" in msg:
            return True
        if "thread worker: error sending packet" in msg:
            return True
        if "cannot find video stream" in msg:
            return True
        if "moov atom not found" in msg:
            return True
        if "invalid data found when processing input" in msg:
            return True
        if "unsupported codec" in msg:
            return True
        if "decode" in err_name and "error" in err_name:
            return True
    return False


@torch.no_grad()
def evaluate_loss(
    accelerator: Accelerator,
    model,
    raw_model,
    processor,
    data_loader: DataLoader,
    device: torch.device,
    args: TrainArgs,
) -> dict[str, float]:
    unwrapped = accelerator.unwrap_model(model)
    disable_lora_hooks(unwrapped)
    unwrapped.eval()
    unwrapped.base_model.eval()
    raw_model.eval()

    total_samples = torch.zeros(1, device=device)
    total_ce = torch.zeros(1, device=device)
    total_kl = torch.zeros(1, device=device)
    total_combined = torch.zeros(1, device=device)
    total_answer_tokens = torch.zeros(1, device=device)
    total_context_tokens = torch.zeros(1, device=device)
    skipped_batches = torch.zeros(1, device=device)
    skipped_samples = torch.zeros(1, device=device)

    for batch in data_loader:
        local_skip = False
        text_batch = None
        ctx_features = None
        ctx_attn_mask = None
        ctx_position_ids = None
        outputs = None
        teacher_batch = None
        teacher_outputs = None
        local_error = ""
        try:
            text_batch = move_text_batch_to_device(batch, device)
            disable_lora_hooks(unwrapped)
            ctx_features, ctx_attn_mask, ctx_position_ids = extract_stage1_ctx_features(
                raw_model,
                processor,
                batch,
                device,
                num_target_layers=unwrapped.hypernet.n_layers,
                video_size_longest_edge=args.video_size_longest_edge,
                video_load_backend=args.video_load_backend,
            )
            enable_lora_hooks(unwrapped)
            outputs = model(
                ctx_features=ctx_features,
                ctx_attn_mask=ctx_attn_mask,
                ctx_position_ids=ctx_position_ids,
                n_ctx_chunks=torch.ones(ctx_features.shape[0], dtype=torch.int32, device=device),
                input_ids=text_batch["input_ids"],
                attention_mask=text_batch["attention_mask"],
                labels=text_batch["labels"],
            )
            disable_lora_hooks(unwrapped)
            if args.kl_weight > 0:
                teacher_batch = prepare_smolvlm_teacher_batch(
                    processor,
                    batch["teacher_prompt_messages"],
                    batch["teacher_full_messages"],
                    device,
                    video_fps=batch["video_fps"],
                    max_frames=batch["max_frames"],
                    video_size_longest_edge=args.video_size_longest_edge,
                    video_load_backend=args.video_load_backend,
                )
                teacher_outputs = raw_model(
                    input_ids=teacher_batch["input_ids"],
                    attention_mask=teacher_batch.get("attention_mask"),
                    pixel_values=teacher_batch.get("pixel_values"),
                    pixel_attention_mask=teacher_batch.get("pixel_attention_mask"),
                    labels=None,
                    return_dict=True,
                    use_cache=False,
                )
        except Exception as e:  # pylint: disable=broad-except
            disable_lora_hooks(unwrapped)
            if _is_skippable_video_error(e):
                local_skip = True
                local_error = f"{type(e).__name__}: {e}"
            else:
                raise

        local_skip_tensor = torch.tensor([1 if local_skip else 0], device=device, dtype=torch.int32)
        global_skip_tensor = accelerator.reduce(local_skip_tensor, reduction="sum")
        if global_skip_tensor.item() > 0:
            skipped_batches += 1
            skipped_samples += torch.tensor([len(batch["ids"])], device=device, dtype=torch.float32)
            if local_skip and accelerator.is_local_main_process:
                print(f"[eval] skipped batch due to video decode error: {local_error}", flush=True)
            continue

        assert text_batch is not None
        assert outputs is not None
        assert ctx_attn_mask is not None
        ce_loss, per_sample_ce, answer_tokens = compute_ce_loss(outputs.logits, text_batch["labels"])
        if args.kl_weight > 0:
            assert teacher_batch is not None
            assert teacher_outputs is not None
            _, per_sample_kl = compute_teacher_kl_loss(
                outputs.logits,
                text_batch["labels"],
                teacher_outputs.logits,
                teacher_batch["labels"],
                temperature=args.kl_temperature,
            )
        else:
            per_sample_kl = torch.zeros_like(per_sample_ce)

        batch_size = torch.tensor([per_sample_ce.shape[0]], device=device, dtype=torch.float32)
        total_samples += batch_size
        total_ce += per_sample_ce.sum()
        total_kl += per_sample_kl.sum()
        total_combined += (per_sample_ce + args.kl_weight * per_sample_kl).sum()
        total_answer_tokens += answer_tokens.sum().to(dtype=torch.float32)
        total_context_tokens += ctx_attn_mask.sum().to(dtype=torch.float32)
        disable_lora_hooks(unwrapped)

    total_samples = accelerator.reduce(total_samples, reduction="sum")
    total_ce = accelerator.reduce(total_ce, reduction="sum")
    total_kl = accelerator.reduce(total_kl, reduction="sum")
    total_combined = accelerator.reduce(total_combined, reduction="sum")
    total_answer_tokens = accelerator.reduce(total_answer_tokens, reduction="sum")
    total_context_tokens = accelerator.reduce(total_context_tokens, reduction="sum")
    skipped_batches = accelerator.reduce(skipped_batches, reduction="sum")
    skipped_samples = accelerator.reduce(skipped_samples, reduction="sum")

    metrics = {
        "ce_loss": (total_ce / total_samples.clamp_min(1)).item(),
        "kl_loss": (total_kl / total_samples.clamp_min(1)).item(),
        "combined_loss": (total_combined / total_samples.clamp_min(1)).item(),
        "samples": total_samples.item(),
        "skipped_batches": skipped_batches.item(),
        "skipped_samples": skipped_samples.item(),
        "answer_tokens_per_sample": (total_answer_tokens / total_samples.clamp_min(1)).item(),
        "context_tokens_per_sample": (total_context_tokens / total_samples.clamp_min(1)).item(),
    }
    set_model_train_state(unwrapped, raw_model)
    return metrics


def save_checkpoint(accelerator: Accelerator, model, output_dir: Path, filename: str) -> Path:
    ckpt_path = output_dir / "checkpoints" / filename
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state_dict = accelerator.unwrap_model(model).state_dict()
        torch.save(state_dict, ckpt_path)
    accelerator.wait_for_everyone()
    return ckpt_path


def trainer_state_path(output_dir: Path, filename: str) -> Path:
    stem = Path(filename).stem
    return output_dir / "checkpoints" / f"{stem}.trainer.pt"


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    python_state = state.get("python_random_state")
    if python_state is not None:
        random.setstate(python_state)
    torch_state = state.get("torch_rng_state")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_trainer_state(
    accelerator: Accelerator,
    optimizer,
    scheduler,
    output_dir: Path,
    filename: str,
    *,
    global_step: int,
    best_ce: float,
    best_combined: float,
    train_batches_seen: int,
) -> Path:
    state_path = trainer_state_path(output_dir, filename)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "best_ce": best_ce,
            "best_combined": best_combined,
            "train_batches_seen": train_batches_seen,
            "rng_state": capture_rng_state(),
            "model_checkpoint_filename": filename,
        }
        torch.save(state, state_path)
    accelerator.wait_for_everyone()
    return state_path


def write_generation_artifacts(output_dir: Path, step: int, rows: list[dict[str, Any]]) -> None:
    jsonl_path = output_dir / "generations" / f"step-{step:06d}.jsonl"
    dump_jsonl(jsonl_path, rows)
    md_path = output_dir / "generations" / f"step-{step:06d}.md"
    with open(md_path, "w") as f:
        f.write(f"# Fixed Generations Step {step}\n\n")
        for row in rows:
            f.write(f"## {row['id']}\n")
            f.write(f"- Dataset: {row['dataset']}\n")
            f.write(f"- Prompt: {row['prompt']}\n")
            f.write(f"- Target: {row['target_text']}\n")
            f.write(f"- Prediction: {row['prediction']}\n\n")


@torch.no_grad()
def generate_fixed_examples(
    accelerator: Accelerator,
    model,
    raw_model,
    processor,
    tokenizer,
    rows: list[dict[str, Any]],
    args: TrainArgs,
) -> list[dict[str, Any]]:
    if not accelerator.is_main_process:
        return []

    device = accelerator.device
    unwrapped = accelerator.unwrap_model(model)
    disable_lora_hooks(unwrapped)
    unwrapped.eval()
    raw_model.eval()

    outputs: list[dict[str, Any]] = []
    for row in rows:
        normalized = normalize_manifest_row(row)
        internalize_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": resolve_video_path(normalized["video_path"])},
                        {"type": "text", "text": args.internalization_prompt},
                    ],
                }
            ]
        ]
        try:
            vlm_inputs = prepare_smolvlm_inputs(
                processor,
                internalize_messages,
                device,
                video_fps=args.video_fps,
                max_frames=args.max_frames,
                video_size_longest_edge=args.video_size_longest_edge,
                video_load_backend=args.video_load_backend,
            )
            ctx_features, ctx_attn_mask, ctx_position_ids = extract_l2l_fused_text_features(
                raw_model,
                vlm_inputs,
                num_target_layers=unwrapped.hypernet.n_layers,
            )
            enable_lora_hooks(unwrapped)
            generated_loras, _ = unwrapped.generate_weights(
                ctx_ids=None,
                ctx_features=ctx_features,
                ctx_attn_mask=ctx_attn_mask,
                ctx_position_ids=ctx_position_ids,
            )
            generated_loras = combine_lora(
                generated_loras,
                torch.ones(1, dtype=torch.int32, device=device),
                lora_bias=unwrapped.hypernet.get_head_bias()
                if unwrapped.hypernet.config.use_bias
                else None,
            )
            prompt_ids = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": normalized["prompt"]}],
                    }
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            apply_lora_to_layers(
                unwrapped.base_model,
                unwrapped.hypernet.layer_indices,
                generated_loras,
                torch.ones(1, dtype=torch.int32, device=device),
                position_ids=None,
            )
            generated = unwrapped.base_model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=args.generation_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion = tokenizer.decode(
                generated[0][prompt_ids.shape[1] :],
                skip_special_tokens=True,
            ).strip()
            outputs.append(
                {
                    "id": normalized["id"],
                    "dataset": normalized["dataset"],
                    "prompt": normalized["prompt"],
                    "target_text": normalized["target_text"],
                    "prediction": completion,
                    "video_path": normalized["video_path"],
                    "task_type": normalized["task_type"],
                }
            )
        except Exception as e:  # pylint: disable=broad-except
            disable_lora_hooks(unwrapped)
            if _is_skippable_video_error(e):
                print(
                    f"[gen] skipped id={normalized['id']} due to video decode error: {type(e).__name__}: {e}",
                    flush=True,
                )
                continue
            raise
        disable_lora_hooks(unwrapped)

    set_model_train_state(unwrapped, raw_model)
    return outputs


def load_generation_rows(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    return load_video_manifest(path)


def load_resume_model_weights(
    args: TrainArgs,
    model,
) -> None:
    if not args.resume_checkpoint:
        return

    checkpoint_state = torch.load(
        args.resume_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint_state)


def load_resume_training_state(
    args: TrainArgs,
    optimizer,
    scheduler,
) -> tuple[int, float, float, int]:
    global_step = 0
    best_ce = float("inf")
    best_combined = float("inf")
    train_batches_seen = 0

    if args.resume_trainer_state:
        trainer_state = torch.load(
            args.resume_trainer_state,
            map_location="cpu",
            weights_only=False,
        )
        optimizer_state = trainer_state.get("optimizer")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = trainer_state.get("scheduler")
        if scheduler_state is not None and not args.resume_ignore_scheduler_state:
            scheduler.load_state_dict(scheduler_state)
        global_step = int(trainer_state.get("global_step", global_step))
        best_ce = float(trainer_state.get("best_ce", best_ce))
        best_combined = float(trainer_state.get("best_combined", best_combined))
        train_batches_seen = int(trainer_state.get("train_batches_seen", train_batches_seen))
        restore_rng_state(trainer_state.get("rng_state"))
        if args.resume_ignore_scheduler_state and global_step > 0:
            for _ in range(global_step):
                scheduler.step()
    elif args.resume_global_step:
        global_step = int(args.resume_global_step)
        for _ in range(global_step):
            scheduler.step()

    return global_step, best_ce, best_combined, train_batches_seen


def main():
    try:
        with open(f"/tmp/frames2lora_main_enter_pid_{os.getpid()}.log", "a") as f:
            f.write("entered main\n")
    except OSError:
        pass
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_layout(output_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    debug_rank_log(accelerator, "accelerator initialized")
    set_seed(args.seed)

    train_rows = load_video_manifest(
        args.train_manifest,
        max_samples=args.max_train_samples,
        default_split="train",
    )
    val_rows = load_video_manifest(
        args.val_manifest,
        max_samples=args.max_val_samples,
        default_split="val",
    )
    val_core_rows = (
        load_video_manifest(args.val_core_manifest, default_split="val")
        if args.val_core_manifest
        else val_rows
    )
    val_gen_rows = (
        load_generation_rows(args.val_gen_manifest)
        if args.val_gen_manifest
        else []
    )
    legacy_rows: list[dict[str, Any]] = []
    for manifest_path in args.legacy_sanity_manifests:
        legacy_rows.extend(
            load_video_manifest(
                manifest_path,
                max_samples=args.legacy_sanity_max_samples_per_manifest,
            )
        )

    debug_rank_log(
        accelerator,
        f"loaded manifests train={len(train_rows)} val={len(val_rows)} val_core={len(val_core_rows)} legacy={len(legacy_rows)}",
    )
    model, raw_model, processor, tokenizer = build_stage1_model(
        args,
        device=accelerator.device,
    )
    debug_rank_log(accelerator, "built stage1 model")
    collator = Stage1Collator(
        tokenizer=tokenizer,
        video_fps=args.video_fps,
        max_frames=args.max_frames,
    )

    train_loader = DataLoader(
        VideoCaptionDataset(train_rows, internalization_prompt=args.internalization_prompt),
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_core_loader = DataLoader(
        VideoCaptionDataset(val_core_rows, internalization_prompt=args.internalization_prompt),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    legacy_loader = None
    if legacy_rows:
        legacy_loader = DataLoader(
            VideoCaptionDataset(legacy_rows, internalization_prompt=args.internalization_prompt),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    # Load model weights before creating the optimizer. This model's custom
    # load_state_dict() rebuilds the hypernetwork modules, so constructing the
    # optimizer first would leave it pointing at stale Parameter objects.
    load_resume_model_weights(args, model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    warmup_steps = int(args.max_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_steps,
    )

    resume_global_step, resume_best_ce, resume_best_combined, resume_train_batches_seen = (
        load_resume_training_state(args, optimizer, scheduler)
    )
    debug_rank_log(accelerator, "loaded resume/trainer state")
    model, optimizer, train_loader, val_core_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_core_loader,
    )
    debug_rank_log(accelerator, "accelerator.prepare completed")
    if legacy_loader is not None:
        legacy_loader = accelerator.prepare(legacy_loader)
        debug_rank_log(accelerator, "legacy loader prepared")

    set_model_train_state(accelerator.unwrap_model(model), raw_model)
    disable_lora_hooks(accelerator.unwrap_model(model))

    if accelerator.is_main_process:
        with open(output_dir / "train_args.json", "w") as f:
            json.dump(asdict(args), f, indent=2)

        wandb_mode = resolve_wandb_mode(args.wandb_mode)
        if wandb_mode == "disabled":
            os.environ["WANDB_DISABLED"] = "true"
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or output_dir.name,
            group=args.wandb_group,
            notes=args.wandb_notes,
            config=asdict(args),
            mode=wandb_mode,
        )
        wandb.log(
            {
                "data/train_samples": len(train_rows),
                "data/val_samples": len(val_rows),
                "data/val_core_samples": len(val_core_rows),
                "data/val_gen_samples": len(val_gen_rows),
                "data/legacy_sanity_samples": len(legacy_rows),
                "data/effective_batch_size": (
                    args.per_device_batch_size
                    * args.gradient_accumulation_steps
                    * accelerator.num_processes
                ),
            },
            step=0,
        )

    global_step = resume_global_step
    best_ce = resume_best_ce
    best_combined = resume_best_combined
    total_ce_running = 0.0
    total_kl_running = 0.0
    total_combined_running = 0.0
    total_answer_tokens = 0.0
    total_context_tokens = 0.0
    running_microbatches = 0
    skipped_train_batches = 0
    skipped_train_samples = 0
    train_batches_seen = resume_train_batches_seen

    train_iterator = iter(train_loader)
    if resume_train_batches_seen and len(train_loader) > 0:
        skip_batches = resume_train_batches_seen % len(train_loader)
        for _ in range(skip_batches):
            try:
                next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                next(train_iterator)
    while global_step < args.max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        train_batches_seen += 1
        if global_step == resume_global_step:
            debug_rank_log(
                accelerator,
                f"fetched first batch ids={batch['ids'][:2]} batch_size={len(batch['ids'])}",
            )

        with accelerator.accumulate(model):
            local_skip = False
            local_error = ""
            text_batch = None
            ctx_features = None
            ctx_attn_mask = None
            ctx_position_ids = None
            outputs = None
            teacher_batch = None
            teacher_outputs = None
            try:
                text_batch = move_text_batch_to_device(batch, accelerator.device)
                if global_step == resume_global_step:
                    debug_rank_log(accelerator, "moved text batch to device")
                disable_lora_hooks(accelerator.unwrap_model(model))
                ctx_features, ctx_attn_mask, ctx_position_ids = extract_stage1_ctx_features(
                    raw_model,
                    processor,
                    batch,
                    accelerator.device,
                    num_target_layers=accelerator.unwrap_model(model).hypernet.n_layers,
                    video_size_longest_edge=args.video_size_longest_edge,
                    video_load_backend=args.video_load_backend,
                )
                if global_step == resume_global_step:
                    debug_rank_log(
                        accelerator,
                        f"extracted ctx features shape={tuple(ctx_features.shape)}",
                    )
                enable_lora_hooks(accelerator.unwrap_model(model))
                outputs = model(
                    ctx_features=ctx_features,
                    ctx_attn_mask=ctx_attn_mask,
                    ctx_position_ids=ctx_position_ids,
                    n_ctx_chunks=torch.ones(
                        ctx_features.shape[0],
                        dtype=torch.int32,
                        device=accelerator.device,
                    ),
                    input_ids=text_batch["input_ids"],
                    attention_mask=text_batch["attention_mask"],
                    labels=text_batch["labels"],
                )
                if global_step == resume_global_step:
                    debug_rank_log(accelerator, "completed first model forward")
                disable_lora_hooks(accelerator.unwrap_model(model))
                if args.kl_weight > 0:
                    with torch.no_grad():
                        teacher_batch = prepare_smolvlm_teacher_batch(
                            processor,
                            batch["teacher_prompt_messages"],
                            batch["teacher_full_messages"],
                            accelerator.device,
                            video_fps=batch["video_fps"],
                            max_frames=batch["max_frames"],
                            video_size_longest_edge=args.video_size_longest_edge,
                            video_load_backend=args.video_load_backend,
                        )
                        teacher_outputs = raw_model(
                            input_ids=teacher_batch["input_ids"],
                            attention_mask=teacher_batch.get("attention_mask"),
                            pixel_values=teacher_batch.get("pixel_values"),
                            pixel_attention_mask=teacher_batch.get("pixel_attention_mask"),
                            labels=None,
                            return_dict=True,
                            use_cache=False,
                        )
            except Exception as e:  # pylint: disable=broad-except
                disable_lora_hooks(accelerator.unwrap_model(model))
                if _is_skippable_video_error(e):
                    local_skip = True
                    local_error = f"{type(e).__name__}: {e}"
                else:
                    raise

            local_skip_tensor = torch.tensor(
                [1 if local_skip else 0],
                device=accelerator.device,
                dtype=torch.int32,
            )
            global_skip_tensor = accelerator.reduce(local_skip_tensor, reduction="sum")
            if global_skip_tensor.item() > 0:
                skipped_train_batches += 1
                skipped_train_samples += len(batch["ids"])
                optimizer.zero_grad(set_to_none=True)
                disable_lora_hooks(accelerator.unwrap_model(model))
                if local_skip and accelerator.is_local_main_process:
                    print(
                        f"[train] skipped batch due to video decode error: {local_error}",
                        flush=True,
                    )
                continue

            assert text_batch is not None
            assert ctx_features is not None
            assert ctx_attn_mask is not None
            assert outputs is not None
            ce_loss, _, answer_tokens = compute_ce_loss(outputs.logits, text_batch["labels"])
            if args.kl_weight > 0:
                assert teacher_batch is not None
                assert teacher_outputs is not None
                kl_loss, _ = compute_teacher_kl_loss(
                    outputs.logits,
                    text_batch["labels"],
                    teacher_outputs.logits,
                    teacher_batch["labels"],
                    temperature=args.kl_temperature,
                )
            else:
                kl_loss = outputs.logits.new_zeros(())
            combined_loss = ce_loss + args.kl_weight * kl_loss
            accelerator.backward(combined_loss)
            if global_step == resume_global_step:
                debug_rank_log(accelerator, "completed first backward")

            total_ce_running += ce_loss.item()
            total_kl_running += kl_loss.item()
            total_combined_running += combined_loss.item()
            total_answer_tokens += float(answer_tokens.sum().item())
            total_context_tokens += float(ctx_attn_mask.sum().item())
            running_microbatches += 1
            disable_lora_hooks(accelerator.unwrap_model(model))

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step == resume_global_step + 1:
                    debug_rank_log(accelerator, "completed first optimizer step")

                if accelerator.is_main_process and global_step % args.log_every == 0:
                    mean_den = max(running_microbatches, 1)
                    wandb.log(
                        {
                            "train/ce_loss": total_ce_running / mean_den,
                            "train/kl_loss": total_kl_running / mean_den,
                            "train/combined_loss": total_combined_running / mean_den,
                            "train/answer_tokens_per_sample": total_answer_tokens
                            / max(mean_den * args.per_device_batch_size, 1),
                            "train/context_tokens_per_sample": total_context_tokens
                            / max(mean_den * args.per_device_batch_size, 1),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/skipped_batches": skipped_train_batches,
                            "train/skipped_samples_local": skipped_train_samples,
                        },
                        step=global_step,
                    )

                total_ce_running = 0.0
                total_kl_running = 0.0
                total_combined_running = 0.0
                total_answer_tokens = 0.0
                total_context_tokens = 0.0
                running_microbatches = 0

                if global_step % args.eval_every == 0:
                    val_metrics = evaluate_loss(
                        accelerator,
                        model,
                        raw_model,
                        processor,
                        val_core_loader,
                        accelerator.device,
                        args,
                    )
                    legacy_metrics = None
                    if legacy_loader is not None:
                        legacy_metrics = evaluate_loss(
                            accelerator,
                            model,
                            raw_model,
                            processor,
                            legacy_loader,
                            accelerator.device,
                            args,
                        )

                    if accelerator.is_main_process:
                        log_payload = {
                            "val_core/ce_loss": val_metrics["ce_loss"],
                            "val_core/kl_loss": val_metrics["kl_loss"],
                            "val_core/combined_loss": val_metrics["combined_loss"],
                            "val_core/skipped_batches": val_metrics["skipped_batches"],
                            "val_core/skipped_samples": val_metrics["skipped_samples"],
                            "val_core/answer_tokens_per_sample": val_metrics[
                                "answer_tokens_per_sample"
                            ],
                            "val_core/context_tokens_per_sample": val_metrics[
                                "context_tokens_per_sample"
                            ],
                        }
                        if legacy_metrics is not None:
                            log_payload.update(
                                {
                                    "legacy_sanity/ce_loss": legacy_metrics["ce_loss"],
                                    "legacy_sanity/kl_loss": legacy_metrics["kl_loss"],
                                    "legacy_sanity/combined_loss": legacy_metrics["combined_loss"],
                                }
                            )
                        wandb.log(log_payload, step=global_step)

                        if val_gen_rows:
                            fixed_generations = generate_fixed_examples(
                                accelerator,
                                model,
                                raw_model,
                                processor,
                                tokenizer,
                                val_gen_rows,
                                args,
                            )
                            write_generation_artifacts(
                                output_dir,
                                global_step,
                                fixed_generations,
                            )

                    # Ensure all ranks finish before checkpoint selection so
                    # that non-main ranks don't race ahead while main is still
                    # doing inference inside generate_fixed_examples.
                    accelerator.wait_for_everyone()

                    if val_metrics["ce_loss"] < best_ce:
                        best_ce = val_metrics["ce_loss"]
                        save_checkpoint(accelerator, model, output_dir, "best-ce.pt")
                        save_trainer_state(
                            accelerator,
                            optimizer,
                            scheduler,
                            output_dir,
                            "best-ce.pt",
                            global_step=global_step,
                            best_ce=best_ce,
                            best_combined=best_combined,
                            train_batches_seen=train_batches_seen,
                        )
                    if val_metrics["combined_loss"] < best_combined:
                        best_combined = val_metrics["combined_loss"]
                        save_checkpoint(accelerator, model, output_dir, "best-combined.pt")
                        save_trainer_state(
                            accelerator,
                            optimizer,
                            scheduler,
                            output_dir,
                            "best-combined.pt",
                            global_step=global_step,
                            best_ce=best_ce,
                            best_combined=best_combined,
                            train_batches_seen=train_batches_seen,
                        )

                if global_step % args.save_every == 0:
                    save_checkpoint(
                        accelerator,
                        model,
                        output_dir,
                        f"step-{global_step:06d}.pt",
                    )
                    save_trainer_state(
                        accelerator,
                        optimizer,
                        scheduler,
                        output_dir,
                        f"step-{global_step:06d}.pt",
                        global_step=global_step,
                        best_ce=best_ce,
                        best_combined=best_combined,
                        train_batches_seen=train_batches_seen,
                    )

    save_checkpoint(accelerator, model, output_dir, f"last-step-{global_step:06d}.pt")
    save_trainer_state(
        accelerator,
        optimizer,
        scheduler,
        output_dir,
        f"last-step-{global_step:06d}.pt",
        global_step=global_step,
        best_ce=best_ce,
        best_combined=best_combined,
        train_batches_seen=train_batches_seen,
    )
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
