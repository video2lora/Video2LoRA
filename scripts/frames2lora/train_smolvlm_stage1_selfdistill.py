import argparse
import hashlib
import json
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
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from ctx_to_lora.data.video_manifest import (
    DATA_ROOT,
    dump_jsonl,
    load_video_manifest,
    normalize_manifest_row,
    resolve_video_path,
)
from scripts.frames2lora.train_smolvlm_online import (
    prepare_smolvlm_inputs,
    prepare_smolvlm_teacher_batch,
)
from scripts.frames2lora.train_smolvlm_stage1 import (
    build_stage1_model,
    compute_ce_loss,
    compute_teacher_kl_loss,
    disable_lora_hooks,
    enable_lora_hooks,
    extract_stage1_ctx_features,
    resolve_wandb_mode,
    save_checkpoint,
    set_model_train_state,
    _is_skippable_video_error,
)
from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers
from ctx_to_lora.modeling.lora_merger import combine_lora


TASK_BANK: tuple[dict[str, str], ...] = (
    {
        "task_type": "caption",
        "prompt": "Describe what is happening in this scene in one concise sentence.",
    },
    {
        "task_type": "caption",
        "prompt": "Caption this video scene briefly and concretely.",
    },
    {
        "task_type": "summary",
        "prompt": "Summarize this scene in 2 concise sentences.",
    },
    {
        "task_type": "summary",
        "prompt": "Give a short summary of the key events in this video scene.",
    },
    {
        "task_type": "qa",
        "prompt": "Question: What is the main action in this scene? Answer briefly.",
    },
    {
        "task_type": "qa",
        "prompt": "Question: What is the focus of this clip? Answer briefly.",
    },
    {
        "task_type": "qa",
        "prompt": "Question: What is happening in this video? Answer in one short sentence.",
    },
)

SCENE_TASK_BANK: tuple[dict[str, str], ...] = (
    {
        "task_type": "caption",
        "prompt": "Describe the visible action in this scene in one concise sentence.",
    },
    {
        "task_type": "caption",
        "prompt": "Caption this short scene briefly and concretely.",
    },
    {
        "task_type": "summary",
        "prompt": "Summarize the key visible event in this scene.",
    },
    {
        "task_type": "qa",
        "prompt": "Question: What is the main visible action in this scene? Answer briefly.",
    },
)

ADJACENT_TASK_BANK: tuple[dict[str, str], ...] = (
    {
        "task_type": "summary",
        "prompt": "Summarize the sequence of visible events in this clip.",
    },
    {
        "task_type": "summary",
        "prompt": "Describe how the action changes across this short video segment.",
    },
    {
        "task_type": "caption",
        "prompt": "Describe the main sequence of actions in this clip.",
    },
    {
        "task_type": "qa",
        "prompt": "Question: What changes or progression are visible across this clip? Answer briefly.",
    },
)

FULL_TASK_BANK: tuple[dict[str, str], ...] = (
    {
        "task_type": "summary",
        "prompt": "Summarize the full video in 2 concise sentences, focusing only on visible content.",
    },
    {
        "task_type": "summary",
        "prompt": "Give a brief visual summary of the full video.",
    },
    {
        "task_type": "caption",
        "prompt": "Describe the main visible topic and activity of this full video.",
    },
    {
        "task_type": "qa",
        "prompt": "Question: What is the overall visible focus of this full video? Answer briefly.",
    },
)


@dataclass
class TrainArgs:
    smolvlm_name_or_path: str
    train_manifest: str
    val_manifest: str
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
    internalization_prompt: str
    kl_temperature: float
    teacher_max_new_tokens: int
    student_generation_max_new_tokens: int
    fixed_prompt_mode: str
    wandb_project: str
    wandb_mode: str
    wandb_group: str | None
    wandb_run_name: str | None
    wandb_notes: str | None


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(
        description="Stage 1 self-distillation for Frames2LoRA using teacher-generated tasks."
    )
    parser.add_argument(
        "--smolvlm-name-or-path",
        default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    )
    parser.add_argument(
        "--train-manifest",
        default=str(DATA_ROOT / "processed" / "finevideo" / "train.readable.jsonl"),
    )
    parser.add_argument(
        "--val-manifest",
        default=str(DATA_ROOT / "processed" / "finevideo" / "val.readable.jsonl"),
    )
    parser.add_argument(
        "--val-gen-manifest",
        default=str(DATA_ROOT / "processed" / "finevideo" / "val_gen_100.readable.jsonl"),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", default="down_proj")
    parser.add_argument("--latent-size", type=int, default=512)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--n-latent-queries", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=9)
    parser.add_argument("--num-self-attn-per-block", type=int, default=0)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--video-size-longest-edge", type=int, default=384)
    parser.add_argument(
        "--internalization-prompt",
        default="Internalize this video for later captioning, summarization, and question answering.",
    )
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=32)
    parser.add_argument("--student-generation-max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--fixed-prompt-mode",
        default="cycle",
        choices=("cycle", "caption_only", "summary_only", "qa_only"),
    )
    parser.add_argument("--wandb-project", default="frames2lora-finevideo-selfdistill")
    parser.add_argument(
        "--wandb-mode",
        default="auto",
        choices=("auto", "online", "offline", "disabled"),
    )
    parser.add_argument("--wandb-group", default="finevideo-selfdistill")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-notes", default=None)
    parsed = parser.parse_args()

    output_dir = parsed.output_dir
    if not output_dir:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output_dir = str(DATA_ROOT / "runs" / f"{timestamp}-smolvlm-stage1-selfdistill")

    def maybe_existing(path: str | None) -> str | None:
        if path and os.path.exists(path):
            return path
        return None

    return TrainArgs(
        smolvlm_name_or_path=parsed.smolvlm_name_or_path,
        train_manifest=parsed.train_manifest,
        val_manifest=parsed.val_manifest,
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
        internalization_prompt=parsed.internalization_prompt,
        kl_temperature=parsed.kl_temperature,
        teacher_max_new_tokens=parsed.teacher_max_new_tokens,
        student_generation_max_new_tokens=parsed.student_generation_max_new_tokens,
        fixed_prompt_mode=parsed.fixed_prompt_mode,
        wandb_project=parsed.wandb_project,
        wandb_mode=parsed.wandb_mode,
        wandb_group=parsed.wandb_group,
        wandb_run_name=parsed.wandb_run_name,
        wandb_notes=parsed.wandb_notes,
    )


def ensure_layout(output_dir: Path) -> None:
    for path in (
        DATA_ROOT,
        DATA_ROOT / "runs",
        output_dir,
        output_dir / "checkpoints",
        output_dir / "generations",
    ):
        path.mkdir(parents=True, exist_ok=True)


class DistillVideoDataset(Dataset):
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
            "dataset": row["dataset"],
            "task_type": row.get("task_type", "caption"),
            "split": row.get("split"),
            "metadata": row.get("metadata", {}),
            "span_type": (row.get("metadata") or {}).get("span_type", "scene"),
            "internalization_prompt": self.internalization_prompt,
        }


def _stable_index(key: str, modulo: int) -> int:
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def _bank_for_span(span_type: str) -> tuple[dict[str, str], ...]:
    if span_type == "scene":
        return SCENE_TASK_BANK
    if span_type == "adjacent":
        return ADJACENT_TASK_BANK
    if span_type == "full":
        return FULL_TASK_BANK
    return TASK_BANK


def _select_task(example_id: str, span_type: str, mode: str, *, train: bool) -> dict[str, str]:
    if mode == "caption_only":
        bank = [task for task in _bank_for_span(span_type) if task["task_type"] == "caption"]
    elif mode == "summary_only":
        bank = [task for task in _bank_for_span(span_type) if task["task_type"] == "summary"]
    elif mode == "qa_only":
        bank = [task for task in _bank_for_span(span_type) if task["task_type"] == "qa"]
    else:
        bank = list(_bank_for_span(span_type))
    if train:
        return random.choice(bank)
    return bank[_stable_index(example_id, len(bank))]


class SelfDistillCollator:
    def __init__(self, *, video_fps: float | None, max_frames: int, prompt_mode: str, train: bool):
        self.video_fps = video_fps
        self.max_frames = max_frames
        self.prompt_mode = prompt_mode
        self.train = train

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        video_messages = []
        teacher_prompt_messages = []
        prompts = []
        distill_task_types = []
        span_types = []

        for example in batch:
            task = _select_task(
                example["id"],
                example["span_type"],
                self.prompt_mode,
                train=self.train,
            )
            prompts.append(task["prompt"])
            distill_task_types.append(task["task_type"])
            span_types.append(example["span_type"])
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
                            {"type": "text", "text": task["prompt"]},
                        ],
                    }
                ]
            )

        return {
            "ids": [example["id"] for example in batch],
            "video_paths": [example["video_path"] for example in batch],
            "datasets": [example["dataset"] for example in batch],
            "metadata": [example["metadata"] for example in batch],
            "prompts": prompts,
            "distill_task_types": distill_task_types,
            "span_types": span_types,
            "video_messages": video_messages,
            "teacher_prompt_messages": teacher_prompt_messages,
            "video_fps": self.video_fps,
            "max_frames": self.max_frames,
        }


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


def build_student_text_batch(tokenizer, prompts: list[str], targets: list[str]) -> dict[str, torch.Tensor]:
    input_ids, attention_masks, labels = [], [], []
    for prompt, target in zip(prompts, targets, strict=True):
        inp_ids, attn_mask, lbl = build_labels(tokenizer, prompt, target)
        input_ids.append(inp_ids)
        attention_masks.append(attn_mask)
        labels.append(lbl)
    batch_input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
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
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }


def move_text_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _prepare_generation_inputs(
    processor,
    prompt_messages,
    device,
    *,
    video_fps: float | None,
    max_frames: int,
    video_size_longest_edge: int | None,
):
    chat_template_kwargs = dict(max_frames=max_frames, padding=True)
    # Batched generation on decoder-only text backbones should left-pad prompts.
    processor.tokenizer.padding_side = "left"
    if video_fps is not None:
        chat_template_kwargs["target_fps"] = video_fps
    if video_size_longest_edge is not None:
        video_size = {"longest_edge": video_size_longest_edge}
        processor.video_size = video_size
        processor.image_processor.size = video_size
    prompt_inputs = processor.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **chat_template_kwargs,
    )
    moved = {}
    for key, value in prompt_inputs.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def generate_teacher_targets(
    raw_model,
    processor,
    tokenizer,
    prompt_messages,
    device,
    *,
    video_fps: float | None,
    max_frames: int,
    video_size_longest_edge: int | None,
    max_new_tokens: int,
) -> list[str]:
    prompt_inputs = _prepare_generation_inputs(
        processor,
        prompt_messages,
        device,
        video_fps=video_fps,
        max_frames=max_frames,
        video_size_longest_edge=video_size_longest_edge,
    )
    generated = raw_model.generate(
        **prompt_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()
    outputs = []
    for row_idx, prompt_len in enumerate(prompt_lens):
        text = tokenizer.decode(
            generated[row_idx][prompt_len:],
            skip_special_tokens=True,
        ).strip()
        if not text:
            text = "Unable to determine."
        outputs.append(text)
    return outputs


def build_teacher_full_messages(prompt_messages, targets: list[str]):
    full_messages = []
    for prompt_message, target in zip(prompt_messages, targets, strict=True):
        full_messages.append(
            [
                prompt_message[0],
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target}],
                },
            ]
        )
    return full_messages


@torch.no_grad()
def evaluate_loss(
    accelerator: Accelerator,
    model,
    raw_model,
    processor,
    tokenizer,
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
    total_teacher_ce = torch.zeros(1, device=device)
    total_kl = torch.zeros(1, device=device)
    total_answer_tokens = torch.zeros(1, device=device)
    total_context_tokens = torch.zeros(1, device=device)
    skipped_batches = torch.zeros(1, device=device)
    skipped_samples = torch.zeros(1, device=device)

    for batch in data_loader:
        local_skip = False
        local_error = ""
        text_batch = None
        teacher_batch = None
        teacher_outputs = None
        outputs = None
        ctx_attn_mask = None
        try:
            teacher_targets = generate_teacher_targets(
                raw_model,
                processor,
                tokenizer,
                batch["teacher_prompt_messages"],
                device,
                video_fps=batch["video_fps"],
                max_frames=batch["max_frames"],
                video_size_longest_edge=args.video_size_longest_edge,
                max_new_tokens=args.teacher_max_new_tokens,
            )
            student_text_batch = build_student_text_batch(tokenizer, batch["prompts"], teacher_targets)
            text_batch = move_text_batch_to_device(student_text_batch, device)
            disable_lora_hooks(unwrapped)
            ctx_features, ctx_attn_mask, ctx_position_ids = extract_stage1_ctx_features(
                raw_model,
                processor,
                batch,
                device,
                num_target_layers=unwrapped.hypernet.n_layers,
                video_size_longest_edge=args.video_size_longest_edge,
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
            teacher_batch = prepare_smolvlm_teacher_batch(
                processor,
                batch["teacher_prompt_messages"],
                build_teacher_full_messages(batch["teacher_prompt_messages"], teacher_targets),
                device,
                video_fps=batch["video_fps"],
                max_frames=batch["max_frames"],
                video_size_longest_edge=args.video_size_longest_edge,
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
        except Exception as exc:  # pylint: disable=broad-except
            disable_lora_hooks(unwrapped)
            if _is_skippable_video_error(exc):
                local_skip = True
                local_error = f"{type(exc).__name__}: {exc}"
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
        assert teacher_batch is not None
        assert teacher_outputs is not None
        assert outputs is not None
        assert ctx_attn_mask is not None
        teacher_ce, per_sample_ce, answer_tokens = compute_ce_loss(outputs.logits, text_batch["labels"])
        _, per_sample_kl = compute_teacher_kl_loss(
            outputs.logits,
            text_batch["labels"],
            teacher_outputs.logits,
            teacher_batch["labels"],
            temperature=args.kl_temperature,
        )
        batch_size = torch.tensor([per_sample_ce.shape[0]], device=device, dtype=torch.float32)
        total_samples += batch_size
        total_teacher_ce += per_sample_ce.sum()
        total_kl += per_sample_kl.sum()
        total_answer_tokens += answer_tokens.sum().to(dtype=torch.float32)
        total_context_tokens += ctx_attn_mask.sum().to(dtype=torch.float32)
        disable_lora_hooks(unwrapped)

    total_samples = accelerator.reduce(total_samples, reduction="sum")
    total_teacher_ce = accelerator.reduce(total_teacher_ce, reduction="sum")
    total_kl = accelerator.reduce(total_kl, reduction="sum")
    total_answer_tokens = accelerator.reduce(total_answer_tokens, reduction="sum")
    total_context_tokens = accelerator.reduce(total_context_tokens, reduction="sum")
    skipped_batches = accelerator.reduce(skipped_batches, reduction="sum")
    skipped_samples = accelerator.reduce(skipped_samples, reduction="sum")

    metrics = {
        "teacher_ce_loss": (total_teacher_ce / total_samples.clamp_min(1)).item(),
        "kl_loss": (total_kl / total_samples.clamp_min(1)).item(),
        "combined_loss": (total_kl / total_samples.clamp_min(1)).item(),
        "samples": total_samples.item(),
        "skipped_batches": skipped_batches.item(),
        "skipped_samples": skipped_samples.item(),
        "answer_tokens_per_sample": (total_answer_tokens / total_samples.clamp_min(1)).item(),
        "context_tokens_per_sample": (total_context_tokens / total_samples.clamp_min(1)).item(),
    }
    set_model_train_state(unwrapped, raw_model)
    return metrics


def write_generation_artifacts(output_dir: Path, step: int, rows: list[dict[str, Any]]) -> None:
    jsonl_path = output_dir / "generations" / f"step-{step:06d}.jsonl"
    dump_jsonl(jsonl_path, rows)
    md_path = output_dir / "generations" / f"step-{step:06d}.md"
    with open(md_path, "w") as f:
        f.write(f"# Fixed Generations Step {step}\n\n")
        for row in rows:
            f.write(f"## {row['id']}\n")
            f.write(f"- Dataset: {row['dataset']}\n")
            f.write(f"- Task: {row['task_type']}\n")
            f.write(f"- Prompt: {row['prompt']}\n")
            f.write(f"- Teacher: {row['teacher_prediction']}\n")
            f.write(f"- Student: {row['student_prediction']}\n\n")


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
        span_type = (normalized.get("metadata") or {}).get("span_type", "scene")
        task = _select_task(normalized["id"], span_type, args.fixed_prompt_mode, train=False)
        prompt = task["prompt"]
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
        teacher_prompt_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": resolve_video_path(normalized["video_path"])},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        ]
        try:
            teacher_prediction = generate_teacher_targets(
                raw_model,
                processor,
                tokenizer,
                teacher_prompt_messages,
                device,
                video_fps=args.video_fps,
                max_frames=args.max_frames,
                video_size_longest_edge=args.video_size_longest_edge,
                max_new_tokens=args.teacher_max_new_tokens,
            )[0]
            ctx_features, ctx_attn_mask, ctx_position_ids = extract_stage1_ctx_features(
                raw_model,
                processor,
                {
                    "video_messages": internalize_messages,
                    "video_fps": args.video_fps,
                    "max_frames": args.max_frames,
                },
                device,
                num_target_layers=unwrapped.hypernet.n_layers,
                video_size_longest_edge=args.video_size_longest_edge,
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
                        "content": [{"type": "text", "text": prompt}],
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
                max_new_tokens=args.student_generation_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            student_prediction = tokenizer.decode(
                generated[0][prompt_ids.shape[1] :],
                skip_special_tokens=True,
            ).strip()
            outputs.append(
                {
                    "id": normalized["id"],
                    "dataset": normalized["dataset"],
                    "task_type": task["task_type"],
                    "span_type": span_type,
                    "prompt": prompt,
                    "teacher_prediction": teacher_prediction,
                    "student_prediction": student_prediction,
                    "video_path": normalized["video_path"],
                }
            )
        except Exception as exc:  # pylint: disable=broad-except
            disable_lora_hooks(unwrapped)
            if _is_skippable_video_error(exc):
                print(
                    f"[gen] skipped id={normalized['id']} due to video decode error: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            raise
        disable_lora_hooks(unwrapped)

    set_model_train_state(unwrapped, raw_model)
    return outputs


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_layout(output_dir)
    set_seed(args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    train_rows = load_video_manifest(args.train_manifest, max_samples=args.max_train_samples)
    val_rows = load_video_manifest(args.val_manifest, max_samples=args.max_val_samples)
    val_gen_rows = load_video_manifest(args.val_gen_manifest) if args.val_gen_manifest else []

    train_dataset = DistillVideoDataset(train_rows, internalization_prompt=args.internalization_prompt)
    val_dataset = DistillVideoDataset(val_rows, internalization_prompt=args.internalization_prompt)

    collator = SelfDistillCollator(
        video_fps=args.video_fps,
        max_frames=args.max_frames,
        prompt_mode="cycle",
        train=True,
    )
    val_collator = SelfDistillCollator(
        video_fps=args.video_fps,
        max_frames=args.max_frames,
        prompt_mode=args.fixed_prompt_mode,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model, raw_model, processor, tokenizer = build_stage1_model(args)
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

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
    )

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
                "data/val_gen_samples": len(val_gen_rows),
                "data/effective_batch_size": (
                    args.per_device_batch_size
                    * args.gradient_accumulation_steps
                    * accelerator.num_processes
                ),
            },
            step=0,
        )

    global_step = 0
    best_kl = float("inf")
    total_teacher_ce_running = 0.0
    total_kl_running = 0.0
    total_answer_tokens = 0.0
    total_context_tokens = 0.0
    running_microbatches = 0
    skipped_train_batches = 0
    skipped_train_samples = 0

    train_iterator = iter(train_loader)
    while global_step < args.max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        with accelerator.accumulate(model):
            local_skip = False
            local_error = ""
            text_batch = None
            teacher_batch = None
            teacher_outputs = None
            outputs = None
            ctx_attn_mask = None
            try:
                teacher_targets = generate_teacher_targets(
                    raw_model,
                    processor,
                    tokenizer,
                    batch["teacher_prompt_messages"],
                    accelerator.device,
                    video_fps=batch["video_fps"],
                    max_frames=batch["max_frames"],
                    video_size_longest_edge=args.video_size_longest_edge,
                    max_new_tokens=args.teacher_max_new_tokens,
                )
                student_text_batch = build_student_text_batch(
                    tokenizer,
                    batch["prompts"],
                    teacher_targets,
                )
                text_batch = move_text_batch_to_device(student_text_batch, accelerator.device)
                disable_lora_hooks(accelerator.unwrap_model(model))
                ctx_features, ctx_attn_mask, ctx_position_ids = extract_stage1_ctx_features(
                    raw_model,
                    processor,
                    batch,
                    accelerator.device,
                    num_target_layers=accelerator.unwrap_model(model).hypernet.n_layers,
                    video_size_longest_edge=args.video_size_longest_edge,
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
                disable_lora_hooks(accelerator.unwrap_model(model))
                with torch.no_grad():
                    teacher_batch = prepare_smolvlm_teacher_batch(
                        processor,
                        batch["teacher_prompt_messages"],
                        build_teacher_full_messages(batch["teacher_prompt_messages"], teacher_targets),
                        accelerator.device,
                        video_fps=batch["video_fps"],
                        max_frames=batch["max_frames"],
                        video_size_longest_edge=args.video_size_longest_edge,
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
            except Exception as exc:  # pylint: disable=broad-except
                disable_lora_hooks(accelerator.unwrap_model(model))
                if _is_skippable_video_error(exc):
                    local_skip = True
                    local_error = f"{type(exc).__name__}: {exc}"
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
            assert teacher_batch is not None
            assert teacher_outputs is not None
            assert outputs is not None
            assert ctx_attn_mask is not None
            teacher_ce_loss, _, answer_tokens = compute_ce_loss(outputs.logits, text_batch["labels"])
            kl_loss, _ = compute_teacher_kl_loss(
                outputs.logits,
                text_batch["labels"],
                teacher_outputs.logits,
                teacher_batch["labels"],
                temperature=args.kl_temperature,
            )
            accelerator.backward(kl_loss)

            total_teacher_ce_running += teacher_ce_loss.item()
            total_kl_running += kl_loss.item()
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

                if accelerator.is_main_process and global_step % args.log_every == 0:
                    mean_den = max(running_microbatches, 1)
                    wandb.log(
                        {
                            "train/teacher_ce_loss": total_teacher_ce_running / mean_den,
                            "train/kl_loss": total_kl_running / mean_den,
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

                total_teacher_ce_running = 0.0
                total_kl_running = 0.0
                total_answer_tokens = 0.0
                total_context_tokens = 0.0
                running_microbatches = 0

                if global_step % args.eval_every == 0:
                    val_metrics = evaluate_loss(
                        accelerator,
                        model,
                        raw_model,
                        processor,
                        tokenizer,
                        val_loader,
                        accelerator.device,
                        args,
                    )
                    if accelerator.is_main_process:
                        wandb.log(
                            {
                                "val/teacher_ce_loss": val_metrics["teacher_ce_loss"],
                                "val/kl_loss": val_metrics["kl_loss"],
                                "val/skipped_batches": val_metrics["skipped_batches"],
                                "val/skipped_samples": val_metrics["skipped_samples"],
                                "val/answer_tokens_per_sample": val_metrics[
                                    "answer_tokens_per_sample"
                                ],
                                "val/context_tokens_per_sample": val_metrics[
                                    "context_tokens_per_sample"
                                ],
                            },
                            step=global_step,
                        )
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
                            write_generation_artifacts(output_dir, global_step, fixed_generations)

                    accelerator.wait_for_everyone()

                    if val_metrics["kl_loss"] < best_kl:
                        best_kl = val_metrics["kl_loss"]
                        save_checkpoint(accelerator, model, output_dir, "best-kl.pt")

                if global_step % args.save_every == 0:
                    save_checkpoint(
                        accelerator,
                        model,
                        output_dir,
                        f"step-{global_step:06d}.pt",
                    )

    save_checkpoint(accelerator, model, output_dir, f"last-step-{global_step:06d}.pt")
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
