import argparse
import json
import math
import os
import random
import time
from collections import OrderedDict
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
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    get_cosine_schedule_with_warmup,
)
from transformers.utils import is_av_available, is_cv2_available, is_decord_available

from ctx_to_lora.configs import (
    AggregatorArguments,
    CtxEncoderArguments,
    HypernetArguments,
)
from ctx_to_lora.model_loading import get_lora_config
from ctx_to_lora.modeling.hypernet import (
    ModulatedPretrainedModel,
    get_hypernet_config,
)


DATA_ROOT = Path(os.environ.get("FRAMES2LORA_DATA_ROOT", "data/frames2lora"))


@dataclass
class TrainArgs:
    smolvlm_name_or_path: str
    base_lm_name_or_path: str
    train_manifest: str
    val_manifest: str | None
    output_dir: str
    epochs: int
    batch_size: int
    eval_batch_size: int
    grad_accum_steps: int
    max_steps: int | None
    learning_rate: float
    weight_decay: float
    warmup_steps: int
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
    questions_per_video: int
    frame_pooling: str
    ctx_feature_mode: str
    kl_weight: float
    kl_temperature: float
    wandb_project: str
    wandb_mode: str
    wandb_group: str | None
    wandb_run_name: str | None
    wandb_notes: str | None


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(
        description="Train Frames2LoRA with online SmolVLM video feature extraction."
    )
    parser.add_argument(
        "--smolvlm-name-or-path",
        default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    )
    parser.add_argument(
        "--base-lm-name-or-path",
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    )
    parser.add_argument(
        "--train-manifest",
        default=str(DATA_ROOT / "processed" / "train.jsonl"),
    )
    parser.add_argument(
        "--val-manifest",
        default=str(DATA_ROOT / "processed" / "val.jsonl"),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        default="q_proj,v_proj,down_proj",
    )
    parser.add_argument("--latent-size", type=int, default=512)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--n-latent-queries", type=int, default=208)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--num-self-attn-per-block", type=int, default=1)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=24)
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
    parser.add_argument("--questions-per-video", type=int, default=4)
    parser.add_argument(
        "--frame-pooling",
        default="mean",
        choices=("mean", "flatten"),
    )
    parser.add_argument(
        "--ctx-feature-mode",
        default="visual_pooled",
        choices=("visual_pooled", "l2l_fused_text"),
    )
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--wandb-project", default="frames2lora-video-centric")
    parser.add_argument(
        "--wandb-mode",
        default="auto",
        choices=("auto", "online", "offline", "disabled"),
    )
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-notes", default=None)
    parsed = parser.parse_args()

    output_dir = parsed.output_dir
    if not output_dir:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output_dir = str(DATA_ROOT / "runs" / f"{timestamp}-smolvlm-online")

    val_manifest = parsed.val_manifest
    if val_manifest and not os.path.exists(val_manifest):
        val_manifest = None

    return TrainArgs(
        smolvlm_name_or_path=parsed.smolvlm_name_or_path,
        base_lm_name_or_path=parsed.base_lm_name_or_path,
        train_manifest=parsed.train_manifest,
        val_manifest=val_manifest,
        output_dir=output_dir,
        epochs=parsed.epochs,
        batch_size=parsed.batch_size,
        eval_batch_size=parsed.eval_batch_size or parsed.batch_size,
        grad_accum_steps=parsed.grad_accum_steps,
        max_steps=parsed.max_steps,
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        warmup_steps=parsed.warmup_steps,
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
        questions_per_video=parsed.questions_per_video,
        frame_pooling=parsed.frame_pooling,
        ctx_feature_mode=parsed.ctx_feature_mode,
        kl_weight=parsed.kl_weight,
        kl_temperature=parsed.kl_temperature,
        wandb_project=parsed.wandb_project,
        wandb_mode=parsed.wandb_mode,
        wandb_group=parsed.wandb_group,
        wandb_run_name=parsed.wandb_run_name,
        wandb_notes=parsed.wandb_notes,
    )


def ensure_layout(output_dir: Path) -> None:
    for path in (
        DATA_ROOT,
        DATA_ROOT / "raw",
        DATA_ROOT / "processed",
        DATA_ROOT / "features",
        DATA_ROOT / "runs",
        DATA_ROOT / "cache",
        output_dir,
        output_dir / "checkpoints",
    ):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def resolve_video_path(video_path_str: str) -> str:
    video_path = Path(video_path_str)
    if not video_path.is_absolute():
        video_path = DATA_ROOT / video_path
    return str(video_path)


class VideoQADataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        questions_per_video: int,
        sample_questions_randomly: bool,
    ):
        grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for row in rows:
            video_path = resolve_video_path(row["video_path"])
            if video_path not in grouped:
                grouped[video_path] = {
                    "id": row.get("id", video_path),
                    "video_path": video_path,
                    "metadata": row.get("metadata", {}),
                    "qas": [],
                }
            grouped[video_path]["qas"].append(
                {
                    "id": row.get("id"),
                    "question": row["question"],
                    "answer": row["answer"],
                }
            )
        self.rows = list(grouped.values())
        self.questions_per_video = questions_per_video
        self.sample_questions_randomly = sample_questions_randomly

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        qas = row["qas"]
        if 0 < self.questions_per_video < len(qas):
            if self.sample_questions_randomly:
                qas = random.sample(qas, k=self.questions_per_video)
            else:
                qas = qas[: self.questions_per_video]
        return {
            "id": row.get("id", str(idx)),
            "video_path": row["video_path"],
            "qas": qas,
            "metadata": row.get("metadata", {}),
        }


def build_labels(tokenizer, question: str, answer: str):
    prompt_messages = [{"role": "user", "content": question}]
    full_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
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


class SmolVLMOnlineCollator:
    def __init__(
        self,
        base_tokenizer,
        video_fps: float | None,
        max_frames: int,
        video_size_longest_edge: int | None = None,
    ):
        self.base_tokenizer = base_tokenizer
        self.video_fps = video_fps
        self.max_frames = max_frames
        self.video_size_longest_edge = video_size_longest_edge

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids, attention_masks, labels = [], [], []
        qa_ids = []
        video_ids = []
        video_messages = []
        teacher_prompt_messages = []
        teacher_full_messages = []
        question_counts = []

        for example in batch:
            qas = example["qas"]
            question_counts.append(len(qas))
            video_ids.append(example["id"])
            video_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": example["video_path"]},
                            {
                                "type": "text",
                                "text": "Internalize this video for later question answering.",
                            },
                        ],
                    }
                ]
            )
            for qa_idx, qa in enumerate(qas):
                inp_ids, attn_mask, lbl = build_labels(
                    self.base_tokenizer,
                    question=qa["question"],
                    answer=qa["answer"],
                )
                input_ids.append(inp_ids)
                attention_masks.append(attn_mask)
                labels.append(lbl)
                qa_ids.append(qa.get("id") or f"{example['id']}-q{qa_idx}")
                teacher_prompt_messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "path": example["video_path"]},
                                {"type": "text", "text": qa["question"]},
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
                                {"type": "text", "text": qa["question"]},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": qa["answer"]}],
                        },
                    ]
                )

        batch_input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.base_tokenizer.pad_token_id,
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
            "ids": qa_ids,
            "video_ids": video_ids,
            "question_counts": torch.tensor(question_counts, dtype=torch.long),
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
            "video_messages": video_messages,
            "teacher_prompt_messages": teacher_prompt_messages,
            "teacher_full_messages": teacher_full_messages,
            "video_fps": self.video_fps,
            "max_frames": self.max_frames,
            "video_size_longest_edge": self.video_size_longest_edge,
        }


def move_text_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def build_base_model(args: TrainArgs, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_lm_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    peft_config = get_lora_config(
        args.base_lm_name_or_path,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_lm_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device.index if device.index is not None else 0},
    )
    from peft import PeftModel

    base_model = PeftModel(base_model, peft_config)
    base_model.train()
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(base_model, "generation_config", None):
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    return base_model, tokenizer


def build_frames2lora_model(args: TrainArgs, device: torch.device):
    base_model, base_tokenizer = build_base_model(args, device)

    processor = AutoProcessor.from_pretrained(
        args.smolvlm_name_or_path,
        trust_remote_code=True,
    )
    vlm = AutoModelForImageTextToText.from_pretrained(
        args.smolvlm_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device.index if device.index is not None else 0},
    )
    vlm.eval()
    for param in vlm.parameters():
        param.requires_grad = False

    ctx_hidden_size = vlm.config.text_config.hidden_size
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
        ctx_encoder_type=(
            "per_layer_activations"
            if args.ctx_feature_mode == "l2l_fused_text"
            else "early_exit"
        ),
    )
    ctx_config = PretrainedConfig(hidden_size=ctx_hidden_size)
    hypernet_config = get_hypernet_config(
        base_model,
        ctx_config,
        hypernet_args,
        aggregator_args,
        ctx_encoder_args,
    )
    model = ModulatedPretrainedModel(
        base_model,
        hypernet_config,
        ctx_encoder_args,
        use_sequence_packing=False,
    )
    model.to(device)
    model.train()
    return model, base_tokenizer, processor, vlm


def prepare_smolvlm_inputs(
    processor,
    video_messages,
    device,
    video_fps: float | None,
    max_frames: int,
    video_size_longest_edge: int | None = None,
    video_load_backend: str = "auto",
):
    def _message_requests_visuals(messages) -> bool:
        for conversation in messages:
            for message in conversation:
                for content in message.get("content", []):
                    if content.get("type") in {"image", "video"}:
                        return True
        return False

    def _has_multimodal_payload(vlm_inputs) -> bool:
        for key, value in vlm_inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            lowered = key.lower()
            if "pixel" in lowered or "image" in lowered or "video" in lowered:
                return True
        return False

    chat_template_kwargs = dict(padding=True)
    processor_name = type(processor).__name__.lower()
    # SmolVLM2 and Idefics3 use different multimodal kwargs for video sampling.
    if "idefics3" in processor_name:
        chat_template_kwargs["num_frames"] = max_frames
        if video_fps is not None:
            chat_template_kwargs["video_fps"] = video_fps
    else:
        chat_template_kwargs["max_frames"] = max_frames
        if video_fps is not None:
            chat_template_kwargs["target_fps"] = video_fps
    if video_load_backend == "auto":
        if is_decord_available():
            video_load_backend = "decord"
        elif is_av_available():
            video_load_backend = "pyav"
        elif is_cv2_available():
            video_load_backend = "opencv"
        else:
            video_load_backend = "torchvision"
    chat_template_kwargs["video_load_backend"] = video_load_backend
    if video_size_longest_edge is not None:
        video_size = {"longest_edge": video_size_longest_edge}
        processor.video_size = video_size
        processor.image_processor.size = video_size

    vlm_inputs = processor.apply_chat_template(
        video_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **chat_template_kwargs,
    )
    if _message_requests_visuals(video_messages) and not _has_multimodal_payload(vlm_inputs):
        raise ValueError(
            "Processor produced no multimodal tensors for a visual prompt. "
            "This model/processor path is not consuming the provided image/video input."
        )
    moved = {}
    for key, value in vlm_inputs.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def prepare_smolvlm_teacher_batch(
    processor,
    prompt_messages,
    full_messages,
    device,
    video_fps: float | None,
    max_frames: int,
    video_size_longest_edge: int | None = None,
    video_load_backend: str = "auto",
):
    chat_template_kwargs = dict(padding=True)
    processor_name = type(processor).__name__.lower()
    if "idefics3" in processor_name:
        chat_template_kwargs["num_frames"] = max_frames
        if video_fps is not None:
            chat_template_kwargs["video_fps"] = video_fps
    else:
        chat_template_kwargs["max_frames"] = max_frames
        if video_fps is not None:
            chat_template_kwargs["target_fps"] = video_fps
    if video_load_backend == "auto":
        if is_decord_available():
            video_load_backend = "decord"
        elif is_av_available():
            video_load_backend = "pyav"
        elif is_cv2_available():
            video_load_backend = "opencv"
        else:
            video_load_backend = "torchvision"
    chat_template_kwargs["video_load_backend"] = video_load_backend
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
    full_inputs = processor.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **chat_template_kwargs,
    )
    labels = full_inputs["input_ids"].clone()
    full_attention_mask = full_inputs.get("attention_mask")
    prompt_attention_mask = prompt_inputs.get("attention_mask")
    if full_attention_mask is None or prompt_attention_mask is None:
        raise ValueError("Teacher SmolVLM inputs require attention masks.")
    labels[full_attention_mask == 0] = -100
    prompt_lens = prompt_attention_mask.sum(dim=1).tolist()
    for row_idx, prompt_len in enumerate(prompt_lens):
        labels[row_idx, :prompt_len] = -100

    moved = {}
    for key, value in full_inputs.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    moved["labels"] = labels.to(device=device)
    return moved


@torch.no_grad()
def extract_video_features(vlm, vlm_inputs, frame_pooling: str):
    outputs = vlm.model(
        input_ids=vlm_inputs["input_ids"],
        attention_mask=vlm_inputs.get("attention_mask"),
        pixel_values=vlm_inputs.get("pixel_values"),
        pixel_attention_mask=vlm_inputs.get("pixel_attention_mask"),
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
    )
    ctx_features = outputs.image_hidden_states
    if ctx_features is None:
        raise ValueError("SmolVLM did not return image_hidden_states for the provided videos.")
    batch_size = vlm_inputs["input_ids"].shape[0]
    if ctx_features.ndim != 3:
        raise ValueError(
            f"Expected SmolVLM image_hidden_states to be rank-3, got shape {tuple(ctx_features.shape)}."
        )
    pixel_attention_mask = vlm_inputs.get("pixel_attention_mask")
    if pixel_attention_mask is None:
        raise ValueError("Expected pixel_attention_mask in SmolVLM inputs for video batching.")

    # `image_hidden_states` is flattened across all valid visual units in the batch.
    # Recover per-example counts from the pixel attention mask, then concatenate each
    # example's valid visual units into one long token sequence and pad across examples.
    valid_visual_units = pixel_attention_mask.view(batch_size, pixel_attention_mask.shape[1], -1)
    valid_visual_units = valid_visual_units.any(dim=-1).sum(dim=-1).tolist()
    if sum(valid_visual_units) != ctx_features.shape[0]:
        raise ValueError(
            "SmolVLM visual feature count mismatch: "
            f"sum(valid_visual_units)={sum(valid_visual_units)} "
            f"vs image_hidden_states.shape[0]={ctx_features.shape[0]}."
        )

    split_features = []
    offset = 0
    for n_units in valid_visual_units:
        sample_features = ctx_features[offset : offset + n_units]
        if frame_pooling == "mean":
            pooled_features = sample_features.mean(dim=1)
        elif frame_pooling == "flatten":
            pooled_features = sample_features.reshape(
                n_units * sample_features.shape[1],
                ctx_features.shape[-1],
            )
        else:
            raise ValueError(f"Unsupported frame_pooling={frame_pooling!r}")
        split_features.append(pooled_features)
        offset += n_units

    ctx_features = pad_sequence(split_features, batch_first=True, padding_value=0.0)
    ctx_attn_mask = torch.zeros(
        ctx_features.shape[:2], dtype=torch.long, device=ctx_features.device
    )
    ctx_position_ids = torch.zeros_like(ctx_attn_mask)
    for sample_idx, sample_features in enumerate(split_features):
        sample_len = sample_features.shape[0]
        ctx_attn_mask[sample_idx, :sample_len] = 1
        ctx_position_ids[sample_idx, :sample_len] = torch.arange(
            sample_len, dtype=torch.long, device=ctx_features.device
        )
    return ctx_features, ctx_attn_mask, ctx_position_ids


@torch.no_grad()
def extract_l2l_fused_text_features(
    vlm,
    vlm_inputs,
    num_target_layers: int,
):
    outputs = vlm.model(
        input_ids=vlm_inputs.get("input_ids"),
        attention_mask=vlm_inputs.get("attention_mask"),
        pixel_values=vlm_inputs.get("pixel_values"),
        pixel_attention_mask=vlm_inputs.get("pixel_attention_mask"),
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("SmolVLM text model did not return hidden_states.")

    # Match the existing Doc2LoRA l2l bias as closely as possible by using
    # embeddings through the input of the last attention block when available.
    if len(hidden_states) == num_target_layers + 1:
        selected_hidden_states = hidden_states[:-1]
    elif len(hidden_states) >= num_target_layers:
        selected_hidden_states = hidden_states[-num_target_layers:]
    else:
        raise ValueError(
            f"Need at least {num_target_layers} hidden-state tensors for layer-to-layer mode, "
            f"but only received {len(hidden_states)}."
        )

    ctx_features = torch.stack(selected_hidden_states, dim=1)
    ctx_attn_mask = vlm_inputs.get("attention_mask")
    if ctx_attn_mask is None:
        raise ValueError("Layer-to-layer mode requires attention_mask from SmolVLM inputs.")
    return ctx_features, ctx_attn_mask, None


def extract_context_features(
    vlm,
    vlm_inputs,
    *,
    ctx_feature_mode: str,
    frame_pooling: str,
    num_target_layers: int,
):
    if ctx_feature_mode == "visual_pooled":
        return extract_video_features(vlm, vlm_inputs, frame_pooling=frame_pooling)
    if ctx_feature_mode == "l2l_fused_text":
        return extract_l2l_fused_text_features(
            vlm,
            vlm_inputs,
            num_target_layers=num_target_layers,
        )
    raise ValueError(f"Unsupported ctx_feature_mode={ctx_feature_mode!r}")


def repeat_video_features_per_question(
    ctx_features: torch.Tensor,
    ctx_attn_mask: torch.Tensor,
    ctx_position_ids: torch.Tensor | None,
    question_counts: torch.Tensor,
):
    repeat_counts = question_counts.to(device=ctx_features.device, dtype=torch.long)
    repeated_position_ids = None
    if ctx_position_ids is not None:
        repeated_position_ids = ctx_position_ids.repeat_interleave(repeat_counts, dim=0)
    return (
        ctx_features.repeat_interleave(repeat_counts, dim=0),
        ctx_attn_mask.repeat_interleave(repeat_counts, dim=0),
        repeated_position_ids,
    )


def compute_video_centric_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    question_counts: torch.Tensor,
):
    shift_labels = nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    vocab_size = logits.shape[-1]
    token_losses = nn.functional.cross_entropy(
        logits.float().view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
    ).view(labels.shape[0], -1)
    token_mask = shift_labels.ne(-100)
    per_qa_loss = (token_losses * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
    per_video_loss = torch.stack(
        [
            chunk.mean()
            for chunk in torch.split(
                per_qa_loss,
                question_counts.detach().to("cpu", dtype=torch.long).tolist(),
            )
        ]
    )
    return per_video_loss.mean(), per_qa_loss, per_video_loss


def compute_teacher_kl_loss(
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
    question_counts: torch.Tensor,
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

    per_qa_kl = []
    for qa_idx in range(student_shift_logits.shape[0]):
        student_mask = student_shift_labels[qa_idx].ne(-100)
        teacher_mask = teacher_shift_labels[qa_idx].ne(-100)
        student_qa_logits = student_shift_logits[qa_idx][student_mask]
        teacher_qa_logits = teacher_shift_logits[qa_idx][teacher_mask]
        if student_qa_logits.shape[0] == 0 or teacher_qa_logits.shape[0] == 0:
            continue
        if student_qa_logits.shape[0] != teacher_qa_logits.shape[0]:
            min_len = min(student_qa_logits.shape[0], teacher_qa_logits.shape[0])
            student_qa_logits = student_qa_logits[:min_len]
            teacher_qa_logits = teacher_qa_logits[:min_len]
        teacher_logp = nn.functional.log_softmax(
            teacher_qa_logits.float() / temperature,
            dim=-1,
        )
        student_logp = nn.functional.log_softmax(
            student_qa_logits.float() / temperature,
            dim=-1,
        )
        teacher_p = teacher_logp.exp()
        kl_tokens = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)
        per_qa_kl.append(kl_tokens.mean())

    if not per_qa_kl:
        zero = student_logits.new_zeros(())
        return zero, zero.new_zeros((0,))

    per_qa_kl = torch.stack(per_qa_kl)
    per_video_kl = torch.stack(
        [
            chunk.mean()
            for chunk in torch.split(
                per_qa_kl,
                question_counts.detach().to("cpu", dtype=torch.long).tolist(),
            )
        ]
    )
    return per_video_kl.mean() * (temperature**2), per_video_kl


@torch.no_grad()
def evaluate(
    model,
    processor,
    vlm,
    data_loader: DataLoader,
    device: torch.device,
    frame_pooling: str,
    ctx_feature_mode: str,
    num_target_layers: int,
) -> dict[str, float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_loss = 0.0
    total_videos = 0
    total_questions = 0
    total_batches = 0
    total_answer_tokens = 0.0
    total_context_tokens = 0.0
    for batch in data_loader:
        text_batch = move_text_batch_to_device(batch, device)
        vlm_inputs = prepare_smolvlm_inputs(
            processor,
            batch["video_messages"],
            device,
            video_fps=batch["video_fps"],
            max_frames=batch["max_frames"],
            video_size_longest_edge=batch["video_size_longest_edge"],
            video_load_backend=args.video_load_backend,
        )
        ctx_features, ctx_attn_mask, ctx_position_ids = extract_context_features(
            vlm,
            vlm_inputs,
            ctx_feature_mode=ctx_feature_mode,
            frame_pooling=frame_pooling,
            num_target_layers=num_target_layers,
        )
        ctx_features, ctx_attn_mask, ctx_position_ids = repeat_video_features_per_question(
            ctx_features,
            ctx_attn_mask,
            ctx_position_ids,
            text_batch["question_counts"],
        )
        outputs = model(
            ctx_features=ctx_features,
            ctx_attn_mask=ctx_attn_mask,
            ctx_position_ids=ctx_position_ids,
            n_ctx_chunks=torch.ones(ctx_features.shape[0], dtype=torch.int32, device=device),
            input_ids=text_batch["input_ids"],
            attention_mask=text_batch["attention_mask"],
            labels=text_batch["labels"],
        )
        raw_loss, _, _ = compute_video_centric_ce_loss(
            outputs.logits,
            text_batch["labels"],
            text_batch["question_counts"],
        )
        num_videos = text_batch["question_counts"].numel()
        num_questions = int(text_batch["question_counts"].sum().item())
        total_loss += raw_loss.item() * num_videos
        total_videos += num_videos
        total_questions += num_questions
        total_batches += 1
        total_answer_tokens += float((text_batch["labels"] != -100).sum().item())
        total_context_tokens += float(ctx_attn_mask.sum().item())
    model.train()
    if total_videos == 0:
        return {
            "loss": float("nan"),
            "ppl": float("nan"),
            "videos": 0.0,
            "questions": 0.0,
            "batches": 0.0,
        }
    mean_loss = total_loss / total_videos
    metrics = {
        "loss": mean_loss,
        "ppl": math.exp(min(mean_loss, 20)),
        "videos": float(total_videos),
        "questions": float(total_questions),
        "batches": float(total_batches),
        "questions_per_video_mean": total_questions / max(total_videos, 1),
        "answer_tokens_per_question_mean": total_answer_tokens / max(total_questions, 1),
        "context_tokens_per_video_mean": total_context_tokens / max(total_videos, 1),
        "context_layers_per_video_mean": float(num_target_layers)
        if ctx_feature_mode == "l2l_fused_text"
        else 1.0,
    }
    memory_metrics = get_peak_cuda_memory_metrics(device)
    if memory_metrics:
        metrics["memory_peak_allocated_gb"] = memory_metrics["memory/peak_allocated_gb"]
        metrics["memory_peak_reserved_gb"] = memory_metrics["memory/peak_reserved_gb"]
    return metrics


def save_checkpoint(model, output_dir: Path, step: int) -> Path:
    ckpt_path = output_dir / "checkpoints" / f"step-{step}.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def resolve_wandb_mode(requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    if os.environ.get("WANDB_DISABLED", "").lower() in {"true", "1", "yes"}:
        return "disabled"
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    return "offline"


def dataset_tag_from_manifest(manifest_path: str | None) -> str:
    if not manifest_path:
        return "unknown-dataset"
    stem = Path(manifest_path).stem
    for suffix in ("-train", "-val", "_train", "_val"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def get_peak_cuda_memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "memory/peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "memory/peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_layout(output_dir)
    set_seed(args.seed)

    if not os.path.exists(args.train_manifest):
        raise FileNotFoundError(f"Train manifest not found: {args.train_manifest}")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        print(
            "[gpu] "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
            f"current_device={torch.cuda.current_device()} "
            f"device_name={torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
    else:
        device = torch.device("cpu")
    train_rows = load_jsonl(args.train_manifest, args.max_train_samples)
    val_rows = (
        load_jsonl(args.val_manifest, args.max_val_samples) if args.val_manifest else []
    )

    model, base_tokenizer, processor, vlm = build_frames2lora_model(args, device)
    collator = SmolVLMOnlineCollator(
        base_tokenizer=base_tokenizer,
        video_fps=args.video_fps,
        max_frames=args.max_frames,
        video_size_longest_edge=args.video_size_longest_edge,
    )

    train_loader = DataLoader(
        VideoQADataset(
            train_rows,
            questions_per_video=args.questions_per_video,
            sample_questions_randomly=True,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = None
    if val_rows:
        val_loader = DataLoader(
            VideoQADataset(
                val_rows,
                questions_per_video=args.questions_per_video,
                sample_questions_randomly=False,
            ),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = (
        args.max_steps
        if args.max_steps is not None
        else math.ceil(len(train_loader) * args.epochs / args.grad_accum_steps)
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    run_name = args.wandb_run_name or output_dir.name
    wandb_mode = resolve_wandb_mode(args.wandb_mode)
    if wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
    print(f"Using wandb mode: {wandb_mode}")
    dataset_tag = dataset_tag_from_manifest(args.train_manifest)
    wandb_tags = [
        dataset_tag,
        f"base:{Path(args.base_lm_name_or_path).name}",
        f"vlm:{Path(args.smolvlm_name_or_path).name}",
        f"rank:{args.lora_r}",
        f"targets:{'-'.join(args.target_modules)}",
        f"qpv:{args.questions_per_video}",
        f"ctx:{args.ctx_feature_mode}",
        f"pool:{args.frame_pooling}",
    ]
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.wandb_group,
        notes=args.wandb_notes,
        tags=wandb_tags,
        config=asdict(args),
        mode=wandb_mode,
    )
    with open(output_dir / "train_args.json", "w") as f:
        json.dump(asdict(args), f, indent=2)
    dataset_metrics = {
        "data/train_questions": float(len(train_rows)),
        "data/train_videos": float(len(train_loader.dataset)),
        "data/val_questions": float(len(val_rows)),
        "data/val_videos": float(len(val_loader.dataset) if val_loader is not None else 0),
        "data/questions_per_video_cap": float(args.questions_per_video),
        "data/nominal_videos_per_optimizer_step": float(args.batch_size * args.grad_accum_steps),
        "data/nominal_questions_per_optimizer_step": float(
            args.batch_size * args.grad_accum_steps * args.questions_per_video
        ),
        "data/eval_videos_per_batch": float(args.eval_batch_size or 0),
    }
    wandb.log(dataset_metrics, step=0)
    print(
        "[data] "
        f"train_videos={int(dataset_metrics['data/train_videos'])} "
        f"train_questions={int(dataset_metrics['data/train_questions'])} "
        f"val_videos={int(dataset_metrics['data/val_videos'])} "
        f"val_questions={int(dataset_metrics['data/val_questions'])} "
        f"nominal_videos_per_optimizer_step={int(dataset_metrics['data/nominal_videos_per_optimizer_step'])} "
        f"nominal_questions_per_optimizer_step={int(dataset_metrics['data/nominal_questions_per_optimizer_step'])}"
    )

    def log_eval_metrics(metrics: dict[str, float], *, step: int, prefix: str) -> None:
        payload = {
            f"{prefix}/loss": metrics["loss"],
            f"{prefix}/ppl": metrics["ppl"],
            f"{prefix}/videos": metrics["videos"],
            f"{prefix}/questions": metrics["questions"],
            f"{prefix}/batches": metrics["batches"],
            f"{prefix}/questions_per_video_mean": metrics.get("questions_per_video_mean", 0.0),
            f"{prefix}/answer_tokens_per_question_mean": metrics.get(
                "answer_tokens_per_question_mean", 0.0
            ),
            f"{prefix}/context_tokens_per_video_mean": metrics.get(
                "context_tokens_per_video_mean", 0.0
            ),
            f"{prefix}/context_layers_per_video_mean": metrics.get(
                "context_layers_per_video_mean", 0.0
            ),
            f"{prefix}/memory_peak_allocated_gb": metrics.get("memory_peak_allocated_gb", 0.0),
            f"{prefix}/memory_peak_reserved_gb": metrics.get("memory_peak_reserved_gb", 0.0),
        }
        wandb.log(payload, step=step)
        print(
            f"[{prefix}] "
            f"step={step} loss={metrics['loss']:.4f} "
            f"videos={int(metrics['videos'])} "
            f"questions={int(metrics['questions'])} "
            f"peak_reserved_gb={metrics.get('memory_peak_reserved_gb', 0.0):.2f}"
        )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    stop_training = False
    accumulated_videos = 0
    accumulated_questions = 0
    accumulated_microbatches = 0
    accumulated_ce_loss = 0.0
    accumulated_kl_loss = 0.0
    accumulated_total_loss = 0.0
    accumulated_per_video_loss = 0.0
    accumulated_answer_tokens = 0.0
    accumulated_context_tokens = 0.0
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader, start=1):
            if device.type == "cuda" and accumulated_microbatches == 0:
                torch.cuda.reset_peak_memory_stats(device)
            text_batch = move_text_batch_to_device(batch, device)
            question_counts_f = text_batch["question_counts"].to(dtype=torch.float32)
            microbatch_videos = int(question_counts_f.numel())
            microbatch_questions = int(question_counts_f.sum().item())
            vlm_inputs = prepare_smolvlm_inputs(
                processor,
                batch["video_messages"],
                device,
                video_fps=batch["video_fps"],
                max_frames=batch["max_frames"],
                video_size_longest_edge=batch["video_size_longest_edge"],
                video_load_backend=args.video_load_backend,
            )
            with torch.no_grad():
                ctx_features, ctx_attn_mask, ctx_position_ids = extract_context_features(
                    vlm,
                    vlm_inputs,
                    ctx_feature_mode=args.ctx_feature_mode,
                    frame_pooling=args.frame_pooling,
                    num_target_layers=model.hypernet.n_layers,
                )
                per_video_ctx_attn_mask = ctx_attn_mask
                ctx_features, ctx_attn_mask, ctx_position_ids = repeat_video_features_per_question(
                    ctx_features,
                    ctx_attn_mask,
                    ctx_position_ids,
                    text_batch["question_counts"],
                )

            outputs = model(
                ctx_features=ctx_features,
                ctx_attn_mask=ctx_attn_mask,
                ctx_position_ids=ctx_position_ids,
                n_ctx_chunks=torch.ones(
                    ctx_features.shape[0],
                    dtype=torch.int32,
                    device=device,
                ),
                input_ids=text_batch["input_ids"],
                attention_mask=text_batch["attention_mask"],
                labels=text_batch["labels"],
            )
            raw_loss, _, per_video_loss = compute_video_centric_ce_loss(
                outputs.logits,
                text_batch["labels"],
                text_batch["question_counts"],
            )
            kl_loss = outputs.logits.new_zeros(())
            if args.kl_weight > 0:
                with torch.no_grad():
                    teacher_batch = prepare_smolvlm_teacher_batch(
                        processor,
                        batch["teacher_prompt_messages"],
                        batch["teacher_full_messages"],
                        device,
                        video_fps=batch["video_fps"],
                        max_frames=batch["max_frames"],
                        video_size_longest_edge=batch["video_size_longest_edge"],
                        video_load_backend=args.video_load_backend,
                    )
                    teacher_outputs = vlm(
                        input_ids=teacher_batch["input_ids"],
                        attention_mask=teacher_batch.get("attention_mask"),
                        pixel_values=teacher_batch.get("pixel_values"),
                        pixel_attention_mask=teacher_batch.get("pixel_attention_mask"),
                        labels=None,
                        return_dict=True,
                        use_cache=False,
                    )
                kl_loss, _ = compute_teacher_kl_loss(
                    outputs.logits,
                    text_batch["labels"],
                    teacher_outputs.logits,
                    teacher_batch["labels"],
                    text_batch["question_counts"],
                    temperature=args.kl_temperature,
                )
            total_loss = raw_loss + args.kl_weight * kl_loss
            loss = total_loss / args.grad_accum_steps
            loss.backward()

            accumulated_videos += microbatch_videos
            accumulated_questions += microbatch_questions
            accumulated_microbatches += 1
            accumulated_ce_loss += raw_loss.item()
            accumulated_kl_loss += kl_loss.item() if args.kl_weight > 0 else 0.0
            accumulated_total_loss += total_loss.item()
            accumulated_per_video_loss += per_video_loss.mean().item()
            accumulated_answer_tokens += float((text_batch["labels"] != -100).sum().item())
            accumulated_context_tokens += float(per_video_ctx_attn_mask.sum().item())

            question_counts_min = question_counts_f.min().item()
            question_counts_max = question_counts_f.max().item()
            question_counts_mean = question_counts_f.mean().item()

            def log_train_metrics(epoch_value: float) -> None:
                memory_metrics = get_peak_cuda_memory_metrics(device)
                avg_den = max(accumulated_microbatches, 1)
                metrics = {
                    "train/loss": accumulated_total_loss / avg_den,
                    "train/ce_loss": accumulated_ce_loss / avg_den,
                    "train/total_loss": accumulated_total_loss / avg_den,
                    "train/kl_loss": accumulated_kl_loss / avg_den,
                    "train/weighted_kl_loss": args.kl_weight * (accumulated_kl_loss / avg_den),
                    "train/per_video_loss": accumulated_per_video_loss / avg_den,
                    "train/videos_per_microbatch": float(microbatch_videos),
                    "train/questions_per_microbatch": float(microbatch_questions),
                    "train/questions_per_video_mean": question_counts_mean,
                    "train/questions_per_video_min": question_counts_min,
                    "train/questions_per_video_max": question_counts_max,
                    "train/answer_tokens_per_question_mean": accumulated_answer_tokens
                    / max(accumulated_questions, 1),
                    "train/context_tokens_per_video_mean": accumulated_context_tokens
                    / max(accumulated_videos, 1),
                    "train/context_layers_per_video_mean": float(model.hypernet.n_layers)
                    if args.ctx_feature_mode == "l2l_fused_text"
                    else 1.0,
                    "train/microbatches_per_optimizer_step": float(accumulated_microbatches),
                    "train/videos_per_optimizer_step": float(accumulated_videos),
                    "train/questions_per_optimizer_step": float(accumulated_questions),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch_value,
                    "train/step": global_step,
                    **memory_metrics,
                }
                wandb.log(metrics, step=global_step)
                if memory_metrics:
                    print(
                        "[train] "
                        f"step={global_step} total_loss={metrics['train/total_loss']:.4f} "
                        f"ce={metrics['train/ce_loss']:.4f} "
                        f"kl={metrics['train/kl_loss']:.4f} "
                        f"videos={microbatch_videos} "
                        f"questions={microbatch_questions} "
                        f"opt_videos={accumulated_videos} "
                        f"opt_questions={accumulated_questions} "
                        f"ctx_tokens_per_video={metrics['train/context_tokens_per_video_mean']:.2f} "
                        f"peak_reserved_gb={memory_metrics['memory/peak_reserved_gb']:.2f}"
                    )

            if step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    log_train_metrics(epoch + (step / max(len(train_loader), 1)))

                if val_loader is not None and global_step % args.eval_every == 0:
                    metrics = evaluate(
                        model,
                        processor,
                        vlm,
                        val_loader,
                        device,
                        frame_pooling=args.frame_pooling,
                        ctx_feature_mode=args.ctx_feature_mode,
                        num_target_layers=model.hypernet.n_layers,
                    )
                    log_eval_metrics(metrics, step=global_step, prefix="val")

                if global_step % args.save_every == 0:
                    ckpt_path = save_checkpoint(model, output_dir, global_step)
                    wandb.log({"checkpoint/step": global_step}, step=global_step)
                    print(f"Saved checkpoint to {ckpt_path}")

                accumulated_videos = 0
                accumulated_questions = 0
                accumulated_microbatches = 0
                accumulated_ce_loss = 0.0
                accumulated_kl_loss = 0.0
                accumulated_total_loss = 0.0
                accumulated_per_video_loss = 0.0
                accumulated_answer_tokens = 0.0
                accumulated_context_tokens = 0.0

                if args.max_steps is not None and global_step >= args.max_steps:
                    stop_training = True
                    break

        if stop_training:
            break

        if step % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.log_every == 0:
                log_train_metrics(epoch + 1)

            if val_loader is not None and global_step % args.eval_every == 0:
                metrics = evaluate(
                    model,
                    processor,
                    vlm,
                    val_loader,
                    device,
                    frame_pooling=args.frame_pooling,
                    ctx_feature_mode=args.ctx_feature_mode,
                    num_target_layers=model.hypernet.n_layers,
                )
                log_eval_metrics(metrics, step=global_step, prefix="val")

            if global_step % args.save_every == 0:
                ckpt_path = save_checkpoint(model, output_dir, global_step)
                wandb.log({"checkpoint/step": global_step}, step=global_step)
                print(f"Saved checkpoint to {ckpt_path}")

            accumulated_videos = 0
            accumulated_questions = 0
            accumulated_microbatches = 0
            accumulated_ce_loss = 0.0
            accumulated_kl_loss = 0.0
            accumulated_total_loss = 0.0
            accumulated_per_video_loss = 0.0
            accumulated_answer_tokens = 0.0
            accumulated_context_tokens = 0.0

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    final_ckpt = save_checkpoint(model, output_dir, global_step or 0)
    if val_loader is not None:
        metrics = evaluate(
            model,
            processor,
            vlm,
            val_loader,
            device,
            frame_pooling=args.frame_pooling,
            ctx_feature_mode=args.ctx_feature_mode,
            num_target_layers=model.hypernet.n_layers,
        )
        log_eval_metrics(metrics, step=max(global_step, 1), prefix="val/final")
    wandb.finish()
    print(f"Training finished. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
