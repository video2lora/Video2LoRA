import os
import json
from pathlib import Path
from typing import Any


DATA_ROOT = Path(os.environ.get("FRAMES2LORA_DATA_ROOT", "data/frames2lora"))

CAPTION_PROMPTS = (
    "Describe what is happening in this video.",
    "Write a short caption for this clip.",
    "Summarize this video in one sentence.",
    "Describe the main event in this clip.",
)

DEFAULT_INTERNALIZATION_PROMPT = "Internalize this video for later captioning."


def resolve_video_path(video_path_str: str) -> str:
    video_path = Path(video_path_str)
    if not video_path.is_absolute():
        video_path = DATA_ROOT / video_path
    return str(video_path)


def relativize_video_path(video_path_str: str) -> str:
    video_path = Path(video_path_str)
    if not video_path.is_absolute():
        return str(video_path)
    try:
        return str(video_path.relative_to(DATA_ROOT))
    except ValueError:
        return str(video_path)


def dump_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def infer_split(row: dict[str, Any], default_split: str | None = None) -> str | None:
    if row.get("split") is not None:
        return str(row["split"])
    sample_id = str(row.get("id", ""))
    for candidate in ("train", "val", "test"):
        if sample_id.startswith(f"{candidate}-") or sample_id.startswith(f"{candidate}_"):
            return candidate
    return default_split


def infer_dataset(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if row.get("dataset") is not None:
        return str(row["dataset"])
    if metadata.get("dataset") is not None:
        return str(metadata["dataset"])
    return "unknown"


def normalize_manifest_row(
    row: dict[str, Any],
    *,
    default_split: str | None = None,
) -> dict[str, Any]:
    prompt = row.get("prompt")
    target_text = row.get("target_text")
    task_type = row.get("task_type")

    if prompt is None and "question" in row:
        prompt = row["question"]
    if target_text is None and "answer" in row:
        target_text = row["answer"]
    if task_type is None:
        task_type = "qa" if "question" in row or "answer" in row else "caption"

    if prompt is None:
        raise ValueError(f"Missing prompt/question in row: {row}")
    if target_text is None:
        raise ValueError(f"Missing target_text/answer in row: {row}")
    if "video_path" not in row:
        raise ValueError(f"Missing video_path in row: {row}")

    metadata = dict(row.get("metadata") or {})
    dataset = infer_dataset(row)
    if "dataset" not in metadata:
        metadata["dataset"] = dataset

    normalized = {
        "id": str(row.get("id", Path(str(row["video_path"])).stem)),
        "video_path": relativize_video_path(str(row["video_path"])),
        "task_type": str(task_type),
        "prompt": str(prompt),
        "target_text": str(target_text),
        "dataset": dataset,
        "split": infer_split(row, default_split=default_split),
        "metadata": metadata,
    }

    for key in ("clip_start_sec", "clip_end_sec"):
        if row.get(key) is not None:
            normalized[key] = float(row[key])

    return normalized


def load_video_manifest(
    path: str | Path,
    *,
    max_samples: int | None = None,
    default_split: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(
                normalize_manifest_row(
                    json.loads(line),
                    default_split=default_split,
                )
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows
