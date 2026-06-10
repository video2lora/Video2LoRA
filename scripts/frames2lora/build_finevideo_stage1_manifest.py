import argparse
import os
import random
from pathlib import Path

from ctx_to_lora.data.finevideo import (
    candidates_to_rows,
    collect_finevideo_span_pools,
    materialize_span_clips,
    sample_fixed_count,
    sample_mixture,
    write_finevideo_manifests,
)


def parse_args():
    data_root = Path(os.environ.get("FRAMES2LORA_DATA_ROOT", "data/frames2lora"))
    finevideo_root = data_root / "raw" / "finevideo"
    processed_root = data_root / "processed" / "finevideo"
    parser = argparse.ArgumentParser(
        description="Build exact-span Stage 1 manifests for FineVideo with ffmpeg-cut scene/adjacent clips."
    )
    parser.add_argument("--train-root", default=str(finevideo_root / "train"))
    parser.add_argument("--val-root", default=str(finevideo_root / "val"))
    parser.add_argument(
        "--clips-root",
        default=str(data_root / "raw" / "finevideo_stage1_spans"),
    )
    parser.add_argument(
        "--train-out",
        default=str(processed_root / "train.jsonl"),
    )
    parser.add_argument(
        "--val-out",
        default=str(processed_root / "val.jsonl"),
    )
    parser.add_argument(
        "--val-scene-out",
        default=str(processed_root / "val_scene.jsonl"),
    )
    parser.add_argument(
        "--val-adjacent-out",
        default=str(processed_root / "val_adjacent.jsonl"),
    )
    parser.add_argument(
        "--val-full-out",
        default=str(processed_root / "val_full.jsonl"),
    )
    parser.add_argument(
        "--val-core-out",
        default=str(processed_root / "val_core.jsonl"),
    )
    parser.add_argument(
        "--val-gen-out",
        default=str(processed_root / "val_gen_100.jsonl"),
    )
    parser.add_argument(
        "--train-target",
        type=int,
        default=0,
        help="Target number of train examples with 60/30/10 span mix. 0 means max unique possible.",
    )
    parser.add_argument("--scene-ratio", type=float, default=0.60)
    parser.add_argument("--adjacent-ratio", type=float, default=0.30)
    parser.add_argument("--full-ratio", type=float, default=0.10)
    parser.add_argument("--val-scene-size", type=int, default=384)
    parser.add_argument("--val-adjacent-size", type=int, default=384)
    parser.add_argument("--val-full-size", type=int, default=384)
    parser.add_argument("--val-gen-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--ffmpeg-preset", default="ultrafast")
    parser.add_argument("--ffmpeg-crf", type=int, default=30)
    parser.add_argument("--ffmpeg-include-audio", action="store_true")
    parser.add_argument("--overwrite-clips", action="store_true")
    parser.add_argument("--skip-clip-materialization", action="store_true")
    return parser.parse_args()


def _stratified_val_gen_rows(
    val_scene_rows: list[dict],
    val_adjacent_rows: list[dict],
    val_full_rows: list[dict],
    *,
    val_gen_size: int,
    seed: int,
) -> list[dict]:
    if val_gen_size <= 0:
        return []
    pools = [
        rows
        for rows in (val_scene_rows, val_adjacent_rows, val_full_rows)
        if rows
    ]
    if not pools:
        return []
    base = val_gen_size // 3
    remainder = val_gen_size - base * 3
    scene_n = base + (1 if remainder > 0 else 0)
    adjacent_n = base + (1 if remainder > 1 else 0)
    full_n = base
    rng = random.Random(seed + 91)
    out: list[dict] = []
    if val_scene_rows:
        out.extend(rng.sample(val_scene_rows, min(scene_n, len(val_scene_rows))))
    if val_adjacent_rows:
        out.extend(rng.sample(val_adjacent_rows, min(adjacent_n, len(val_adjacent_rows))))
    if val_full_rows:
        out.extend(rng.sample(val_full_rows, min(full_n, len(val_full_rows))))
    if len(out) < val_gen_size:
        selected_ids = {row["id"] for row in out}
        remaining = [
            row
            for rows in pools
            for row in rows
            if row["id"] not in selected_ids
        ]
        rng.shuffle(remaining)
        out.extend(remaining[: val_gen_size - len(out)])
    rng.shuffle(out)
    return out[:val_gen_size]


def main():
    args = parse_args()
    random.seed(args.seed)

    train_pools = collect_finevideo_span_pools(
        args.train_root,
        split="train",
        clips_root=args.clips_root,
    )
    val_pools = collect_finevideo_span_pools(
        args.val_root,
        split="val",
        clips_root=args.clips_root,
    )

    print(
        "[pool] train scene/adjacent/full = "
        f"{len(train_pools['scene'])}/{len(train_pools['adjacent'])}/{len(train_pools['full'])}"
    )
    print(
        "[pool] val   scene/adjacent/full = "
        f"{len(val_pools['scene'])}/{len(val_pools['adjacent'])}/{len(val_pools['full'])}"
    )

    train_candidates = sample_mixture(
        train_pools,
        target_total=args.train_target,
        scene_ratio=args.scene_ratio,
        adjacent_ratio=args.adjacent_ratio,
        full_ratio=args.full_ratio,
        seed=args.seed,
    )

    val_scene = sample_fixed_count(
        val_pools["scene"],
        count=args.val_scene_size,
        seed=args.seed + 1,
    )
    val_adjacent = sample_fixed_count(
        val_pools["adjacent"],
        count=args.val_adjacent_size,
        seed=args.seed + 2,
    )
    val_full = sample_fixed_count(
        val_pools["full"],
        count=args.val_full_size,
        seed=args.seed + 3,
    )
    val_candidates = list(val_scene) + list(val_adjacent) + list(val_full)
    random.Random(args.seed + 4).shuffle(val_candidates)

    print(f"[select] train candidates: {len(train_candidates)}")
    print(
        "[select] val scene/adjacent/full = "
        f"{len(val_scene)}/{len(val_adjacent)}/{len(val_full)} (total={len(val_candidates)})"
    )

    if not args.skip_clip_materialization:
        all_candidates = train_candidates + val_candidates
        failures = materialize_span_clips(
            all_candidates,
            num_workers=args.num_workers,
            overwrite=args.overwrite_clips,
            ffmpeg_threads=args.ffmpeg_threads,
            ffmpeg_preset=args.ffmpeg_preset,
            ffmpeg_crf=args.ffmpeg_crf,
            include_audio=args.ffmpeg_include_audio,
        )
        if failures:
            print(f"[clip] failures: {len(failures)}")
            print("[clip] first failures:")
            for sample_id, error in failures[:20]:
                print(f"  {sample_id}: {error}")
            failure_ids = {sample_id for sample_id, _ in failures}
            train_candidates = [
                candidate for candidate in train_candidates if candidate.sample_id not in failure_ids
            ]
            val_scene = [candidate for candidate in val_scene if candidate.sample_id not in failure_ids]
            val_adjacent = [
                candidate for candidate in val_adjacent if candidate.sample_id not in failure_ids
            ]
            val_full = [candidate for candidate in val_full if candidate.sample_id not in failure_ids]
            val_candidates = list(val_scene) + list(val_adjacent) + list(val_full)
            random.Random(args.seed + 4).shuffle(val_candidates)

    train_rows = candidates_to_rows(train_candidates)
    val_scene_rows = candidates_to_rows(val_scene)
    val_adjacent_rows = candidates_to_rows(val_adjacent)
    val_full_rows = candidates_to_rows(val_full)
    val_rows = candidates_to_rows(val_candidates)
    val_gen_rows = _stratified_val_gen_rows(
        val_scene_rows,
        val_adjacent_rows,
        val_full_rows,
        val_gen_size=args.val_gen_size,
        seed=args.seed + 10,
    )

    write_finevideo_manifests(
        train_rows=train_rows,
        val_rows=val_rows,
        val_scene_rows=val_scene_rows,
        val_adjacent_rows=val_adjacent_rows,
        val_full_rows=val_full_rows,
        val_gen_rows=val_gen_rows,
        train_out=args.train_out,
        val_out=args.val_out,
        val_scene_out=args.val_scene_out,
        val_adjacent_out=args.val_adjacent_out,
        val_full_out=args.val_full_out,
        val_core_out=args.val_core_out,
        val_gen_out=args.val_gen_out,
    )

    print(f"Wrote train rows: {len(train_rows)} -> {args.train_out}")
    print(f"Wrote val rows: {len(val_rows)} -> {args.val_out}")
    print(f"Wrote val_scene rows: {len(val_scene_rows)} -> {args.val_scene_out}")
    print(f"Wrote val_adjacent rows: {len(val_adjacent_rows)} -> {args.val_adjacent_out}")
    print(f"Wrote val_full rows: {len(val_full_rows)} -> {args.val_full_out}")
    print(f"Wrote val_core rows: {min(1024, len(val_rows))} -> {args.val_core_out}")
    print(f"Wrote val_gen rows: {len(val_gen_rows)} -> {args.val_gen_out}")


if __name__ == "__main__":
    main()
