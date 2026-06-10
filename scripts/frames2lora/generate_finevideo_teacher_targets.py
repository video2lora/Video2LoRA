import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils import is_av_available, is_cv2_available, is_decord_available

from ctx_to_lora.data.video_manifest import load_video_manifest, dump_jsonl, resolve_video_path


def parse_args():
    parser = argparse.ArgumentParser(description='Generate offline teacher targets for FineVideo manifests.')
    parser.add_argument('--input-manifest', required=True)
    parser.add_argument('--output-manifest', required=True)
    parser.add_argument('--smolvlm-name-or-path', default='HuggingFaceTB/SmolVLM2-2.2B-Instruct')
    parser.add_argument('--per-device-batch-size', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--max-frames', type=int, default=12)
    parser.add_argument('--video-size-longest-edge', type=int, default=384)
    parser.add_argument('--video-fps', type=float, default=None)
    parser.add_argument(
        '--video-load-backend',
        default='auto',
        choices=('auto', 'decord', 'pyav', 'opencv', 'torchvision'),
    )
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--flush-every', type=int, default=256)
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--merge-only', action='store_true')
    return parser.parse_args()


def is_skippable_video_error(exc: BaseException) -> bool:
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, UnicodeDecodeError):
            return True
        if isinstance(cur, IndexError) and 'tuple index out of range' in str(cur).lower():
            return True
        err_name = cur.__class__.__name__.lower()
        err_mod = cur.__class__.__module__.lower()
        msg = str(cur).lower()
        if 'av.' in err_mod or 'ffmpeg' in err_mod or 'pyav' in err_mod:
            return True
        if 'invalid total_num_frames' in msg:
            return True
        if 'cannot open' in msg and 'video' in msg:
            return True
        if 'error opening input file' in msg:
            return True
        if 'moov atom not found' in msg:
            return True
        if 'invalid data found when processing input' in msg:
            return True
        if 'unsupported codec' in msg:
            return True
        if 'decode' in err_name and 'error' in err_name:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def prepare_generation_inputs(
    processor,
    messages,
    device,
    *,
    max_frames,
    video_fps,
    video_size_longest_edge,
    video_load_backend,
):
    processor.tokenizer.padding_side = 'left'
    chat_template_kwargs = dict(padding=True)
    processor_name = type(processor).__name__.lower()
    if 'idefics3' in processor_name:
        chat_template_kwargs['num_frames'] = max_frames
        if video_fps is not None:
            chat_template_kwargs['video_fps'] = video_fps
    else:
        chat_template_kwargs['max_frames'] = max_frames
        if video_fps is not None:
            chat_template_kwargs['target_fps'] = video_fps
    if video_load_backend == 'auto':
        if is_decord_available():
            video_load_backend = 'decord'
        elif is_av_available():
            video_load_backend = 'pyav'
        elif is_cv2_available():
            video_load_backend = 'opencv'
        else:
            video_load_backend = 'torchvision'
    chat_template_kwargs['video_load_backend'] = video_load_backend
    if video_size_longest_edge is not None:
        video_size = {'longest_edge': video_size_longest_edge}
        processor.video_size = video_size
        processor.image_processor.size = video_size
    prompt_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
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


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def generate_rows(
    model,
    processor,
    tokenizer,
    rows: list[dict[str, Any]],
    device: str,
    args,
) -> list[dict[str, Any]]:
    messages = []
    for row in rows:
        messages.append([
            {
                'role': 'user',
                'content': [
                    {'type': 'video', 'path': row['video_path']},
                    {'type': 'text', 'text': row['prompt']},
                ],
            }
        ])
    inputs = prepare_generation_inputs(
        processor,
        messages,
        device,
        max_frames=args.max_frames,
        video_fps=args.video_fps,
        video_size_longest_edge=args.video_size_longest_edge,
        video_load_backend=args.video_load_backend,
    )
    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    input_len = inputs['input_ids'].shape[1]
    generated_only = generated[:, input_len:]
    decoded = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

    out_rows = []
    for row, target_text in zip(rows, decoded, strict=True):
        out_row = dict(row)
        out_row['target_text'] = target_text.strip() or 'Unable to determine.'
        metadata = dict(out_row.get('metadata') or {})
        metadata['teacher_model'] = args.smolvlm_name_or_path
        metadata['teacher_generated'] = True
        metadata['teacher_prompt'] = out_row['prompt']
        metadata['teacher_max_new_tokens'] = args.max_new_tokens
        out_row['metadata'] = metadata
        out_rows.append(out_row)
    return out_rows


def merge_shards(output_path: Path, tmp_dir: Path) -> int:
    merged = []
    for shard_file in sorted(tmp_dir.glob(f'{output_path.stem}.rank*.jsonl')):
        with open(shard_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    merged.append(json.loads(line))
    dump_jsonl(output_path, merged)
    return len(merged)


def main():
    args = parse_args()
    output_path = Path(args.output_manifest)
    tmp_dir = Path(args.output_dir) if args.output_dir else output_path.parent / f'.tmp-{output_path.stem}'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merged = merge_shards(output_path, tmp_dir)
        print(f'[merge] wrote {merged} rows to {output_path}', flush=True)
        return

    rows = load_video_manifest(args.input_manifest, max_samples=args.max_samples)
    world = args.num_shards
    rank = args.shard_index
    if rank < 0 or rank >= world:
        raise ValueError(f'Invalid shard index {rank} for num_shards={world}')
    shard = rows[rank::world]
    shard_path = tmp_dir / f'{output_path.stem}.rank{rank:02d}.jsonl'

    processor = AutoProcessor.from_pretrained(args.smolvlm_name_or_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForImageTextToText.from_pretrained(
        args.smolvlm_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to('cuda').eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, 'generation_config', None):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    written = 0
    skipped = 0
    started = time.time()
    buffer: list[dict[str, Any]] = []

    with open(shard_path, 'w') as fout:
        for batch_rows in chunked(shard, args.per_device_batch_size):
            normalized_rows = []
            for row in batch_rows:
                row = dict(row)
                row['video_path'] = resolve_video_path(row['video_path'])
                normalized_rows.append(row)
            try:
                out_rows = generate_rows(model, processor, tokenizer, normalized_rows, 'cuda', args)
                buffer.extend(out_rows)
                written += len(out_rows)
            except Exception as exc:  # pylint: disable=broad-except
                if len(normalized_rows) > 1:
                    for row in normalized_rows:
                        try:
                            out_rows = generate_rows(model, processor, tokenizer, [row], 'cuda', args)
                            buffer.extend(out_rows)
                            written += len(out_rows)
                        except Exception as single_exc:  # pylint: disable=broad-except
                            if not is_skippable_video_error(single_exc):
                                raise
                            skipped += 1
                            print(f'[rank{rank}] skipped {row["id"]}: {type(single_exc).__name__}: {single_exc}', flush=True)
                    continue
                if not is_skippable_video_error(exc):
                    raise
                skipped += len(normalized_rows)
                for row in normalized_rows:
                    print(f'[rank{rank}] skipped {row["id"]}: {type(exc).__name__}: {exc}', flush=True)
            if len(buffer) >= args.flush_every:
                for row in buffer:
                    fout.write(json.dumps(row) + '\n')
                fout.flush()
                buffer.clear()
                elapsed = max(time.time() - started, 1e-6)
                rate = written / elapsed
                print(f'[rank{rank}] written={written} skipped={skipped} rate={rate:.2f} rows/s', flush=True)
        for row in buffer:
            fout.write(json.dumps(row) + '\n')
        fout.flush()


if __name__ == '__main__':
    main()
