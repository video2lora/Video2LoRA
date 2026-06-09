import os
import sys
import base64
import urllib.request
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

# Add repository root and src directory to sys.path to enable local imports
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
src_dir = str(Path(repo_root) / "src")
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import torch
import cv2
import matplotlib.pyplot as plt
from IPython.display import HTML, display

# Local repository imports
from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers
from ctx_to_lora.modeling.lora_merger import combine_lora
from scripts.video2lora.train_smolvlm_online import (
    prepare_smolvlm_inputs,
    extract_l2l_fused_text_features,
)


@dataclass
class Video2LoRAConfig:
    smolvlm_name_or_path: str
    train_manifest: str
    val_manifest: str
    output_dir: str
    lora_r: int = 16
    lora_dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    latent_size: int = 512
    dropout_rate: float = 0.0
    n_latent_queries: int = 8
    num_blocks: int = 9
    num_self_attn_per_block: int = 0
    video_fps: Optional[float] = None
    max_frames: int = 12
    video_size_longest_edge: int = 384
    video_load_backend: str = "auto"
    internalization_prompt: str = "Internalize this video for later captioning."
    kl_weight: float = 0.0
    generation_max_new_tokens: int = 128


def patch_num2words():
    """
    Force-refresh Hugging Face dependency cache for num2words if installed mid-session.
    """
    import importlib.util
    if importlib.util.find_spec("num2words") is not None:
        try:
            from num2words import num2words as num2words_func
            import transformers.utils.import_utils
            transformers.utils.import_utils._num2words_available = True
            for mod_name in list(sys.modules.keys()):
                if "processing_smolvlm" in mod_name or "smolvlm" in mod_name:
                    mod = sys.modules[mod_name]
                    if hasattr(mod, "num2words"):
                        setattr(mod, "num2words", num2words_func)
        except Exception:
            pass



def download_qualitative_videos(examples):
    """
    Download the qualitative videos from the project website repo if not found locally.
    """
    print("Checking/downloading qualitative benchmark video files from project website...")
    for item in examples:
        video_path = item["video_path"]
        dir_name = os.path.dirname(video_path)
        
        # 1. Clean up any broken symlinks or conflicting files along the directories path
        # to avoid FileNotFoundError during folder creation.
        parts = video_path.split('/')
        current_path = ""
        for part in parts[:-1]:  # skip the filename
            if current_path:
                current_path = os.path.join(current_path, part)
            else:
                current_path = part
            if os.path.lexists(current_path) and not os.path.isdir(current_path):
                print(f"Removing conflict at '{current_path}' (broken link or file) to allow directory creation...")
                try:
                    if os.path.islink(current_path):
                        os.unlink(current_path)
                    else:
                        os.remove(current_path)
                except Exception as e:
                    print(f"Failed to remove conflict at {current_path}: {e}")
        
        # 2. Safely create directory structure
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create directory {dir_name}: {e}")
            
        # 3. Check and retrieve the video file
        if not os.path.exists(video_path):
            url = f"https://video2lora.github.io/{video_path}"
            print(f"Downloading {video_path} from {url}...")
            try:
                urllib.request.urlretrieve(url, video_path)
                print("Download successful.")
            except Exception as e:
                print(f"Failed to download: {e}")
        else:
            print(f"Found local video file: {video_path}")

    print(f"\nLoaded {len(examples)} qualitative examples successfully.")


def show_video_frames(video_path, num_frames=4):
    """
    Helper function to load keyframes from a video path and plot them in a grid.
    """
    if not os.path.exists(video_path):
        print(f"Video file not found at: {video_path} (Please provide a valid video to inspect).")
        return
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    fig, axes = plt.subplots(1, num_frames, figsize=(16, 4))
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[i].imshow(frame)
            axes[i].axis('off')
            axes[i].set_title(f"Frame {idx}")
        else:
            axes[i].text(0.5, 0.5, "Frame Read Error", ha="center", va="center")
            axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    cap.release()


def run_internalization(example, model, raw_model, processor, config, device):
    """
    Extracts visual features layer-by-layer and generates the dynamic LoRA adapter.
    """
    # Ensure any patched LoRA hooks are cleared before feature extraction
    model.reset()

    internalize_messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": example["video_path"]},
                    {"type": "text", "text": config.internalization_prompt}
                ]
            }
        ]
    ]

    vlm_inputs = prepare_smolvlm_inputs(
        processor,
        internalize_messages,
        device,
        video_fps=config.video_fps,
        max_frames=config.max_frames,
        video_size_longest_edge=config.video_size_longest_edge,
        video_load_backend=config.video_load_backend
    )


    ctx_features, ctx_attn_mask, ctx_position_ids = extract_l2l_fused_text_features(
        raw_model,
        vlm_inputs,
        num_target_layers=model.hypernet.n_layers
    )

    generated_loras, _ = model.generate_weights(
        ctx_ids=None,
        ctx_features=ctx_features,
        ctx_attn_mask=ctx_attn_mask,
        ctx_position_ids=ctx_position_ids
    )

    generated_loras = combine_lora(
        generated_loras,
        torch.ones(1, dtype=torch.int32, device=device),
        lora_bias=model.hypernet.get_head_bias() if model.hypernet.config.use_bias else None
    )

    return generated_loras


def display_comparison(
    video_path,
    question_prompt,
    ground_truth,
    base_model_output,
    video2lora_output,
    dataset_name
):
    """
    Renders a beautifully styled comparison board with local HTML5 video player.
    """
    # Use the hosted website URL directly to avoid embedding massive base64 binaries
    # that would exceed GitHub's 100 MB file limit.
    if video_path.startswith(("http://", "https://")):
        video_src = video_path
    else:
        video_src = f"https://video2lora.github.io/{video_path}"

        
    html_content = f"""
    <div style="font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 900px; margin: 20px auto; color: #1e293b; background-color: #f8fafc; padding: 24px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.05);">
        
        <!-- Header Source Badge -->
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <h3 style="margin: 0; font-size: 18px; font-weight: 700; color: #0f172a;">Qualitative Comparison</h3>
            <span style="background-color: #e2e8f0; color: #475569; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600;">{dataset_name}</span>
        </div>

        <!-- Local HTML5 Video Player -->
        <div style="text-align: center; margin-bottom: 20px; background: #000; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
            <video src="{video_src}" controls loop muted autoplay style="width: 100%; max-height: 400px; object-fit: contain; display: block; margin: 0 auto;"></video>
        </div>
        <div style="text-align: center; margin-top: -12px; margin-bottom: 16px;">
            <a href="{video_src}" target="_blank" style="color: #2563eb; font-size: 13px; font-weight: 500; text-decoration: none;">🔗 Click here to open video file directly (GitHub preview fallback)</a>
        </div>
        
        <!-- Question Prompt Card -->
        <div style="background-color: #f0f4ff; border: 1px solid #dbeafe; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
            <h5 style="margin: 0 0 6px 0; color: #1e40af; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; font-weight: 700;">QUESTION PROMPT</h5>
            <p style="margin: 0; font-size: 15px; line-height: 1.5; color: #1e293b;">{question_prompt}</p>
        </div>
        
        <!-- Ground Truth Card -->
        <div style="background-color: #fefbeb; border: 1px solid #fef3c7; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
            <h5 style="margin: 0 0 6px 0; color: #854d0e; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; font-weight: 700;">GROUND TRUTH</h5>
            <p style="margin: 0; font-size: 15px; line-height: 1.5; color: #1e293b; font-weight: 500;">{ground_truth}</p>
        </div>
        
        <!-- Model Comparisons Grid -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            
            <!-- Base Model Card -->
            <div style="background-color: #fff; border: 1.5px solid #ea580c; border-radius: 8px; padding: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                <h5 style="margin: 0 0 6px 0; color: #ea580c; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; font-weight: 700;">BASE MODEL</h5>
                <p style="margin: 0; font-size: 14px; line-height: 1.5; color: #334155;">{base_model_output}</p>
            </div>
            
            <!-- Video2LoRA Card -->
            <div style="background-color: #f0fdf4; border: 1.5px solid #16a34a; border-radius: 8px; padding: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                <h5 style="margin: 0 0 6px 0; color: #16a34a; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; font-weight: 700;">VIDEO2LORA</h5>
                <p style="margin: 0; font-size: 14px; line-height: 1.5; color: #1e293b; font-weight: 500;">{video2lora_output}</p>
            </div>
            
        </div>
        
    </div>
    """
    display(HTML(html_content))


def run_inference_and_compare(example, model, raw_model, processor, tokenizer, config, device):
    """
    Runs both base model inference (with visual tokens) and Video2LoRA inference
    (zero visual tokens) for a single example, and displays their side-by-side comparison.
    """
    import gc

    # 1. Base Model Inference (with visual tokens)
    model.reset()
    torch.cuda.empty_cache()
    gc.collect()

    base_messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": example["video_path"]},
                    {"type": "text", "text": example["prompt"]}
                ]
            }
        ]
    ]

    base_inputs = prepare_smolvlm_inputs(
        processor,
        base_messages,
        device,
        video_fps=config.video_fps,
        max_frames=config.max_frames,
        video_size_longest_edge=config.video_size_longest_edge,
        video_load_backend=config.video_load_backend
    )

    with torch.inference_mode():
        base_generated_ids = raw_model.generate(
            **base_inputs,
            max_new_tokens=config.generation_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        base_pred = tokenizer.decode(
            base_generated_ids[0][base_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # 2. Video2LoRA Inference (zero visual tokens)
        generated_loras = run_internalization(example, model, raw_model, processor, config, device)

        # Inject dynamic weights
        model.patch_lora_forward()
        apply_lora_to_layers(
            model.base_model,
            model.hypernet.layer_indices,
            generated_loras,
            torch.ones(1, dtype=torch.int32, device=device),
            position_ids=None
        )

        prompt_ids = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": example["prompt"]}]
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        if hasattr(prompt_ids, "keys"):
            input_ids = prompt_ids["input_ids"]
            attention_mask = prompt_ids.get("attention_mask", torch.ones_like(input_ids))
        else:
            input_ids = prompt_ids
            attention_mask = torch.ones_like(input_ids)

        generated_ids = model.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config.generation_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        video2lora_pred = tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

    # Reset model and clean up
    model.reset()
    torch.cuda.empty_cache()
    gc.collect()

    # Display side-by-side comparison
    display_comparison(
        video_path=example["video_path"],
        question_prompt=example["prompt"],
        ground_truth=example["target_text"],
        base_model_output=base_pred,
        video2lora_output=video2lora_pred,
        dataset_name=example["dataset"]
    )

