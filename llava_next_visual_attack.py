"""
Visual soft-prompt attack retargeted to LLaVA-NeXT (llama3-llava-next-8b-hf).

Port of `llava_llama_v2_visual_attack.py` to the HuggingFace LlavaNext API, so the
adversarial images work on the SAME model that `prompt-injection-agentic-ai`
analyzes. Adds a configurable spatial mask so the perturbation ("prompt injection")
is confined to a small region of the image.

Workflow (run on a GPU server):
  1) Sanity-check preprocessing matches HF (cheap, no training):
       python llava_next_visual_attack.py --sanity_check \
         --image_file clean_images/coco_4.jpg
  2) Quick smoke run to confirm loss decreases:
       python llava_next_visual_attack.py ... --n_iters 50 --eval_every 25
  3) Full run:
       python llava_next_visual_attack.py ... --n_iters 2000

See build_pixel_values / sanity_check_preprocess in llava_next_utils/preprocess.py.
"""
import argparse
import csv
import os
import random

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from llava_next_utils.preprocess import (
    build_pixel_values,
    build_region_mask,
    sanity_check_preprocess,
)

MODEL_ID = "llava-hf/llama3-llava-next-8b-hf"

# maps CLI instruction names to the CSV column that holds the target answers
INSTRUCTION_MAP = {"french": "fr", "english": "en", "spanish": "es",
                   "right": "Republican", "left": "Democrat"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default=MODEL_ID)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--data_path", type=str,
                   default="instruction_data/coco_4/Attack/dataset.csv")
    p.add_argument("--instruction", type=str, default="injection",
                   choices=["positive", "negative", "neutral", "formal", "informal",
                            "french", "english", "spanish", "left", "right",
                            "injection", "spam"])
    p.add_argument("--image_file", type=str, default="clean_images/coco_4.jpg")
    p.add_argument("--save_dir", type=str, default="output/llava_next/coco_4/Attack/injection")
    p.add_argument("--n_iters", type=int, default=2000)
    p.add_argument("--num_rows", type=int, default=40, help="how many CSV rows to train on")
    p.add_argument("--batch_size", type=int, default=4,
                   help="rows sampled per iteration; lower this if you OOM")
    p.add_argument("--eps", type=float, default=32.0, help="L-inf budget (used as eps/255)")
    p.add_argument("--alpha", type=float, default=1.0, help="PGD step size (used as alpha/255)")
    # region mask
    p.add_argument("--region_mode", type=str, default="margin",
                   choices=["margin", "bottom", "corner", "box"])
    p.add_argument("--region_frac", type=float, default=0.1,
                   help="band width (margin/bottom) or square side (corner) as fraction of image")
    p.add_argument("--region_x", type=int, default=0)
    p.add_argument("--region_y", type=int, default=0)
    p.add_argument("--region_w", type=int, default=0, help="0 = to right edge")
    p.add_argument("--region_h", type=int, default=0, help="0 = to bottom edge")
    # utilities
    p.add_argument("--sanity_check", action="store_true",
                   help="compare differentiable preprocessing vs HF processor and exit")
    p.add_argument("--eval_every", type=int, default=100,
                   help="print a sample generation + save checkpoint every N iters")
    return p.parse_args()


def read_csv_columns(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        # DictReader may leave a BOM on the first header; normalize keys
        cols = {(k.lstrip("﻿") if k else k): [r[k] for r in rows]
                for k in reader.fieldnames}
    return cols


def build_examples(processor, clean_pil, questions, targets, device):
    """
    Precompute per-row (input_ids, labels, attention_mask). These do NOT depend on
    the adversarial pixels (image-token count is fixed by image size), so we build
    them once. Image placeholder tokens are expanded here by the processor.
    """
    tok = processor.tokenizer
    examples = []
    for q, tgt in zip(questions, targets):
        conversation = [{"role": "user",
                         "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        enc = processor(images=clean_pil, text=prompt, return_tensors="pt")
        prompt_ids = enc["input_ids"][0]  # (Lp,) with <image> already expanded

        tgt_ids = torch.tensor(tok(tgt, add_special_tokens=False)["input_ids"],
                               dtype=prompt_ids.dtype)

        input_ids = torch.cat([prompt_ids, tgt_ids], dim=0)
        labels = torch.cat(
            [torch.full((prompt_ids.shape[0],), -100, dtype=input_ids.dtype), tgt_ids],
            dim=0,
        )
        attn = torch.ones_like(input_ids)
        examples.append({
            "input_ids": input_ids.unsqueeze(0).to(device),
            "labels": labels.unsqueeze(0).to(device),
            "attention_mask": attn.unsqueeze(0).to(device),
        })
    return examples


def main():
    args = parse_args()
    # reduce allocator fragmentation (must be set before CUDA context init)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    print(">>> Loading processor + model:", args.model_id)
    processor = LlavaNextProcessor.from_pretrained(args.model_id)
    clean_pil = Image.open(args.image_file).convert("RGB")

    # ---- sanity check preprocessing (no model needed) ----
    if args.sanity_check:
        res = sanity_check_preprocess(clean_pil, processor)
        print("=== preprocessing sanity check ===")
        for k, v in res.items():
            print(f"  {k}: {v}")
        # Judge on mean + p99, NOT raw max: on real photos a few sharp-edge pixels
        # disagree between torch and PIL bicubic (max ~0.1) while 99.9% of pixels
        # match to <0.015. Those isolated outliers are averaged out through the
        # model and do not affect attack transfer; a *systematic* mismatch shows
        # up in the mean / p99 instead.
        ok = (res.get("match_shape")
              and res.get("mean_abs_diff", 1) < 0.02
              and res.get("p99_abs_diff", 1) < 0.03)
        if ok:
            print(">>> OK: preprocessing matches HF. Isolated max-pixel diffs from "
                  "torch-vs-PIL bicubic are expected and harmless (see p99 / frac_over_0p05).")
        else:
            print(">>> WARNING: systematic mismatch (mean/p99 too high); attack may not transfer.")
        return

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=None,
    ).to(device)
    model.requires_grad_(False)
    # Gradient checkpointing trades compute for activation memory (recomputes
    # activations during backward instead of storing them) -- needed to fit the
    # 8B model's forward/backward on a single ~22GB GPU. HF only applies
    # checkpointing when model.training is True, so use train() instead of
    # eval() here; this model has 0 dropout by default so behavior is unaffected.
    model.gradient_checkpointing_enable()
    model.train()

    # ---- data ----
    cols = read_csv_columns(args.data_path)
    col = INSTRUCTION_MAP.get(args.instruction, args.instruction)
    question_key = list(cols.keys())[0]  # first column is the question
    questions = cols[question_key][:args.num_rows]
    targets = cols[col][:args.num_rows]
    print(f">>> {len(questions)} (question,target) pairs from column '{col}'")

    examples = build_examples(processor, clean_pil, questions, targets, device)

    # ---- adversarial variable + region mask ----
    os.makedirs(args.save_dir, exist_ok=True)
    img01 = to_tensor(clean_pil).float().to(device)          # (3,H,W) in [0,1]
    _, H, W = img01.shape
    mask = build_region_mask(H, W, args, device=device)      # (1,H,W)
    eps, alpha = args.eps / 255.0, args.alpha / 255.0

    delta = ((torch.rand_like(img01) * 2 - 1) * eps) * mask
    delta = delta.detach().requires_grad_(True)

    save_image(img01, os.path.join(args.save_dir, "clean_prompt.bmp"))
    print(f">>> image {H}x{W}, region '{args.region_mode}' covers "
          f"{int(mask.sum().item())}/{H*W} px ({100*mask.mean().item():.1f}%)")

    loss_buffer = []
    for t in range(args.n_iters + 1):
        batch = random.sample(examples, min(args.batch_size, len(examples)))
        delta.grad = None
        loss_val = 0.0
        # Per-sample gradient accumulation: backward after EACH forward so only one
        # sequence's activations live at a time -> peak memory is independent of
        # batch_size. Rebuilding pixel_values per sample is cheap vs the 8B forward.
        for ex in batch:
            x_adv = (img01 + mask * delta).clamp(0, 1)
            pixel_values, image_sizes = build_pixel_values(
                x_adv, processor, dtype=model.dtype, device=device)
            out = model(
                input_ids=ex["input_ids"],
                attention_mask=ex["attention_mask"],
                labels=ex["labels"],
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                use_cache=False,
            )
            (out.loss / len(batch)).backward()
            loss_val += out.loss.item() / len(batch)

        with torch.no_grad():
            g = delta.grad
            delta.data = (delta.data - alpha * g.sign() * mask).clamp(-eps, eps)
        delta.grad = None
        loss_buffer.append(loss_val)

        if t % 20 == 0:
            print(f"iter {t:4d}  loss {loss_val:.4f}")

        if args.eval_every and t % args.eval_every == 0:
            with torch.no_grad():
                x_adv = (img01 + mask * delta).clamp(0, 1)
            _save_checkpoint(args, x_adv, loss_buffer, t)
            _sample_generation(model, processor, x_adv, examples[0], device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    x_final = (img01 + mask * delta).clamp(0, 1)
    save_image(x_final, os.path.join(args.save_dir, "bad_prompt.bmp"))
    _save_loss_curve(args, loss_buffer)
    print(">>> [Done] saved", os.path.join(args.save_dir, "bad_prompt.bmp"))


def _save_checkpoint(args, x_adv, loss_buffer, t):
    save_image(x_adv.detach(),
               os.path.join(args.save_dir, f"bad_prompt_temp_{t}.bmp"))
    _save_loss_curve(args, loss_buffer)


def _save_loss_curve(args, loss_buffer):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(len(loss_buffer)), loss_buffer, label="target loss")
        plt.xlabel("iter"); plt.ylabel("loss"); plt.legend(loc="best")
        plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))
        plt.close()
        torch.save(loss_buffer, os.path.join(args.save_dir, "loss"))
    except Exception as e:
        print("  (loss curve skipped:", e, ")")


@torch.no_grad()
def _sample_generation(model, processor, x_adv, example, device):
    """Print a short generation on the current adversarial image for visibility."""
    # generate() needs eval mode + real KV caching; both get disturbed by the
    # train()/gradient-checkpointing state the training loop needs, so switch
    # out of that state here and restore it afterward.
    was_training = model.training
    model.gradient_checkpointing_disable()
    model.eval()
    try:
        with torch.no_grad():
            from llava_next_utils.preprocess import build_pixel_values
            pv, isz = build_pixel_values(x_adv.detach(), processor,
                                         dtype=model.dtype, device=device)
            # regenerate from the prompt part only (labels==-100 positions)
            labels = example["labels"][0]
            prompt_len = int((labels == -100).sum().item())
            prompt_ids = example["input_ids"][:, :prompt_len]
            gen = model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                pixel_values=pv, image_sizes=isz,
                max_new_tokens=40, do_sample=False,
            )
            text = processor.tokenizer.decode(gen[0, prompt_ids.shape[1]:],
                                              skip_special_tokens=True)
            print("   sample >>>", text.replace("\n", " "))
    except Exception as e:
        print("   (sample generation skipped:", e, ")")
    finally:
        model.gradient_checkpointing_enable()
        model.train(was_training)


if __name__ == "__main__":
    main()
