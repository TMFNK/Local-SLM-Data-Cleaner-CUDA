"""
train/merge_export.py: merge the LoRA adapter into the base weights.

CUDA-project replacement for `mlx_lm.fuse`. Output is a plain HF checkpoint
that llama.cpp's convert_hf_to_gguf.py understands.

Usage:
    python3 train/merge_export.py --adapter-path adapters --save-path fused
"""
from __future__ import annotations
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter-path", default="adapters")
    ap.add_argument("--save-path", default="fused")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f">> loading base model {args.model} (fp16, CPU is fine for merging)")
    base = AutoModelForCausalLM.from_pretrained(args.model,
                                                torch_dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, args.adapter_path).merge_and_unload()
    merged.save_pretrained(args.save_path)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.save_path)
    print(f">> merged model saved to {args.save_path}")


if __name__ == "__main__":
    main()
