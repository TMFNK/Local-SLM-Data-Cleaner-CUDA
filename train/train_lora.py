"""
train/train_lora.py: LoRA fine-tune on CUDA (HF Transformers + PEFT + TRL).

CUDA replacement for the original project's `mlx_lm.lora` step. Reads the
chat-format JSONL that synth/generate.py writes and saves a PEFT adapter.

Usage:
    python3 train/train_lora.py --iters 1000 --batch-size 4
"""
from __future__ import annotations
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", default="data")
    ap.add_argument("--iters", type=int, default=1000, help="training steps")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--adapter-path", default="adapters")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="trade speed for VRAM if you hit out-of-memory")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA GPU visible. This trainer needs the Lenovo's "
                 "GTX 1650.\nCheck `nvidia-smi`, and that torch was installed "
                 "with CUDA support (pip on Linux includes it by default).")
    print(f">> GPU: {torch.cuda.get_device_name(0)}")

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("json", data_files={
        "train": os.path.join(args.data, "train.jsonl"),
        "valid": os.path.join(args.data, "valid.jsonl"),
    })

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    cfg = SFTConfig(
        output_dir=args.adapter_path,
        max_steps=args.iters,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=True,                      # GTX 1650 (Turing): fp16, no bf16
        gradient_checkpointing=args.grad_checkpoint,
        max_length=1024,
        logging_steps=25,
        eval_strategy="steps", eval_steps=200,
        save_strategy="steps", save_steps=200, save_total_limit=2,
        model_init_kwargs={"torch_dtype": "float16"},
        report_to=[],
    )
    trainer = SFTTrainer(model=args.model, args=cfg,
                         train_dataset=ds["train"], eval_dataset=ds["valid"],
                         peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.adapter_path)
    print(f">> adapter saved to {args.adapter_path}")


if __name__ == "__main__":
    main()
