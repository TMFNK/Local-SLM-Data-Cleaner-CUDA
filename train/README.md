# Training: Qwen3-0.6B (instruct) with CUDA (Transformers + PEFT + TRL)

Fine-tune locally on the Lenovo's GTX 1650, then export to GGUF for llama.cpp.

## 0. Install

```bash
make setup       # installs requirements.txt + requirements-train.txt, checks the GPU
make llama-cpp   # builds llama.cpp with CUDA for conversion + serving
```

## 1. Data

`synth/generate.py` writes `data/train.jsonl`, `data/valid.jsonl`,
`data/test.jsonl` in chat format (`{"messages": [...]}`), which TRL's
`SFTTrainer` consumes directly.

```bash
python3 synth/generate.py --n 2000 --out data --seed 0
```

Start small and plot a learning curve (250 → 500 → 1k → 2k): data is free, so
train on increasing slices and stop where eval flattens.

## 2. LoRA fine-tune

```bash
python3 train/train_lora.py \
  --model Qwen/Qwen3-0.6B \
  --data data \
  --iters 1000 \
  --batch-size 4 \
  --adapter-path adapters
```

Flags: `--lr` (default 1e-4), `--grad-checkpoint` (slower, less VRAM — use
with `--batch-size 2` if you hit out-of-memory on the 4 GB card). LoRA targets
the attention projections (q/k/v/o), r=16, fp16 (Turing has no bf16).

Tip: run the model with thinking disabled for this task (terse JSON, not
chain-of-thought): pass `/no_think` in the system prompt if you keep Qwen3's
thinking template.

## 3. Merge + export

```bash
make fuse   # python3 train/merge_export.py -> fused/ (plain HF checkpoint)
make gguf   # convert_hf_to_gguf.py + llama-quantize -> qwen3-0.6b-cleaner-q8_0.gguf
```

## 4. Serve + evaluate

```bash
make serve   # llama-server -ngl 99 (all layers on the GPU)
make eval    # in a second terminal: score against data/test.jsonl
```
