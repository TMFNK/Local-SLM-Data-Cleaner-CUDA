# Local SLM Data Cleaner (CUDA edition)

Private CUDA port of `Local-SLM-Data-Cleaner`. Same idea — fine-tune Qwen3-0.6B
on synthetic data to clean messy SAP-style master data, fully local — but the
training and serving run on the Lenovo Legion (GTX 1650 4 GB) under Pop!_OS
instead of Apple MLX on the Mac.

Only the training stack changed: HF Transformers + PEFT (LoRA) + TRL replace
`mlx-lm`. The data generator, eval harness, `clean.py`, and the llama.cpp
serving/GGUF flow are identical to the original project.

## 1. Get the project onto the Lenovo

No GitHub — copy it directly (from the Mac):

```bash
scp -r ~/Documents/03-Eddie-Python-Projects/python/Local-SLM-Data-Cleaner-CUDA \
    <user>@<lenovo-ip>:~/projects/
# or copy the folder with a USB stick; nothing outside the folder is needed
```

## 2. Pop!_OS prerequisites (one-time)

```bash
nvidia-smi   # must show the GTX 1650. If missing:
sudo apt install system76-driver-nvidia && sudo reboot

sudo apt install python3-venv python3-pip cmake build-essential git
```

Optional but recommended: work inside a venv (`python3 -m venv .venv &&
source .venv/bin/activate`).

## 3. Run the pipeline (on the Lenovo)

```bash
make setup        # python deps + GPU visibility check
make llama-cpp    # clone + build llama.cpp with CUDA (once)
make model        # download Qwen3-0.6B (~1.2 GB, cached)
make data         # generate synthetic data (N=1000 default)
make sanity       # must report ~100%
make baseline-serve   # terminal 1: serve the STOCK model
make baseline         # terminal 2: the 'before' score
# stop terminal 1, then:
make train        # LoRA fine-tune on the GPU
make fuse gguf    # merge adapter, convert + quantize to GGUF
make serve        # terminal 1: serve YOUR model
make eval         # terminal 2: the 'after' score
make demo         # clean one messy record
```

`make help` lists everything; every variable is overridable
(`make train ITERS=1500`, `make data N=2000`).

## VRAM notes (4 GB card)

- Training defaults (batch 4, seq 1024, fp16 LoRA on a 0.6B model) fit in
  4 GB. If you hit `CUDA out of memory`:
  `make train BATCH=2` — or add `--grad-checkpoint` by running
  `python3 train/train_lora.py --batch-size 2 --grad-checkpoint` directly.
- Don't train while a llama-server is running; it holds VRAM.
- The GTX 1650 is Turing: fp16 only, no bf16. The scripts already assume this.

## Background

The concepts (why synthetic data, why an SLM, how the eval works) are
unchanged from the original project: see [docs/concepts.md](docs/concepts.md)
and [train/README.md](train/README.md) for the training specifics.
