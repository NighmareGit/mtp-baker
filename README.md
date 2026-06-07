# mtp-baker

**Reliable toolkit for building and repairing MTP (Multi-Token Prediction) GGUF models**, especially for the Qwen family.

This project was created because of bugs in popular quantization/export pipelines (e.g. Unsloth GGUF export truncating trailing MTP tensors). It gives you control over the process.

## Current Focus (v0.1)

- **GGUF MTP Head Grafting** — Merge MTP draft heads into any base GGUF (the fastest way to get working MTP models right now).
- **Verification** — Check that MTP tensors are present and intact.
- **Safe Quantization** — Re-quantize while protecting MTP layers at higher precision.

Future versions will add:
- HF → GGUF conversion with MTP awareness
- Full MTP head grafting + training from raw Hugging Face checkpoints

## Installation (Development)

```bash
cd mtp-baker
pip install -e "[dev]"   # or just pip install -e .
```

Or run directly with uv / pipx once published.

## Quick Start

```bash
# Graft MTP heads
mtp-baker graft \
    --base /path/to/base-Q4_K_M.gguf \
    --heads /path/to/MTP-Q8_0-heads.gguf \
    --output /path/to/output-mtp.gguf

# Verify
mtp-baker verify /path/to/output-mtp.gguf

# Safe re-quantize (protect MTP tensors)
mtp-baker quantize \
    --input output-mtp-f16.gguf \
    --output output-mtp-Q5_K_M.gguf \
    --mtp-precision q8_0
```

## Why This Exists

Unsloth's GGUF export pipeline currently truncates trailing tensors (including MTP draft heads) on Qwen3.5/3.6 models. This toolkit lets you work around that and produce reliable MTP GGUFs for `llama.cpp` (including turboquant/ik forks).

## Contributing / Roadmap

- More robust grafting (handle different tensor naming schemes)
- imatrix generation that protects MTP tensors
- Full PyTorch MTP head grafting + LoRA training
- Docker image

Pull requests and issues welcome.
