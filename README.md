# arm-inference-audit

**Verify whether your llama.cpp build on Arm is actually using KleidiAI kernels — and make that check reproducible.**

## What this is

On Arm Neoverse-V2 (Google Axion), a default llama.cpp build can report `backend: CPU` while containing **zero** KleidiAI matmul symbols. Enabling `GGML_CPU_KLEIDIAI=ON` compiles real KleidiAI kernels, but runtime output still does not advertise which path is active.

This repo ships a small verifier and locked benchmark scripts so the gap is measurable, not anecdotal.

## What you get

- `scripts/audit.sh` — claims vs contains check (`kai_run_matmul` symbol counts)
- Locked `llama-bench` scripts for base vs KleidiAI builds
- Raw CSVs and environment notes under `results/`
- Full write-up in [`FINDINGS.md`](FINDINGS.md)

## Quick start

```bash
# 1. Clone and pin llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 9558fa44c

# 2. Build both variants
cmake -B build-base -DCMAKE_BUILD_TYPE=Release
cmake --build build-base -j$(nproc)

cmake -B build-kai -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-kai -j$(nproc)

# 3. Run the verifier (from this repo)
./scripts/audit.sh
