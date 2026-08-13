# arm-inference-audit

**A tool that verifies whether your llama.cpp build on Arm is actually using the accelerated kernels it claims to be — and quantifies what the gap costs you.**

## The finding

On Arm64 (Graviton4 / Neoverse-V2), a llama.cpp build can report a normal CPU backend while containing **zero** KleidiAI matmul kernels. Adding the correct architecture flags produces a binary with 10 `kai_run_matmul` symbols (149 `kai_*` total). The difference is invisible in normal runtime output.

This project makes the invisible visible, measures the performance and cost impact, and ships a one-command verifier.

## Hardware

| Item | Value |
|------|-------|
| Instance | AWS Graviton4 (Neoverse-V2), 8 vCPU |
| Flags present | `sve2`, `i8mm`, `bf16`, `asimddp` (dotprod) |
| llama.cpp commit | `9558fa44c` |

## Builds compared

| Build | CMake flags | `kai_run_matmul` symbols |
|-------|-------------|--------------------------|
| **build-base** | Default / documented path | **0** |
| **build-kai** | `-DGGML_CPU_KLEIDIAI=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9-a+sve2+i8mm+bf16+dotprod"` | **10** |

Sample kernels present only in `build-kai`:
kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod
kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod
kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm


## Performance results (Qwen2.5-7B-Instruct Q4_0)

### Prefill (pp512) vs Decode (tg128) — 8 threads

| Configuration | Prefill t/s | Decode t/s |
|---------------|-------------|------------|
| build-base + f16 KV | 96.1 | 25.6 |
| build-kai + f16 KV | 98.2 | 24.6 |
| build-base + q8_0 KV | **123.8** | 25.3 |
| build-kai + q8_0 KV | **127.4** | 24.3 |

**Observation:** Correct KleidiAI flags give a small prefill gain. Quantizing the KV cache to `q8_0` (or `q4_0`) produces a much larger prefill improvement (~+28–30%) with almost no decode penalty.

### Cost impact (same instance)

Instance: `c8g.2xlarge` @ **$0.216/hr** (ap-south-1 on-demand).

| Configuration | Decode $/M tokens | Prefill $/M tokens |
|---------------|-------------------|--------------------|
| Naive (base + f16 KV) | $2.34 | $0.62 |
| Corrected (kai + f16 KV) | $2.44 | $0.61 |
| **Best (kai + q8_0 KV)** | $2.47 | **$0.47** |

On the same Graviton4 hardware, the combination of correct architecture flags + KV quantization reduces prefill cost by ~24%.

## How to reproduce

```bash
# 1. Clone and pin
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 9558fa44c

# 2. Build the two variants
cmake -B build-base -DGGML_CPU_KLEIDIAI=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-base -j$(nproc)

cmake -B build-kai -DGGML_CPU_KLEIDIAI=ON -DGGML_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH="armv9-a+sve2+i8mm+bf16+dotprod" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-kai -j$(nproc)

# 3. Run the verifier
./scripts/audit.sh
