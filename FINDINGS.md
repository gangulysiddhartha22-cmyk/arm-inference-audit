# Findings

KleidiAI kernel dispatch and throughput on Arm Neoverse-V2 (Axion `c4a-standard-8`), llama.cpp `9558fa44c`, single-stream.

## Environment

- CPU: Neoverse-V2, 8 cores  
- Flags: `sve2`, `i8mm`, `bf16`, `asimddp`  
- No SME  

## Builds

| Build | Flag | `kai_run_matmul` symbols |
|-------|------|--------------------------|
| build-base | default | **0** |
| build-kai | `-DGGML_CPU_KLEIDIAI=ON` | **10** (149 `kai_*` total) |

Runtime output is identical (`backend: CPU`). Dispatch is invisible without inspecting the binary.

**Framing:** This compares KleidiAI Q4_0 kernels vs ggml native Arm kernels — not accelerated vs unaccelerated. Both use `i8mm` / `dotprod` / `sve`.

**Note:** Upstream #26630 (zero kernels on documented build line) does **not** reproduce on this platform.

## Method

- Models: Qwen2.5-0.5B / 7B Q4_0  
- `-p 512 -n 128`, threads 4 & 8, `-r 3`, 4 outer loops  
- `taskset -c 0-7`, warmup run, CSV only  

## Results (7B, 8 threads, medians)

| Config | Prefill t/s | Decode t/s |
|--------|-------------|------------|
| base + f16 KV | 96.1 | 25.6 |
| kai + f16 KV | 98.2 | 24.6 |
| base + q8_0 KV | **123.8** | 25.3 |
| kai + q8_0 KV | **127.4** | 24.3 |

- KleidiAI: slight prefill gain, slight decode loss  
- KV `q8_0`/`q4_0`: **~+28–30% prefill**, almost no decode penalty  

## Cost (same instance)

KV quantization cuts prefill $/M tokens by ~24%. KleidiAI alone is a smaller effect.

## Verifier

`scripts/audit.sh`:

1. Claims (runtime banner)  
2. Contains (`nm` symbol counts)  

## Limits

Single-stream, one CPU, one commit, no SME, shared tenancy.
