# Findings

Measured KleidiAI kernel dispatch and inference throughput on Google Axion
(Arm Neoverse-V2), llama.cpp, single-stream.

---

## 1. Environment

| Item | Value |
|---|---|
| Instance | GCP `c4a-standard-8` (Google Axion) |
| CPU | Arm Neoverse-V2, 8 cores, 1 socket, 1 thread/core, stepping r0p1 |
| Cache | L1d 512 KiB, L1i 512 KiB, L2 16 MiB, L3 80 MiB |
| NUMA | 1 node, CPUs 0-7 |
| RAM | 32 GB |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-1008-gcp, aarch64 |
| Compiler | gcc 15.2.0 |
| CMake | 4.2.3 |
| perf | 7.0.12 |
| llama.cpp commit | `9558fa44c92746a58dd07ad1bf0c889715b938a6` (build 10396, ggml 0.19.0) |

Relevant `lscpu` flags present: `asimddp sve sve2 svei8mm svebf16 i8mm bf16`.
Notably **absent**: SME (confirmed by `HAVE_SME - Failed` at configure time).

---

## 2. The two builds

Both builds are configured from the same working tree at the same pinned
commit. The **only** varied option is `GGML_CPU_KLEIDIAI`.

```bash
# build-base — the standard build line
cmake -B build-base -DCMAKE_BUILD_TYPE=Release
cmake --build build-base -j8

# build-kai — KleidiAI explicitly enabled
cmake -B build-kai -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-kai -j8
```

### 2.1 Configure-time CPU feature detection is identical

Both builds independently auto-detected the host architecture and emitted the
same compile flags:

```
-- ARM detected flags: -mcpu=neoverse-v2+crc+sve2-aes+sve2-sha3+sve2-sm4+nossbs
-- GGML_MACHINE_SUPPORTS_dotprod - Success
-- GGML_MACHINE_SUPPORTS_i8mm    - Success
-- GGML_MACHINE_SUPPORTS_sve     - Success
-- GGML_MACHINE_SUPPORTS_sme     - Failed
-- HAVE_DOTPROD       - Success
-- HAVE_SVE           - Success
-- HAVE_MATMUL_INT8   - Success
-- HAVE_SME           - Failed
-- Adding CPU backend variant ggml-cpu:
     -U__ARM_FEATURE_SME;
     -mcpu=neoverse-v2+crc+sve2-aes+sve2-sha3+sve2-sm4+nossbs+dotprod+i8mm+sve+nosme
```

`build-kai` additionally emits one line:

```
-- Using KleidiAI optimized kernels if applicable
```

This is a **configure-time claim**, not a runtime guarantee. The qualifier
"if applicable" is not resolved until dispatch.

### 2.2 What each build actually contains

```bash
nm -D --defined-only build-base/bin/libggml-cpu.so | grep -c kai_
find build-base -name '*.a' | xargs -r nm | grep -c kai_
```

| Build | `kai_*` in `libggml-cpu.so` | `kai_*` in static archives |
|---|---|---|
| `build-base` | 0 | 0 |
| `build-kai` | 149 | 0 |

Static archives were checked separately to rule out a false negative from
static linking. `build-base` contains no KleidiAI code by either measure.

### 2.3 Runtime reports nothing about the active kernel path

The two builds are **indistinguishable at runtime**. `build-kai`, containing
149 KleidiAI symbols, reports:

```
| model         |       size |     params | backend | threads | test |            t/s |
| qwen2 1B Q4_0 | 403.20 MiB |   630.17 M | CPU     |       8 |  pp8 | 1161.53 ± 0.00 |
| qwen2 1B Q4_0 | 403.20 MiB |   630.17 M | CPU     |       8 |  tg8 |  227.38 ± 0.00 |
build: 9558fa44c (10396)
```

The `backend` column reads `CPU`. There is no banner line, no backend
qualifier, and no log message identifying whether KleidiAI kernels are
compiled in or dispatched. A user cannot determine from runtime output which
kernel path their binary is using.

**This is the gap this project addresses.** The dispatch decision is invisible
at exactly the point where a user would want to verify it.

---

## 3. Important framing: this is not "accelerated vs unaccelerated"

`build-base` is **not** an unoptimized baseline. At this commit it compiles
ggml's own hand-written Arm kernels:

- `ggml/src/ggml-cpu/arch/arm/quants.c`
- `ggml/src/ggml-cpu/arch/arm/repack.cpp`

built with `i8mm`, `dotprod`, and `sve` enabled, and with `HAVE_MATMUL_INT8`
passing. Both arms of this comparison use Arm int8 matrix-multiply
instructions.

**The comparison in this repository is KleidiAI's Q4_0 kernels versus ggml's
native Arm Q4_0 kernels — not acceleration versus no acceleration.**

---

## 4. Negative result: issue #26630 does not reproduce here

Upstream issue #26630 reports that the documented KleidiAI build line compiles
zero `kai_run_matmul` kernels, with the suggested workaround being
`-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"`.
That report was filed on Cortex-X925 with gcc 13.3.

**This does not reproduce on Neoverse-V2 at commit `9558fa44c` with gcc 15.2.0.**
CMake auto-detects the correct `-mcpu` string, including `dotprod`, `i8mm`, and
`sve`, with no manual architecture flags. Passing an explicit
`GGML_CPU_ARM_ARCH` therefore produces no differentiation on this platform, and
`build-kai` compiles 149 KleidiAI symbols without any workaround.

This was the project's original hypothesis. It was tested first, found not to
apply here, and the actual differentiating variable — KleidiAI enablement —
was isolated instead.

---

## 5. Benchmark methodology

```bash
M=~/models/<model>.gguf
# discard one warm-up run to warm the page cache
./build-base/bin/llama-bench -m $M -p 512 -n 128 -t 8 -r 1 >/dev/null 2>&1

for i in 1 2; do
  for b in build-base build-kai; do
    LD_LIBRARY_PATH=$PWD/$b/bin taskset -c 0-7 ./$b/bin/llama-bench \
      -m $M -p 512 -n 128 -t 8 -r 3 -o csv | sed "s/^/$b,/" >> out.csv
  done
done
```

Controls applied:

- Identical model file, prompt length (512), and generation length (128)
  across every configuration.
- `taskset -c 0-7` pinning; mask recorded.
- Page cache warmed by a discarded run before timing.
- Builds **interleaved** rather than batched, so drift under shared tenancy
  cannot be mistaken for a build effect.
- `LD_LIBRARY_PATH` scoped per-invocation, so `build-base` cannot load
  KleidiAI shared libraries from `build-kai`.
- All output written directly to CSV. No number in this document was
  transcribed by hand.

### Models

Both models are Apache-2.0 licensed.

| Model | Quant | File | Bytes | SHA256 |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | Q4_0 | `qwen2.5-0.5b-q4_0.gguf` | 428730208 | `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed` |
| Qwen2.5-7B-Instruct | Q4_0 | `qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf` | 3983228352 | `09cda9ec3aff8b5d3a760ddd16d26351e2dbf01f5c4ad4d05e18ac666708a6a2` |
| Qwen2.5-7B-Instruct | Q4_0 | `qwen2.5-7b-instruct-q4_0-00002-of-00002.gguf` | 448162496 | `db92b5c30da42fbfa33af1c501f4be6fcba1c85348d879d794b475154c0fe564` |

The 7B model ships as a two-part split GGUF; `llama-bench` is pointed at part
00001 and resolves part 00002 automatically.

---

## 6. Results

> **Status: PRELIMINARY — n=2 per cell, `-r 3` internal repeats, `t=8`.**
> To be superseded by the n=4 sweep with a thread-count axis.
> Raw data: `results/bench-0.5b-n2.csv`, `results/bench-7b-n2.csv`.

### Qwen2.5-0.5B-Instruct Q4_0

| Build | pp512 (tok/s) | tg128 (tok/s) |
|---|---|---|
| `build-base` | 715.822884 / 715.963729 | 270.026352 / 259.343281 |
| `build-kai` | 722.767914 / 722.782527 | 223.631083 / 220.219991 |

### Qwen2.5-7B-Instruct Q4_0

| Build | pp512 (tok/s) | tg128 (tok/s) |
|---|---|---|
| `build-base` | 96.485658 / 96.475970 | 27.039917 / 27.078776 |
| `build-kai` | 98.611238 / 98.619846 | 25.601570 / 25.457576 |

### Observation

Across both model scales, enabling KleidiAI **improves prefill throughput and
reduces decode throughput** relative to ggml's native Arm kernels. The sign of
the effect is consistent at both scales; the magnitude differs by model size.

Run-to-run repeatability is high — e.g. `build-kai` 7B prefill measured
98.611238 and 98.619846 tok/s across separate process invocations — which
indicates a deterministic code-path difference rather than measurement noise.

Per-cell medians, standard deviations, and percentage deltas are deferred to
the n=4 sweep rather than computed from n=2.

---

## 7. Open items

- [ ] n=4 sweep with `-t 4` / `-t 8` axis; replace Section 6.
- [ ] Third evidence layer: `perf record` / `perf report` to confirm which
      kernels actually execute, not merely which are present in the binary.
      (`perf` 7.0.12 is installed and available.)
- [ ] Cost analysis: $/1M tokens at GCP `c4a-standard-8` on-demand pricing,
      labelled single-stream batch=1.

---

## 8. Limitations

- **Single-stream, batch=1.** All measurements come from `llama-bench`. Under
  concurrent serving (`llama-server`) the prefill/decode mix shifts and the
  trade-off may differ. No claim is made about serving throughput.
- **One platform.** Neoverse-V2 via Google Axion only. Results are not assumed
  to transfer to Graviton, Ampere, or Cortex cores.
- **One commit.** `9558fa44c`. KleidiAI integration was under active upstream
  development; behaviour at other commits may differ.
- **Shared tenancy.** Run-to-run variance was low (see raw CSVs), but the
  instance is not bare metal.
- **No SME.** Neoverse-V2 does not implement SME, so KleidiAI falls back to its
  i8mm/dotprod kernel paths. SME-capable hardware is untested here.
- **One quantization.** Q4_0 only. KleidiAI's kernel coverage varies by quant
  type; Q4_K_M and others are untested here.

---

## 9. Build environment notes

Two observations affecting reproducibility on Arm build hosts, recorded for
completeness:

1. **The build fetches web assets at compile time.** The `llama-ui-assets`
   target downloads a UI bundle from Hugging Face during `cmake --build`. When
   the version-pinned URL returned an HTTP error, the build fell back to
   `latest`; when the fallback also failed, the build aborted. An offline or
   firewalled build host will fail here for reasons unrelated to the source.
   Building only the needed target (`--target llama-bench`) avoids this.

2. **Renaming a build directory breaks the resulting binaries.** CMake bakes
   the absolute build path into each binary's RPATH, so a renamed build
   directory produces `error while loading shared libraries:
   libllama-bench-impl.so`. Re-configure in place rather than renaming.
