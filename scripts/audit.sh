#!/bin/bash
set -u

cd "$(dirname "$0")/.."

MODEL=${MODEL:-~/models/qwen2.5-0.5b-q4_0.gguf}
BUILDS=${BUILDS:-"build-base build-kai"}

echo "=== arm-inference-audit — dispatch verifier ==="
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo

for b in $BUILDS; do
  echo "--- $b ---"
  BIN="./$b/bin/llama-bench"
  LIB="./$b/bin/libggml-cpu.so"

  [[ -x "$BIN" ]] || { echo " missing: $BIN"; echo; continue; }

  echo "[1] CLAIMS — what runtime output reports"
  LD_LIBRARY_PATH="$PWD/$b/bin" timeout 60s "$BIN" \
      -m "$MODEL" -p 8 -n 8 -r 1 </dev/null 2>&1 \
    | grep -iE "kleidi|backend|^build:" | head -5
  echo " (no KleidiAI indication above = dispatch is invisible at runtime)"

  echo
  echo "[2] CONTAINS — compiled kernel symbols"
  if [[ -f "$LIB" ]]; then
    ALL=$(nm -D --defined-only "$LIB" 2>/dev/null | grep -c "kai_" || true)
    MM=$(nm -D --defined-only "$LIB" 2>/dev/null | grep -c "kai_run_matmul" || true)
    echo " libggml-cpu.so: kai_* = $ALL   kai_run_matmul* = $MM"

    if [[ "$MM" -gt 0 ]]; then
      echo " sample kai_run_matmul symbols:"
      nm -D --defined-only "$LIB" 2>/dev/null | grep "kai_run_matmul" | head -8 | sed 's/^/   /'
    fi
  else
    echo " libggml-cpu.so not found"
  fi

  STATIC=$(find "$b" -name '*.a' -print0 2>/dev/null | xargs -0 -r nm 2>/dev/null | grep -c "kai_" || true)
  echo " static archives: kai_* = $STATIC"
  echo
done

echo "=== done ==="
