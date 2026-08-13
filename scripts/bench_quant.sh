#!/bin/bash
set -u

cd ~/llama.cpp

OUT=~/bench_quant.csv
rm -f "$OUT"

# Models
M_Q4_0=~/models/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
M_Q4KM=~/models/qwen2.5-7b-instruct-q4_k_m.gguf

# Warmup (one model is enough)
./build-base/bin/llama-bench -m "$M_Q4_0" -p 512 -n 128 -t 8 -r 1 >/dev/null 2>&1

for quant in Q4_0 Q4_K_M; do
  if [[ "$quant" == "Q4_0" ]]; then
    M="$M_Q4_0"
  else
    M="$M_Q4KM"
  fi

  for b in build-base build-kai; do
    for t in 4 8; do
      echo "Running $quant | $b | t=$t ..."

      LD_LIBRARY_PATH="$PWD/$b/bin" \
      taskset -c 0-7 \
      "./$b/bin/llama-bench" \
        -m "$M" \
        -p 512 -n 128 -t "$t" -r 3 -o csv \
      | grep -v build_commit \
      | sed "s/^/$quant,$b,$t,/" >> "$OUT"
    done
  done
done

echo "DONE" >> "$OUT"
