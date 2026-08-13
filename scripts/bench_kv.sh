#!/bin/bash
set -u

cd ~/llama.cpp

OUT=~/bench_kv.csv
rm -f "$OUT"

M=~/models/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf

# Warmup
./build-base/bin/llama-bench -m "$M" -p 512 -n 128 -t 8 -r 1 >/dev/null 2>&1

for kv in f16 q8_0 q4_0; do
  for b in build-base build-kai; do
    for t in 4 8; do
      echo "Running KV=$kv | $b | t=$t ..."

      LD_LIBRARY_PATH="$PWD/$b/bin" \
      taskset -c 0-7 \
      "./$b/bin/llama-bench" \
        -m "$M" \
        -p 512 -n 128 -t "$t" -r 3 \
        --cache-type-k $kv --cache-type-v $kv \
        -o csv \
      | grep -v build_commit \
      | sed "s/^/$kv,$b,$t,/" >> "$OUT"
    done
  done
done

echo "DONE" >> "$OUT"
