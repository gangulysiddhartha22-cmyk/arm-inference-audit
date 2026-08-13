#!/bin/bash
set -u

cd ~/llama.cpp

M=~/models/qwen2.5-0.5b-q4_0.gguf
OUT=~/bench0.5b_n4.csv

rm -f "$OUT"

# Warm the page cache (not measured)
./build-base/bin/llama-bench \
  -m "$M" -p 512 -n 128 -t 8 -r 1 \
  >/dev/null 2>&1

for i in 1 2 3 4; do
  for b in build-base build-kai; do
    for t in 4 8; do

      LD_LIBRARY_PATH="$PWD/$b/bin" \
      taskset -c 0-7 \
      "./$b/bin/llama-bench" \
        -m "$M" \
        -p 512 \
        -n 128 \
        -t "$t" \
        -r 3 \
        -o csv \
      | grep -v build_commit \
      | sed "s/^/$b,$t,/" >> "$OUT"

    done
  done
done

echo "DONE" >> "$OUT"
