#!/usr/bin/env bash
# After the review fixes: final must give the same output as final-prev, at the same speed,
# and a beam width above the decoder context must not crash.
set -euo pipefail
GGML_DIR=$1; MODEL=$2; OUT=$3
ROOT=$(cd "$(dirname "$0")" && pwd)
IN=$ROOT/data/eng_Latn.devtest
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
for v in final-prev final; do
  cmake -S "$ROOT" -B "$ROOT/build-$v" -DGGML_DIR="$GGML_DIR" -DNMT_SRC="$ROOT/src/$v" >/dev/null
  cmake --build "$ROOT/build-$v" --target nmtbench2 -j "$JOBS" > "$ROOT/build-$v.log" 2>&1 || { grep -iE "error" "$ROOT/build-$v.log" | head -40; exit 1; }
  cp "$ROOT/build-$v/nmtbench2" "$ROOT/bin-$v"
done
S=$OUT/summary.md
(lscpu 2>/dev/null | grep 'Model name') >> "$S" || sysctl -n machdep.cpu.brand_string >> "$S"
for B in 4 1; do
  for v in final-prev final; do "$ROOT/bin-$v" "$MODEL" "$IN" "$OUT/q$B-$v.txt" 0 $B 1012 > /dev/null 2>&1; done
  echo "beam=$B identical outputs on 1012 sentences: $(paste -d'\t' "$OUT/q$B-final-prev.txt" "$OUT/q$B-final.txt" | awk -F'\t' '$1==$2' | wc -l)/$(wc -l < "$OUT/q$B-final.txt")" >> "$S"
done
for r in 1 2; do for B in 4 1; do for v in final-prev final; do
  echo "r$r beam=$B $v: $("$ROOT/bin-$v" "$MODEL" "$IN" "$OUT/t.txt" 0 $B 100 2>&1 | grep RESULT)" >> "$S"
done; done; done
set +e
"$ROOT/bin-final" "$MODEL" "$IN" "$OUT/big.txt" 0 300 2 > "$OUT/big.log" 2>&1
echo "beam=300 (above n_decoder_ctx), 2 sentences: exit $? $(grep RESULT "$OUT/big.log")" >> "$S"
cat "$S"
