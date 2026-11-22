#!/usr/bin/env bash
# Compare final (AMX buffer type skipped) with final-amx (AMX kept), interleaved.
# usage: amx.sh GGML_DIR MODEL OUT_DIR [N=200] [ROUNDS=3]
set -euo pipefail
GGML_DIR=$1; MODEL=$2; OUT=$3; N=${4:-200}; ROUNDS=${5:-3}
ROOT=$(cd "$(dirname "$0")" && pwd)
IN=$ROOT/data/eng_Latn.devtest
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
build() {
  cmake -S "$ROOT" -B "$ROOT/build-$1" -DGGML_DIR="$GGML_DIR" -DNMT_SRC="$ROOT/src/$1" >/dev/null
  cmake --build "$ROOT/build-$1" --target nmtbench2 -j "$JOBS" > "$ROOT/build-$1.log" 2>&1 || { grep -iE "error" "$ROOT/build-$1.log" | head -40; exit 1; }
  cp "$ROOT/build-$1/nmtbench2" "$ROOT/bin-$1"
}
for v in final final-amx; do build $v; done
S=$OUT/summary.md
lscpu | grep -E 'Model name|Flags' | sed 's/Flags:.*\(amx[a-z_0-9 ]*\).*/AMX flags: \1/' | grep -E 'Model name|AMX' >> "$S" || true
for v in final final-amx; do echo "$v: $(BUFINFO=1 "$ROOT/bin-$v" "$MODEL" "$IN" "$OUT/b.txt" 0 1 1 2>&1 | grep BUFINFO | tr '\n' ' ')" >> "$S"; done
for v in final final-amx; do "$ROOT/bin-$v" "$MODEL" "$IN" "$OUT/q-$v.txt" 0 4 100 > /dev/null 2>&1; done
echo "identical 4-beam outputs on 100 sentences: $(paste -d'\t' "$OUT/q-final.txt" "$OUT/q-final-amx.txt" | awk -F'\t' '$1==$2' | wc -l)/$(wc -l < "$OUT/q-final.txt")" >> "$S"
for r in $(seq 1 "$ROUNDS"); do for B in 4 1; do for v in final final-amx; do
  echo "r$r beam=$B $v: $("$ROOT/bin-$v" "$MODEL" "$IN" "$OUT/t.txt" 0 $B $N 2>&1 | grep RESULT)" >> "$S"
done; done; done
cat "$S"
