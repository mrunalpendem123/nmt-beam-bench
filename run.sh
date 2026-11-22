#!/usr/bin/env bash
# Build every variant, then measure work counters, quality and time.
# usage: run.sh GGML_DIR MODEL OUT_DIR [N_TIME=200] [ROUNDS=3] [N_QUALITY=1012]
set -euo pipefail
GGML_DIR=$1; MODEL=$2; OUT=$3; N=${4:-200}; ROUNDS=${5:-3}; QN=${6:-1012}; CN=$(( QN < 100 ? QN : 100 ))
ROOT=$(cd "$(dirname "$0")" && pwd)
IN=$ROOT/data/eng_Latn.devtest; REF=$ROOT/data/hin_Deva.devtest
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

build() { # variant target
  cmake -S "$ROOT" -B "$ROOT/build-$1" -DGGML_DIR="$GGML_DIR" -DNMT_SRC="$ROOT/src/$1" >/dev/null
  cmake --build "$ROOT/build-$1" --target "$2" -j "$JOBS" >/dev/null
  cp "$ROOT/build-$1/$2" "$ROOT/bin-$1"
}
for v in base final vocab; do build $v nmtbench2; done
for v in cnt-base cnt-final; do build $v cntbench; done

{
  echo "# Machine"
  uname -a
  (lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)|Architecture") || sysctl -n machdep.cpu.brand_string hw.ncpu 2>/dev/null || true
  echo
} > "$OUT/summary.md"

echo "## Work counters ($CN sentences, CPU)" >> "$OUT/summary.md"
for B in 2 4 8; do
  for v in cnt-base cnt-final; do
    echo "beam=$B $v: $("$ROOT/bin-$v" "$MODEL" "$IN" "$OUT/cnt-$v-b$B.txt" 0 $B $CN 2>&1 | grep COUNTERS)" >> "$OUT/summary.md"
  done
done

echo -e "\n## Quality ($QN FLORES devtest sentences, beam 4)" >> "$OUT/summary.md"
"$ROOT/bin-base"  "$MODEL" "$IN" "$OUT/q-base.txt"  0 4 $QN 2>&1 | grep RESULT >> "$OUT/summary.md"
"$ROOT/bin-final" "$MODEL" "$IN" "$OUT/q-final.txt" 0 4 $QN 2>&1 | grep RESULT >> "$OUT/summary.md"
NMT_VOCAB=$ROOT/data/vocab_hin_K3.txt "$ROOT/bin-vocab" "$MODEL" "$IN" "$OUT/q-vocab.txt" 0 4 $QN 2>&1 | grep RESULT >> "$OUT/summary.md"
python3 "$ROOT/score.py" "$REF" "$OUT/q-base.txt" "$OUT/q-final.txt" | grep -v "first diff\|A:\|B:" >> "$OUT/summary.md"
python3 "$ROOT/score.py" "$REF" "$OUT/q-final.txt" "$OUT/q-vocab.txt" | grep -v "first diff\|A:\|B:" >> "$OUT/summary.md"

echo -e "\n## Time ($N sentences per run, $ROUNDS interleaved rounds, CPU, default threads)" >> "$OUT/summary.md"
for r in $(seq 1 "$ROUNDS"); do
  for B in 4 1; do
    echo "r$r beam=$B base:  $("$ROOT/bin-base"  "$MODEL" "$IN" "$OUT/t.txt" 0 $B $N 2>&1 | grep RESULT)" >> "$OUT/summary.md"
    echo "r$r beam=$B final: $("$ROOT/bin-final" "$MODEL" "$IN" "$OUT/t.txt" 0 $B $N 2>&1 | grep RESULT)" >> "$OUT/summary.md"
    if [ "$B" = 4 ]; then
      echo "r$r beam=$B vocab: $(NMT_VOCAB=$ROOT/data/vocab_hin_K3.txt "$ROOT/bin-vocab" "$MODEL" "$IN" "$OUT/t.txt" 0 $B $N 2>&1 | grep RESULT)" >> "$OUT/summary.md"
    fi
  done
done
cat "$OUT/summary.md"
