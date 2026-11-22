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
  cmake --build "$ROOT/build-$1" --target "$2" -j "$JOBS" > "$ROOT/build-$1.log" 2>&1 || { grep -iE "error" "$ROOT/build-$1.log" | head -40; exit 1; }
  if [ -f "$ROOT/build-$1/$2.exe" ]; then cp "$ROOT/build-$1/$2.exe" "$ROOT/bin-$1.exe"; else cp "$ROOT/build-$1/$2" "$ROOT/bin-$1"; fi
}
for v in base batch final vocab; do build $v nmtbench2; done
for v in cnt-base cnt-final; do build $v cntbench; done
cmake -S "$ROOT/mm" -B "$ROOT/build-mm" -DGGML_DIR="$GGML_DIR" >/dev/null && cmake --build "$ROOT/build-mm" -j "$JOBS" > "$ROOT/build-mm.log" 2>&1 || true

{
  echo "# Machine"
  uname -a
  (lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)|Architecture|Thread|L2|L3") || sysctl -n machdep.cpu.brand_string hw.ncpu 2>/dev/null || true
  echo
} > "$OUT/summary.md"

echo "## Weight buffers" >> "$OUT/summary.md"
for v in base batch final; do echo "$v: $(BUFINFO=1 "$ROOT/bin-$v" "$MODEL" "$IN" "$OUT/b.txt" 0 1 1 2>&1 | grep BUFINFO | tr '\n' ' ')" >> "$OUT/summary.md"; done
echo >> "$OUT/summary.md"
if [ -x "$ROOT/build-mm/mmbench" ]; then echo "## Single matmul, plain vs CPU_REPACK" >> "$OUT/summary.md"; "$ROOT/build-mm/mmbench" 200 2>/dev/null >> "$OUT/summary.md"; echo >> "$OUT/summary.md"; fi
echo "## One thread, beam 1 and 4 ($CN sentences)" >> "$OUT/summary.md"
for B in 1 4; do for v in base batch final; do echo "NT=1 beam=$B $v: $(NT=1 "$ROOT/bin-$v" "$MODEL" "$IN" "$OUT/t1.txt" 0 $B $CN 2>&1 | grep RESULT)" >> "$OUT/summary.md"; done; done
echo >> "$OUT/summary.md"
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
"${PYTHON:-python3}" "$ROOT/score.py" "$REF" "$OUT/q-base.txt" "$OUT/q-final.txt" | grep -v "first diff\|A:\|B:" >> "$OUT/summary.md"
"${PYTHON:-python3}" "$ROOT/score.py" "$REF" "$OUT/q-final.txt" "$OUT/q-vocab.txt" | grep -v "first diff\|A:\|B:" >> "$OUT/summary.md"

echo -e "\n## Time ($N sentences per run, $ROUNDS interleaved rounds, CPU, default threads)" >> "$OUT/summary.md"
for r in $(seq 1 "$ROUNDS"); do
  for B in 4 1; do
    echo "r$r beam=$B base:  $("$ROOT/bin-base"  "$MODEL" "$IN" "$OUT/t.txt" 0 $B $N 2>&1 | grep RESULT)" >> "$OUT/summary.md"
    echo "r$r beam=$B batch: $("$ROOT/bin-batch" "$MODEL" "$IN" "$OUT/t.txt" 0 $B $N 2>&1 | grep RESULT)" >> "$OUT/summary.md"
    echo "r$r beam=$B final: $("$ROOT/bin-final" "$MODEL" "$IN" "$OUT/t.txt" 0 $B $N 2>&1 | grep RESULT)" >> "$OUT/summary.md"
    if [ "$B" = 4 ]; then
      echo "r$r beam=$B vocab: $(NMT_VOCAB=$ROOT/data/vocab_hin_K3.txt "$ROOT/bin-vocab" "$MODEL" "$IN" "$OUT/t.txt" 0 $B $N 2>&1 | grep RESULT)" >> "$OUT/summary.md"
    fi
  done
done
cat "$OUT/summary.md"
