# nmt-beam-bench

Benchmarks for proposed changes to QVAC's translation engine (tetherto/qvac, packages/translation-nmtcpp, IndicTrans2 en->indic dist-200M q4_0).

Variants (sources under `src/`):

- `base`: origin/main at fbadcc26, unmodified
- `batch`: batched beam search only (weight buffers chosen as on main)
- `final`: batched beam search (one decode per step, shared KV cells by sequence id), weights chosen per matrix so CPU_REPACK is used, tied output projection copied only when it gets a faster buffer
- `vocab`: `final` plus a Hindi output vocabulary (`data/vocab_hin_K3.txt`, target pieces seen >= 3 times in Hindi Wikipedia, plus punctuation, digits, Latin and specials)
- `cnt-base`, `cnt-final`: `base` and `final` with counters for decoder runs, KV bytes copied and KV bytes allocated

`run.sh` builds each variant against ggml from tetherto/qvac-fabric-llm.cpp, then reports work counters, quality on FLORES-200 devtest eng_Latn->hin_Deva (1012 sentences, BLEU and chrF++ with sacrebleu), and CPU time in interleaved rounds. The workflow runs it on GitHub's Linux x86, Linux ARM64, macOS and Windows runners.

FLORES-200 data: CC BY-SA 4.0. Model: IndicTrans2 (AI4Bharat), MIT.

## x86 repack `vzeroupper` bug (qvac-fabric-llm.cpp)

On Intel Xeon, `final` was first about as slow as `base` at 4 beams. The cause is in ggml: `gemv_q4_b32_8x8_q8_0_lut_avx` and `gemm_q4_b32_8x8_q8_0_lut_avx` (`ggml/src/ggml-cpu/arch/x86/repack.cpp`) take the LUT as a `__m256i` argument, and GCC emits no `vzeroupper` on their exit. The dirty upper YMM state makes later SSE code on the thread (here glibc `exp` in the beam log-softmax) much slower on Intel.

- `patches/ggml-x86-repack-vzeroupper.patch`: the fix, `_mm256_zeroupper()` after the 6 call sites (q4_0, iq4_nl, mxfp4; gemv and gemm).
- `mm/`: single mat-mul + `exp` microbenchmark and an objdump scan for functions that use ymm with no `vzeroupper`.
- `patches/fabric*-test-only.patch`, `patches/fabric*-fix-and-test.patch`: a regression test (`tests/test-repack-upper-state.cpp`) that reads XINUSE (`XGETBV` with `ECX=1`) after each repack mat-mul. Workflow `upper-state-test.yml`: with GCC 13 on Linux it fails in 5 of 6 cases before the fix on both Intel and AMD, and passes after, on fabric `temp-11739` and `temp-12032`. MSVC and clang builds were clean before the fix.

## AMX

Workflow `amx.yml` (with the fix): on Xeon 6975P the AMX buffer type is about 5% faster at 4 beams and about 15% faster at 1 beam than CPU_REPACK for the decoder matrices, with identical outputs, so `final` keeps AMX. Earlier `final` runs skipped AMX; their Intel times are slightly higher than the code now gives.
