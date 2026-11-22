# nmt-beam-bench

Benchmarks for proposed changes to QVAC's translation engine (tetherto/qvac, packages/translation-nmtcpp, IndicTrans2 en->indic dist-200M q4_0).

Variants (sources under `src/`):

- `base`: origin/main at fbadcc26, unmodified
- `final`: batched beam search (one decode per step, shared KV cells by sequence id), weights chosen per matrix so CPU_REPACK is used, tied output projection copied only when it gets a faster buffer
- `vocab`: `final` plus a Hindi output vocabulary (`data/vocab_hin_K3.txt`, target pieces seen >= 3 times in Hindi Wikipedia, plus punctuation, digits, Latin and specials)
- `cnt-base`, `cnt-final`: `base` and `final` with counters for decoder runs, KV bytes copied and KV bytes allocated

`run.sh` builds each variant against ggml from tetherto/qvac-fabric-llm.cpp, then reports work counters, quality on FLORES-200 devtest eng_Latn->hin_Deva (1012 sentences, BLEU and chrF++ with sacrebleu), and CPU time in interleaved rounds. The workflow runs it on GitHub's Linux x86, Linux ARM64, macOS and Windows runners.

FLORES-200 data: CC BY-SA 4.0. Model: IndicTrans2 (AI4Bharat), MIT.
