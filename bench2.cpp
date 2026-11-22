// Translate a file line by line; write outputs and per-line times.
// usage: nmtbench2 MODEL IN OUT GPU BEAM [LIMIT]
// env: NT threads, REP repetition penalty, NGRAM no-repeat ngram, LP length penalty, FLASH=1
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
static double peak_rss_mb() { PROCESS_MEMORY_COUNTERS c; GetProcessMemoryInfo(GetCurrentProcess(), &c, sizeof(c)); return c.PeakWorkingSetSize / 1e6; }
#else
#include <sys/resource.h>
static double peak_rss_mb() {
  struct rusage ru; getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
  return ru.ru_maxrss / 1e6;   // bytes
#else
  return ru.ru_maxrss / 1e3;   // kilobytes
#endif
}
#endif
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "nmt.hpp"
#include "nmt_utils.hpp"
int main(int argc, char** argv) {
  if (getenv("NT")) nmtSetThreadCountForTesting(atoi(getenv("NT")));
  nmt_context_params p = nmt_context_default_params();
  p.use_gpu = atoi(argv[4]);
  if (getenv("FLASH")) p.flash_attn = atoi(getenv("FLASH"));
  nmt_context* ctx = nmtInitFromFileWithParams(argv[1], p);
  if (!ctx) { fprintf(stderr, "load failed\n"); return 1; }
  ctx->setBeamSize(atoi(argv[5]));
  ctx->setMaxLength(512);
  if (getenv("REP")) ctx->setRepetitionPenalty(atof(getenv("REP")));
  if (getenv("NGRAM")) ctx->setNoRepeatNgramSize(atoi(getenv("NGRAM")));
  if (getenv("LP")) ctx->setLengthPenalty(atof(getenv("LP")));
  int limit = argc > 6 ? atoi(argv[6]) : 1 << 30;
  std::ifstream in(argv[2]); std::ofstream out(argv[3]); std::ofstream tms(std::string(argv[3]) + ".ms");
  std::string line; int n = 0; double tot = 0;
  auto run = [&](const std::string& s) { nmtResetState(ctx); nmt_full(ctx, ("eng_Latn hin_Deva " + s).c_str()); return std::string(nmt_get_output(ctx)); };
  run("Warm up.");
  while (n < limit && std::getline(in, line)) {
    auto t = std::chrono::steady_clock::now(); auto o = run(line);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    tot += ms; out << o << "\n"; tms << ms << "\n"; n++;
  }
  fprintf(stderr, "RESULT n=%d avg %.1f ms/sentence total %.1f s peakRSS %.0f MB\n", n, tot / n, tot / 1000, peak_rss_mb());
  nmt_free(ctx);
}
