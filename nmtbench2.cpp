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
#include <map>
#include <algorithm>
#include "nmt.hpp"
#include "nmt_utils.hpp"
// PROF=1: time every graph node (computed one at a time) and sum by op / weight buffer
static std::map<std::string, std::pair<double,long>> g_prof;
static std::chrono::steady_clock::time_point g_last;
static bool prof_cb(ggml_tensor* t, bool ask, void*) {
  if (ask) { g_last = std::chrono::steady_clock::now(); return true; }
  double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - g_last).count();
  std::string k = ggml_op_desc(t);
  if (t->src[0] && t->src[0]->buffer && (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_GET_ROWS)) {
    k += std::string(" w=") + ggml_backend_buffer_name(t->src[0]->buffer) + " " + std::to_string(t->src[0]->ne[0]) + "x" + std::to_string(t->src[0]->ne[1]) + " n=" + std::to_string(t->src[1]->ne[1]);
  }
  auto& e = g_prof[k]; e.first += us; e.second++;
  return true;
}
int main(int argc, char** argv) {
  if (getenv("NT")) nmtSetThreadCountForTesting(atoi(getenv("NT")));
  nmt_context_params p = nmt_context_default_params();
  p.use_gpu = atoi(argv[4]);
  if (getenv("FLASH")) p.flash_attn = atoi(getenv("FLASH"));
  nmt_context* ctx = nmtInitFromFileWithParams(argv[1], p);
  if (!ctx) { fprintf(stderr, "load failed\n"); return 1; }
  if (getenv("BUFINFO")) {
    auto bn = [](ggml_tensor* t) { return t && t->buffer ? ggml_backend_buffer_name(t->buffer) : "none"; };
    fprintf(stderr, "BUFINFO dec0.attn_q=%s dec0.mlp_0=%s dec_embeddings=%s lm_head=%s\n", bn(ctx->model.layers_decoder[0].attn_q_w), bn(ctx->model.layers_decoder[0].mlp_0_w), bn(ctx->model.m_decoder_embeddings), bn(ctx->model.m_lm_head_w));
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) { auto* d = ggml_backend_dev_get(i); fprintf(stderr, "BUFINFO device %s: %s\n", ggml_backend_dev_name(d), ggml_backend_dev_description(d)); }
  }
  ctx->setBeamSize(atoi(argv[5]));
  ctx->setMaxLength(512);
  if (getenv("REP")) ctx->setRepetitionPenalty(atof(getenv("REP")));
  if (getenv("NGRAM")) ctx->setNoRepeatNgramSize(atoi(getenv("NGRAM")));
  if (getenv("LP")) ctx->setLengthPenalty(atof(getenv("LP")));
  int limit = argc > 6 ? atoi(argv[6]) : 1 << 30;
  std::ifstream in(argv[2]); std::ofstream out(argv[3]); std::ofstream tms(std::string(argv[3]) + ".ms");
  std::string line; int n = 0; double tot = 0;
  bool prof = getenv("PROF") != nullptr;
  auto run = [&](const std::string& s) { nmtResetState(ctx);
    if (prof) for (auto* sc : {&ctx->state->sched_encode, &ctx->state->sched_cross, &ctx->state->sched_decode}) ggml_backend_sched_set_eval_callback(sc->sched, prof_cb, nullptr);
    nmt_full(ctx, ("eng_Latn hin_Deva " + s).c_str()); return std::string(nmt_get_output(ctx)); };
  run("Warm up.");
  while (n < limit && std::getline(in, line)) {
    auto t = std::chrono::steady_clock::now(); auto o = run(line);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    tot += ms; out << o << "\n"; tms << ms << "\n"; n++;
  }
  fprintf(stderr, "RESULT n=%d avg %.1f ms/sentence total %.1f s peakRSS %.0f MB\n", n, tot / n, tot / 1000, peak_rss_mb());
  if (prof) {
    std::vector<std::pair<double,std::string>> v; double all = 0;
    for (auto& [k, e] : g_prof) { v.push_back({e.first, k + " calls=" + std::to_string(e.second)}); all += e.first; }
    std::sort(v.rbegin(), v.rend());
    for (size_t i = 0; i < v.size() && i < 25; ++i) fprintf(stderr, "PROF %7.1f ms/sentence %5.1f%% %s\n", v[i].first / 1000 / n, 100 * v[i].first / all, v[i].second.c_str());
  }
  nmt_free(ctx);
}
