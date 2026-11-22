// Time one q4_0 matrix multiply in the plain CPU layout vs the CPU_REPACK layout.
// Shapes are the IndicTrans2 decoder ones; N = number of beams in the batch.
// usage: mmbench [ITERS=200]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

static ggml_backend_buffer_type_t find_repack() {
  ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  auto reg = ggml_backend_dev_backend_reg(dev);
  auto fn = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
  if (!fn) return nullptr;
  for (auto* b = fn(dev); b && *b; ++b) if (std::string(ggml_backend_buft_name(*b)) == "CPU_REPACK") return *b;
  return nullptr;
}

// returns microseconds per multiply
static double run(ggml_backend_t be, ggml_backend_buffer_type_t wbuft, int K, int M, int N, int iters) {
  ggml_init_params ip = {ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
  ggml_context* wctx = ggml_init(ip);
  ggml_tensor* w = ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_0, K, M);
  ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, wbuft);
  ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.f, 0.05f);
  std::vector<float> f((size_t) K * M);
  for (auto& v : f) v = nd(rng);
  std::vector<uint8_t> q(ggml_nbytes(w));
  ggml_quantize_chunk(GGML_TYPE_Q4_0, f.data(), q.data(), 0, M, K, nullptr);
  ggml_backend_tensor_set(w, q.data(), 0, q.size());

  ggml_context* gctx = ggml_init(ip);
  ggml_tensor* x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, N);
  ggml_tensor* y = ggml_mul_mat(gctx, w, x);
  ggml_cgraph* g = ggml_new_graph(gctx);
  ggml_build_forward_expand(g, y);
  ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, be);
  std::vector<float> xv((size_t) K * N);
  for (auto& v : xv) v = nd(rng);
  ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);

  for (int i = 0; i < 10; ++i) ggml_backend_graph_compute(be, g);
  auto t = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, g);
  double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count() / iters;
  ggml_backend_buffer_free(gbuf); ggml_backend_buffer_free(wbuf);
  ggml_free(gctx); ggml_free(wctx);
  return us;
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 200;
  ggml_backend_t be = ggml_backend_cpu_init();
  ggml_backend_buffer_type_t plain = ggml_backend_cpu_buffer_type();
  ggml_backend_buffer_type_t rep = find_repack();
  if (!rep) { printf("no CPU_REPACK buffer type\n"); return 1; }
  struct S { const char* name; int K, M; } shapes[] = {
    {"lm_head 512->122672", 512, 122672}, {"attn 512->512", 512, 512},
    {"ffn_up 512->2048", 512, 2048}, {"ffn_down 2048->512", 2048, 512}};
  printf("| shape | N | plain us | repack us | repack/plain |\n|---|---|---|---|---|\n");
  for (auto& s : shapes) for (int N : {1, 4}) {
    int it = s.M > 100000 ? iters / 4 : iters * 4;
    double a = run(be, plain, s.K, s.M, N, it), b = run(be, rep, s.K, s.M, N, it);
    printf("| %s | %d | %.1f | %.1f | %.2f |\n", s.name, N, a, b, b / a);
  }
  ggml_backend_free(be);
}
