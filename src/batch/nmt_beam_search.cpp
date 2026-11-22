// NOLINTBEGIN
#include "nmt_beam_search.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <tuple>
#include <vector>

#include "nmt.hpp"
#include "nmt_graph_decoder.hpp"
#include "nmt_state_backend.hpp"
#include "inference-addon-cpp/Logger.hpp"

struct beam_candidate {
  std::vector<nmt_vocab::id> tokens;
  float score;            // normalized score (for comparison only)
  float cumulative_score; // sum of log probabilities (unnormalized)
  bool finished;
  // Sequence id of this beam in the shared self-attention cache, or -1 for a
  // finished beam, which needs no cache.
  int seq_id;

  beam_candidate();
  beam_candidate(const std::vector<nmt_vocab::id>& tokens_, float score_);
};

beam_candidate::beam_candidate()
    : score(std::numeric_limits<float>::lowest()),
      cumulative_score(std::numeric_limits<float>::lowest()), finished(false),
      seq_id(-1) {}

beam_candidate::beam_candidate(
    const std::vector<nmt_vocab::id>& tokens_, float score_)
    : tokens(tokens_), score(score_), cumulative_score(score_), finished(false),
      seq_id(-1) {}

void nmt_set_beam_size(struct nmt_context* ctx, int beam_size) {
  if (!ctx) {
    return;
  }
  ctx->model.config.beam_size = beam_size;
}

int nmtDecodeBeamSearch(
    struct nmt_context* ctx, int beamSize,
    int maxTokens) { // NOLINT(readability-identifier-naming)
  const int vocab_size = ctx->model.hparams.n_tgt_vocab;
  nmt_state& state = *ctx->state;

  // All beams share one self-attention cache, on the backend the decoder runs
  // on. Each cell records which beams' sequences contain it, so a beam that
  // continues another shares its prefix instead of copying it, and every live
  // beam is decoded in a single batch per step. A beam needs at most one
  // decoder context of cells, and live beams never exceed beamSize.
  const int n_ctx_pad = GGML_PAD(ctx->model.hparams.n_decoder_ctx, 256);
  nmt_kv_cache beam_kv;
  if (!nmtKvCacheInit(
          beam_kv,
          state.backends[0],
          ctx->itype,
          ctx->model.hparams.n_text_state,
          ctx->model.hparams.n_decoder_layers,
          beamSize * n_ctx_pad)) {
    QLOG(
        qvac_lib_inference_addon_cpp::logger::Priority::ERROR,
        "Failed to initialize the beam search KV cache");
    return -1;
  }
  nmtKvCacheClear(beam_kv);
  std::swap(state.kv_self, beam_kv);

  std::vector<beam_candidate> beams;
  beam_candidate initial_beam;
  initial_beam.tokens = {ctx->vocab.bos_token_id};
  initial_beam.score = 0.0f;
  initial_beam.cumulative_score = 0.0f;
  initial_beam.seq_id = 0;
  beams.push_back(initial_beam);

  // Keep 2*beam_size candidates at each step, following HuggingFace's beam
  // search
  const size_t num_candidates_to_keep = 2 * beamSize;

  nmt_batch& batch = state.batch;
  std::vector<float> all_logits;
  std::vector<int> batch_row(beamSize * 2, -1);

  for (int step = 0; step < maxTokens - 1; ++step) {
    // One batch row per live beam. Live beams all have the same length.
    batch.n_tokens = 0;
    for (int beam_idx = 0; beam_idx < static_cast<int>(beams.size());
         ++beam_idx) {
      batch_row[beam_idx] = -1;
      const auto& beam = beams[beam_idx];
      if (beam.finished) {
        continue;
      }
      const int row = batch.n_tokens++;
      batch_row[beam_idx] = row;
      batch.token[row] = beam.tokens.back();
      batch.pos[row] = static_cast<nmt_pos>(beam.tokens.size()) - 1;
      batch.n_seq_id[row] = 1;
      batch.seq_id[row][0] = beam.seq_id;
      batch.logits[row] = 1;
    }

    if (batch.n_tokens == 0) {
      break;
    }

    if (!nmtDecodeGraph(*ctx, batch, state)) {
      QLOG(
          qvac_lib_inference_addon_cpp::logger::Priority::ERROR,
          "Beam search decode step failed; keeping the beams so far");
      break;
    }

    all_logits.resize(static_cast<size_t>(batch.n_tokens) * vocab_size);
    ggml_backend_tensor_get(
        state.logits_tensor,
        all_logits.data(),
        0,
        sizeof(float) * all_logits.size());

    // Store: normalized_score (for comparison), cumulative_score, beam_idx,
    // token_id
    std::priority_queue<
        std::tuple<float, float, int, nmt_vocab::id>,
        std::vector<std::tuple<float, float, int, nmt_vocab::id>>,
        std::greater<>>
        top_candidates;

    for (int beam_idx = 0; beam_idx < static_cast<int>(beams.size());
         ++beam_idx) {
      const auto& beam = beams[beam_idx];
      if (beam.finished) {
        top_candidates.emplace(beam.score, beam.cumulative_score, beam_idx, -1);
        if (top_candidates.size() > num_candidates_to_keep) {
          top_candidates.pop();
        }
        continue;
      }

      const auto row_begin = all_logits.begin() +
          static_cast<std::ptrdiff_t>(batch_row[beam_idx]) * vocab_size;
      std::vector<float> next_token_logits_vec(row_begin, row_begin + vocab_size);

      // Apply penalties and filters
      if (ctx->model.config.repetition_penalty > 0.0) {
        applyRepetitionPenalty(
            next_token_logits_vec,
            beam.tokens,
            ctx->model.config.repetition_penalty);
      }

      if (ctx->model.config.top_k > 0) {
        applyTopKFilter(
            next_token_logits_vec,
            ctx->state->decoders[0].logits_id,
            ctx->model.config.top_k);
      }

      std::for_each(
          ctx->vocab.bad_word_ids.begin(),
          ctx->vocab.bad_word_ids.end(),
          [&](const int32_t bad_word_index) {
            next_token_logits_vec[bad_word_index] = -INFINITY;
          });

      applyNoRepeatNgramFilter(
          next_token_logits_vec,
          beam.tokens,
          ctx->model.config.no_repeat_ngram_size);

      std::vector<float> logprobs(vocab_size);
      nmtComputeLogprobs(next_token_logits_vec, vocab_size, logprobs);

      // Maintain a fixed-size min-heap of this beam's best continuations
      std::priority_queue<
          std::pair<float, int>, // (score, token_id)
          std::vector<std::pair<float, int>>,
          std::greater<std::pair<float, int>> // Min-heap: smallest score on top
          >
          beam_top_k;

      const size_t new_size = beam.tokens.size() + 1;
      const float length_norm =
          (ctx->model.config.length_penalty > 0.0f)
              ? powf(
                    static_cast<float>(new_size),
                    static_cast<float>(ctx->model.config.length_penalty))
              : 1.0f;

      for (int token_id = 0; token_id < vocab_size; ++token_id) {
        if (logprobs[token_id] > -INFINITY) {
          float cumulative_score = beam.cumulative_score + logprobs[token_id];
          float normalized_score = cumulative_score / length_norm;

          if (beam_top_k.size() < num_candidates_to_keep) {
            beam_top_k.emplace(normalized_score, token_id);
          } else if (normalized_score > beam_top_k.top().first) {
            beam_top_k.pop(); // Remove smallest
            beam_top_k.emplace(normalized_score, token_id);
          }
        }
      }

      while (!beam_top_k.empty()) {
        float normalized_score = beam_top_k.top().first;
        int token_id = beam_top_k.top().second;
        beam_top_k.pop();

        float cumulative_score = beam.cumulative_score + logprobs[token_id];
        top_candidates.emplace(
            normalized_score, cumulative_score, beam_idx, token_id);
        if (top_candidates.size() > num_candidates_to_keep) {
          top_candidates.pop();
        }
      }
    }

    std::vector<std::tuple<float, float, int, nmt_vocab::id>> all_candidates;
    while (!top_candidates.empty()) {
      all_candidates.push_back(top_candidates.top());
      top_candidates.pop();
    }
    std::reverse(all_candidates.begin(), all_candidates.end());

    std::vector<beam_candidate> new_beams;
    int selected = 0;

    for (const auto& candidate : all_candidates) {
      if (selected >= beamSize)
        break;

      float normalized_score = std::get<0>(candidate);
      float cumulative_score = std::get<1>(candidate);
      int beam_idx = std::get<2>(candidate);
      nmt_vocab::id token_id = std::get<3>(candidate);

      if (token_id == -1) {
        new_beams.push_back(beams[beam_idx]);
      } else {
        beam_candidate new_beam;
        new_beam.tokens = beams[beam_idx].tokens;
        new_beam.tokens.push_back(token_id);
        new_beam.score = normalized_score;
        new_beam.cumulative_score = cumulative_score;

        bool is_eos = (token_id == 2);
        new_beam.finished = is_eos;
        // For now, the parent's sequence; renumbered below.
        new_beam.seq_id = is_eos ? -1 : beams[beam_idx].seq_id;

        new_beams.push_back(new_beam);
      }
      selected++;
    }

    // Give each live beam a sequence id in [0, beamSize) that holds its
    // parent's cells. Copy into temporary ids first, since a parent can have
    // several children and its id may be reused by another beam.
    int n_live = 0;
    for (auto& beam : new_beams) {
      if (beam.finished || beam.seq_id < 0) {
        continue;
      }
      nmtKvCacheSeqCp(state.kv_self, beam.seq_id, beamSize + n_live);
      beam.seq_id = n_live++;
    }
    for (int seq = 0; seq < beamSize; ++seq) {
      nmtKvCacheSeqRm(state.kv_self, seq);
    }
    for (int seq = 0; seq < n_live; ++seq) {
      nmtKvCacheSeqCp(state.kv_self, beamSize + seq, seq);
      nmtKvCacheSeqRm(state.kv_self, beamSize + seq);
    }

    beams = std::move(new_beams);
  }

  auto best_beam = std::max_element(
      beams.begin(),
      beams.end(),
      [](const beam_candidate& a, const beam_candidate& b) {
        return a.score < b.score;
      });

  if (best_beam != beams.end()) {
    ctx->state->decoder_inputs = best_beam->tokens;
    // Count actual output tokens (excluding BOS) for consistent metrics
    ctx->state->n_decode =
        static_cast<int32_t>(best_beam->tokens.size()) - 1; // -1 for BOS
  }

  std::swap(state.kv_self, beam_kv);
  nmtKvCacheFree(beam_kv);

  return 0;
}
// NOLINTEND
