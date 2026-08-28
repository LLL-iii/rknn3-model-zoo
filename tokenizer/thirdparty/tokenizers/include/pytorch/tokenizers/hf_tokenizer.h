/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Used by many Huggingface models. Adapted from a combination of the original
// rust implementation (https://github.com/huggingface/tokenizers/tree/main)
// and the corresponding support in llama.cpp
// (https://github.com/ggerganov/llama.cpp)
#pragma once

// Standard
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

// Local
#include <nlohmann/json.hpp>
#include <pytorch/tokenizers/bpe_tokenizer_base.h>
#include <pytorch/tokenizers/error.h>
#include <pytorch/tokenizers/normalizer.h>
#include <pytorch/tokenizers/post_processor.h>
#include <pytorch/tokenizers/pre_tokenizer.h>
#include <pytorch/tokenizers/result.h>
#include <pytorch/tokenizers/token_decoder.h>

namespace tokenizers {
namespace detail {

// Flat, sorted merge entry: (fid, sid) -> (rank, mid).
// Stored in a contiguous std::vector and binary-searched, avoiding the
// ~32-byte-per-node overhead of std::unordered_map (which for 250K BPE
// merge rules was ~20 MB; the flat vector is ~8 MB and cache-friendly).
// Token ids are uint32_t: every supported vocabulary is far below 2^32, so
// three uint64 fields collapse to uint32 and the entry shrinks from 32 to 16
// bytes, cutting the merge map roughly in half.
struct MergeEntry {
  uint32_t fid;
  uint32_t sid;
  uint32_t mid;
  uint32_t rank;
};

// BPE merge map: (token_id_1, token_id_2) -> (rank, merged_token_id).
// Sorted by (fid, sid) for std::lower_bound lookup.
using MergeMap = std::vector<MergeEntry>;

// Lookup helper: returns pointer to matching entry or nullptr.
inline const MergeEntry* merge_lookup(
    const MergeMap& map, uint64_t fid, uint64_t sid) {
  auto it = std::lower_bound(
      map.begin(), map.end(),
      MergeEntry{static_cast<uint32_t>(fid), static_cast<uint32_t>(sid), 0, 0},
      [](const MergeEntry& a, const MergeEntry& b) {
        return a.fid < b.fid || (a.fid == b.fid && a.sid < b.sid);
      });
  if (it != map.end() && it->fid == fid && it->sid == sid) {
    return &*it;
  }
  return nullptr;
}

} // namespace detail

// Simple Word structure to mimic Rust's Word behavior
struct HFWord {
  std::vector<uint64_t> tokens;
  std::vector<size_t> byte_lengths;

  void add(uint64_t token_id, size_t byte_len) {
    tokens.push_back(token_id);
    byte_lengths.push_back(byte_len);
  }

  size_t size() const {
    return tokens.size();
  }

  // Apply all possible merges using the explicit (id, id) -> (rank, merged_id)
  // merge map. Mirrors HuggingFace's Word::merge_all: an intrusive doubly
  // linked list over the symbols plus a min-heap keyed on merge rank, giving
  // O(n log n) instead of the naive rescan-all-pairs O(n^2). Lookups are by
  // integer id pair, so there is no per-pair string allocation. Defined out of
  // line in hf_tokenizer.cpp.
  void merge_all(const detail::MergeMap& merge_map);
};

class HFTokenizer : public detail::BPETokenizerBase {
 public:
  /*-- Public Interface --*/

  /**
   * Default initialize with no loaded data
   */
  explicit HFTokenizer() {}
  ~HFTokenizer() {}

  /**
   * Load the model data into the
   */
  Error load(const std::string& tokenizer_path) override;

  Result<std::vector<uint64_t>> encode(
      const std::string& input,
      int8_t bos = 0,
      int8_t eos = 0) const override;

  using BPETokenizerBase::decode;

  Result<std::string> decode(
      const std::vector<uint64_t>& tokens,
      bool skip_special_tokens = false) const;

 private:
  Error _encode(
      const std::string& input,
      std::vector<uint64_t>& ret,
      uint64_t& last_piece_token_len) const override;

  void _decode(const std::string& input, std::string& ret) const override;

  std::vector<std::string> _decode(
      const std::vector<std::string>& pieces) const;

  Result<std::vector<uint64_t>> byte_pair_encode_(
      const std::string& piece,
      const detail::TokenMap& encoder) const override;

  // Override the virtual _byte_pair_merge method to use explicit merges
  // specified in tokenizer.json. Different from Tiktoken (another user of
  // BPETokenizerBase, but doesn't use explicit merge rules).
  std::vector<uint64_t> _byte_pair_merge(
      const std::string& piece,
      const detail::TokenMap& ranks,
      std::function<uint64_t(uint64_t, uint64_t)> func) const override;

  Error parse_special_tokens(const nlohmann::json& parsed_json);
  Error parse_tokens(const nlohmann::json& parsed_json);
  Error setup_normalizer(const nlohmann::json& parsed_json);
  Error setup_pretokenizer(const nlohmann::json& parsed_json);
  Error setup_postprocessor(const nlohmann::json& parsed_json);
  Error setup_decoder(const nlohmann::json& parsed_json);
  Error parse_merges(const nlohmann::json& parsed_json);
  Error setup_special_token_ids(
      const std::string& path,
      const nlohmann::json& parsed_json,
      const std::string& model_config_json,
      const std::string& special_tokens_map_json);

  Normalizer::Ptr _normalizer;
  PreTokenizer::Ptr _pretokenizer;
  PostProcessor::Ptr _postprocessor;
  TokenDecoder::Ptr _decoder;

  std::unique_ptr<detail::MergeMap> merge_map_;
  bool byte_fallback_ = false;
  bool unk_token_is_configured_ = false;
};

} // namespace tokenizers
