/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/hf_tokenizer.h>

#include <unicode.h>

// Standard
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <queue>
#include <string>
#include <string_view>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __GLIBC__
#include <malloc.h>
#endif

// Third Party
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {
bool fs_is_directory(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
bool fs_exists(const std::string& path) {
    return access(path.c_str(), F_OK) == 0;
}
std::string fs_path_join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

// ── Reliable JSON scanner for tokenizer.json ─────────────────────────
// We hand-scan the large vocab/merges sections to avoid building a full
// in-memory JSON node tree.  The key correctness requirement is that
// bracket counting must skip JSON string literals in one shot (not a
// char-by-char in_str state machine that trips on escaped quotes /
// braces inside tokens).

inline bool is_ws(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}
inline const char* skip_ws(const char* p, const char* end) {
  while (p < end && is_ws(*p)) ++p;
  return p;
}

// Skip a JSON string literal starting at p (must point at '"').
// Returns pointer past the closing quote.
static const char* skip_json_string(const char* p, const char* end) {
  if (p >= end || *p != '"') return p;
  ++p;
  while (p < end) {
    if (*p == '\\') { p += 2; continue; }   // skip escaped char
    if (*p == '"') { ++p; break; }
    ++p;
  }
  return p;
}

// Decode a JSON string literal starting at p (must point at '"').
// Appends decoded bytes to `arena` and returns a view into the newly appended
// bytes.  The caller must have reserved `arena` large enough that push_back
// never reallocates while the returned view is outstanding (otherwise it
// dangles).  Advances *out past the closing quote.
static std::string_view decode_json_string_into(std::string& arena,
                                                const char* p, const char* end,
                                                const char** out) {
  const size_t start = arena.size();
  if (p >= end || *p != '"') { *out = p; return std::string_view(); }
  ++p;
  while (p < end && *p != '"') {
    unsigned char c = (unsigned char)*p;
    if (c == '\\' && p + 1 < end) {
      ++p;
      unsigned char e = (unsigned char)*p;
      switch (e) {
        case '"': arena.push_back('"'); ++p; break;
        case '\\': arena.push_back('\\'); ++p; break;
        case '/': arena.push_back('/'); ++p; break;
        case 'b': arena.push_back('\b'); ++p; break;
        case 'f': arena.push_back('\f'); ++p; break;
        case 'n': arena.push_back('\n'); ++p; break;
        case 'r': arena.push_back('\r'); ++p; break;
        case 't': arena.push_back('\t'); ++p; break;
        case 'u': {
          unsigned cp = 0;
          if (p + 4 < end) {
            for (int i = 1; i <= 4; ++i) {
              unsigned char h = (unsigned char)p[i];
              cp <<= 4;
              if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
              else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
              else cp = 0xFFFDu;
            }
            p += 5;
          } else p = end;
          if (cp < 0x80) arena.push_back((char)cp);
          else if (cp < 0x800) {
            arena.push_back((char)(0xC0 | (cp >> 6)));
            arena.push_back((char)(0x80 | (cp & 0x3F)));
          } else {
            arena.push_back((char)(0xE0 | (cp >> 12)));
            arena.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            arena.push_back((char)(0x80 | (cp & 0x3F)));
          }
          break;
        }
        default: ++p; break;
      }
    } else {
      arena.push_back((char)c);
      ++p;
    }
  }
  if (p < end && *p == '"') ++p;
  *out = p;
  return std::string_view(arena.data() + start, arena.size() - start);
}

// Decode into a fresh std::string (used by the small merge/config sections).
static std::string decode_json_string(const char* p, const char* end,
                                      const char** out) {
  std::string s;
  decode_json_string_into(s, p, end, out);
  return s;
}

static bool decode_uint(const char* p, const char* end, uint64_t* out,
                        const char** out_end) {
  uint64_t v = 0; bool any = false;
  while (p < end && *p >= '0' && *p <= '9') {
    v = v * 10 + (uint64_t)(*p - '0'); any = true; ++p;
  }
  *out = v; *out_end = p; return any;
}

// Find the value range [start, end) of the JSON key `key` anywhere in data.
// Skips string literals whole, so braces/quotes inside token strings can
// never corrupt the bracket count.
static std::pair<const char*, const char*>
find_key_value(const char* data, size_t len, const char* key) {
  std::string pat = "\"" + std::string(key) + "\"";
  const char* p = data;
  const char* end = data + len;
  while (p + pat.size() <= end) {
    const char* hit = nullptr;
    // linear scan for `"key"`, skipping over string literals so a token
    // that happens to contain `"key"` inside its value is not matched.
    for (const char* q = p; q + pat.size() <= end; ) {
      if (*q == '"') {
        // A string literal here could be a key or a value.  If it matches
        // pat AND is followed by ':', it's our key; otherwise skip it whole.
        if (memcmp(q, pat.data(), pat.size()) == 0) {
          const char* after = skip_ws(q + pat.size(), end);
          if (after < end && *after == ':') { hit = q; break; }
        }
        q = skip_json_string(q, end);
      } else {
        ++q;
      }
    }
    if (!hit) break;
    // Must be a key: followed by optional ws + ':'
    const char* after = skip_ws(hit + pat.size(), end);
    if (after < end && *after == ':') {
      const char* vs = skip_ws(after + 1, end);
      if (vs >= end) return {nullptr, nullptr};
      char open = *vs, close = 0;
      if (open == '{') close = '}';
      else if (open == '[') close = ']';
      else if (open == '"') {
        return {vs, skip_json_string(vs, end)};
      } else {
        const char* q = vs;
        while (q < end && *q != ',' && *q != '}' && *q != ']') ++q;
        return {vs, q};
      }
      int depth = 0;
      const char* q = vs;
      while (q < end) {
        char c = *q;
        if (c == '"') { q = skip_json_string(q, end); continue; }
        if (c == open) ++depth;
        else if (c == close) { --depth; if (depth == 0) { ++q; break; } }
        ++q;
      }
      return {vs, q};
    }
    p = hit + 1;
  }
  return {nullptr, nullptr};
}
}


namespace tokenizers {

void HFWord::merge_all(const detail::MergeMap& merge_map) {
  const size_t n = tokens.size();
  if (n < 2) {
    return;
  }

  std::vector<int64_t> prev(n), next(n);
  std::vector<uint32_t> version(n, 0);
  std::vector<uint8_t> alive(n, 1);
  for (size_t i = 0; i < n; ++i) {
    prev[i] = static_cast<int64_t>(i) - 1;
    next[i] = (i + 1 < n) ? static_cast<int64_t>(i + 1) : -1;
  }

  // (rank, pos, version): lowest rank wins, ties broken by leftmost position to
  // match left-to-right BPE order. `version` tracks only the left symbol; a
  // candidate whose right neighbor changed is rejected below by recomputing the
  // pair's rank and comparing to `c.rank`. That check relies on merge ranks
  // being unique (they are merge indices), so a changed pair cannot
  // coincidentally carry the same rank.
  //
  // Candidate is kept at 12 bytes (uint32 rank/pos/version; merge indices and
  // symbol positions are both < 2^31 for every supported vocabulary/input) so
  // heap sift operations copy a third of the bytes of the old 24-byte struct.
  struct Candidate {
    uint32_t rank;
    int32_t pos;
    uint32_t version;
  };
  struct Compare {
    bool operator()(const Candidate& a, const Candidate& b) const {
      if (a.rank != b.rank) {
        return a.rank > b.rank;
      }
      return a.pos > b.pos;
    }
  };

  // Prefer std::make_heap over n individual push_heap calls: initial
  // construction is O(n) instead of O(n log n), which matters when a whole
  // 100k-char input arrives as a single word (null pre_tokenizer).
  std::vector<Candidate> heap;
  heap.reserve(n);
  for (size_t i = 0; i + 1 < n; ++i) {
    const auto* e =
        detail::merge_lookup(merge_map, tokens[i], tokens[i + 1]);
    if (e) {
      heap.push_back({e->rank, static_cast<int32_t>(i), 0});
    }
  }
  std::make_heap(heap.begin(), heap.end(), Compare());

  auto push_pair = [&](int64_t i) {
    if (i < 0 || next[i] < 0) {
      return;
    }
    const auto* e =
        detail::merge_lookup(merge_map, tokens[i], tokens[next[i]]);
    if (e) {
      heap.push_back({e->rank, static_cast<int32_t>(i), version[i]});
      std::push_heap(heap.begin(), heap.end(), Compare());
    }
  };

  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), Compare());
    const Candidate c = heap.back();
    heap.pop_back();
    const int64_t i = c.pos;
    if (!alive[i] || version[i] != c.version || next[i] < 0) {
      continue; // stale entry
    }
    const int64_t j = next[i];
    const auto* e = detail::merge_lookup(merge_map, tokens[i], tokens[j]);
    if (!e || e->rank != c.rank) {
      continue; // superseded by a different merge at this position
    }

    // Merge j into i.
    tokens[i] = e->mid;
    byte_lengths[i] += byte_lengths[j];
    alive[j] = 0;
    const int64_t jn = next[j];
    next[i] = jn;
    if (jn >= 0) {
      prev[jn] = i;
    }
    ++version[i];

    // Only the two adjacencies touching the merged symbol can change.
    push_pair(prev[i]);
    push_pair(i);
  }

  // Compact surviving symbols in list order (head is always index 0; it is only
  // ever a left operand, never removed).
  std::vector<uint64_t> merged_tokens;
  std::vector<size_t> merged_byte_lengths;
  merged_tokens.reserve(tokens.size());
  merged_byte_lengths.reserve(byte_lengths.size());
  for (int64_t i = 0; i != -1; i = next[i]) {
    merged_tokens.push_back(tokens[i]);
    merged_byte_lengths.push_back(byte_lengths[i]);
  }
  tokens = std::move(merged_tokens);
  byte_lengths = std::move(merged_byte_lengths);
}

namespace {
// Helper to extract token string from either string or object format
std::string extract_token_string(const json& token_json) {
  if (token_json.is_string()) {
    return token_json.get<std::string>();
  } else if (token_json.is_object() && token_json.contains("content")) {
    return token_json["content"].get<std::string>();
  }
  return "";
};
} // namespace

// -------------------------public method start-------------------------------

Error HFTokenizer::load(const std::string& path) {
  std::string model_json = path;
  std::string model_config_json;
  std::string special_tokens_map_json;

  if (fs_is_directory(path)) {
    model_json = fs_path_join(path, "tokenizer.json");
    if (!fs_exists(model_json)) {
      TK_LOG(Info, "no tokenizer.json found in %s", path.c_str());
      return Error::LoadFailure;
    }
    model_config_json = fs_path_join(path, "tokenizer_config.json");
    if (!fs_exists(model_config_json)) {
      model_config_json.clear();
    }
    special_tokens_map_json = fs_path_join(path, "special_tokens_map.json");
    if (!fs_exists(special_tokens_map_json)) {
      special_tokens_map_json.clear();
    }
  }

  int fd = open(model_json.c_str(), O_RDONLY);
  if (fd < 0) {
    TK_LOG(Info, "failed to open encoder file: %s", path.c_str());
    return Error::LoadFailure;
  }
  struct stat st;
  fstat(fd, &st);
  size_t data_len = st.st_size;
  const char *data = (const char *)mmap(nullptr, data_len, PROT_READ,
                                        MAP_PRIVATE, fd, 0);
  close(fd);
  if (data == MAP_FAILED) {
    TK_LOG(Error, "mmap failed for %s", model_json.c_str());
    return Error::LoadFailure;
  }
  // ── added_tokens (small; nlohmann parses the sub-range) ───────
  {
    auto rg = find_key_value(data, data_len, "added_tokens");
    if (rg.first) {
      try {
        std::string frag(rg.first, rg.second);
        json j = json::parse(frag);
        std::vector<std::pair<std::string, uint64_t>> sp;
        if (j.is_array()) {
          for (auto& e : j) {
            if (e.contains("content") && e.contains("id"))
              sp.emplace_back(e["content"].get<std::string>(),
                              e["id"].get<uint64_t>());
          }
        }
        auto r = detail::build_token_map(std::move(sp));
        if (!r.ok()) { munmap((void*)data, data_len); return r.error(); }
        auto rx = detail::build_special_token_regex(*r);
        if (!rx.ok()) { munmap((void*)data, data_len); return rx.error(); }
        special_token_regex_ = std::move(*rx);
        special_token_map_.emplace(std::move(*r));
      } catch (const std::exception& e) {
        munmap((void*)data, data_len);
        TK_LOG(Error, "added_tokens parse failed: %s", e.what());
        return Error::LoadFailure;
      }
    }
  }

  // ── vocab: hand-scan "key": uint pairs ────────────────────────
  {
    auto rg = find_key_value(data, data_len, "vocab");
    if (rg.first) {
      // A single arena holds every decoded token string.  Pre-reserving it to
      // the whole vocab section guarantees no reallocation, so the string_views
      // collected below never dangle while StringIntegerMap::init copies them.
      // This replaces one std::string heap object per token (with its allocator
      // churn and SSO object header) with one contiguous allocation, and the
      // arena is freed immediately after the map is built.
      std::string arena;
      arena.reserve(static_cast<size_t>(rg.second - rg.first));
      std::vector<std::pair<std::string_view, uint64_t>> tp;

      // Count entries first so tp is reserved exactly.  A fixed len/10 divisor
      // over-allocates ~1.7x on gemma4 (393K entries reserved from a 692K
      // estimate -> ~7 MB wasted at peak, per massif).  This pre-pass only
      // skips strings and digits, allocating nothing.
      size_t n_entries = 0;
      {
        const char* q = rg.first + 1;
        const char* qe = rg.second - 1;
        while (q < qe) {
          q = skip_ws(q, qe);
          if (q >= qe || *q != '"') break;
          q = skip_json_string(q, qe);                     // skip key
          q = skip_ws(q, qe);
          if (q >= qe || *q != ':') break;
          q = skip_ws(q + 1, qe);
          while (q < qe && *q >= '0' && *q <= '9') ++q;    // skip uint value
          ++n_entries;
          q = skip_ws(q, qe);
          if (q < qe && *q == ',') { ++q; continue; }
          break;
        }
      }
      tp.reserve(n_entries);

      const char* p = rg.first + 1;          // past '{'
      const char* e = rg.second - 1;         // before '}'
      while (p < e) {
        p = skip_ws(p, e);
        if (p >= e || *p != '"') break;
        std::string_view key = decode_json_string_into(arena, p, e, &p);
        p = skip_ws(p, e);
        if (p >= e || *p != ':') break;
        p = skip_ws(p + 1, e);
        uint64_t id = 0;
        if (!decode_uint(p, e, &id, &p)) break;
        if (!special_token_map_->tryGetString(id))
          tp.emplace_back(key, id);
        p = skip_ws(p, e);
        if (p < e && *p == ',') { ++p; continue; }
        break;
      }
      auto r = detail::build_token_map(std::move(tp));
      if (!r.ok()) { munmap((void*)data, data_len); return r.error(); }
      token_map_.emplace(std::move(*r));
    }
  }
  vocab_size_ = token_map_->size() + special_token_map_->size();

  // ── merges: hand-scan "a b" string array ──────────────────────
  {
    auto rg = find_key_value(data, data_len, "merges");
    if (rg.first) {
      merge_map_ = std::make_unique<detail::MergeMap>();
      const char* p = rg.first + 1;          // past '['
      const char* e = rg.second - 1;         // before ']'
      uint32_t mi = 0;
      while (p < e) {
        p = skip_ws(p, e);
        if (p >= e) break;
        std::string f, sec;
        if (*p == '"') {
          // String format: "a b"  (gemma4)
          std::string s = decode_json_string(p, e, &p);
          if (!s.empty() && s.rfind("#version", 0) != 0) {
            size_t sp = s.find(' ');
            if (sp != std::string::npos) {
              f = s.substr(0, sp);
              sec = s.substr(sp + 1);
            }
          }
        } else if (*p == '[') {
          // Array format: ["a", "b"]  (Qwen3)
          ++p;
          p = skip_ws(p, e);
          if (p < e && *p == '"') f = decode_json_string(p, e, &p);
          p = skip_ws(p, e);
          if (p < e && *p == ',') { ++p; p = skip_ws(p, e); }
          if (p < e && *p == '"') sec = decode_json_string(p, e, &p);
          p = skip_ws(p, e);
          if (p < e && *p == ']') ++p;
        } else {
          break;  // unknown format
        }
        if (!f.empty() && !sec.empty()) {
          auto fid = token_map_->tryGetInteger(f);
          auto sid = token_map_->tryGetInteger(sec);
          if (fid && sid) {
            std::string merged = f + sec;
            auto mid = token_map_->tryGetInteger(merged);
            if (mid)
              merge_map_->push_back(
                  detail::MergeEntry{static_cast<uint32_t>(*fid),
                                     static_cast<uint32_t>(*sid),
                                     static_cast<uint32_t>(*mid), mi});
          }
        }
        ++mi;
        p = skip_ws(p, e);
        if (p < e && *p == ',') { ++p; continue; }
        break;
      }
      std::sort(merge_map_->begin(), merge_map_->end(),
                [](const detail::MergeEntry& a, const detail::MergeEntry& b) {
                  return a.fid < b.fid ||
                         (a.fid == b.fid && a.sid < b.sid);
                });
    }
  }

  // ── config sections (small; nlohmann parses each sub-range) ──
  json cfg;
  {
    auto parse_sub = [&](const char *key) {
      auto rg = find_key_value(data, data_len, key);
      if (rg.first) {
        std::string frag(rg.first, rg.second);
        try { cfg[key] = json::parse(frag); }
        catch (...) {}
      }
    };
    parse_sub("normalizer");
    parse_sub("pre_tokenizer");
    parse_sub("post_processor");
    parse_sub("decoder");

    cfg["model"] = json::object();
    {
      auto rg = find_key_value(data, data_len, "byte_fallback");
      if (rg.first) {
        std::string frag(rg.first, rg.second);
        try { cfg["model"]["byte_fallback"] = json::parse(frag); }
        catch (...) {}
      }
    }
    {
      auto rg = find_key_value(data, data_len, "unk_token");
      if (rg.first) {
        std::string frag(rg.first, rg.second);
        try { cfg["model"]["unk_token"] = json::parse(frag); }
        catch (...) {}
      }
    }
  }

  munmap((void*)data, data_len);
#if defined(__GLIBC__)
  malloc_trim(0);
#endif

  // ── Setup config-driven components ─────────────────────────
  TK_CHECK_OK_OR_RETURN_ERROR(setup_normalizer(cfg));
  TK_CHECK_OK_OR_RETURN_ERROR(setup_pretokenizer(cfg));
  TK_CHECK_OK_OR_RETURN_ERROR(setup_postprocessor(cfg));
  TK_CHECK_OK_OR_RETURN_ERROR(setup_decoder(cfg));
  TK_CHECK_OK_OR_RETURN_ERROR(setup_special_token_ids(
      path, cfg, model_config_json, special_tokens_map_json));

  initialized_ = true;
  return Error::Ok;
}

Result<std::vector<uint64_t>>
HFTokenizer::encode(const std::string& input, int8_t bos, int8_t eos) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }

  auto encode_result = encode_with_special_token_(input, *special_token_map_);
  if (!encode_result.ok()) {
    return encode_result.error();
  }
  std::vector<uint64_t> tokens = std::move((*encode_result).first);

  bool add_special = (bos > 0 || eos > 0);
  if (_postprocessor) {
    tokens = _postprocessor->process(tokens, add_special);
  }

  return tokens;
}

Result<std::string> HFTokenizer::decode(
    const std::vector<uint64_t>& tokens,
    bool skip_special_tokens) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  std::vector<std::string> pieces;
  for (uint64_t token : tokens) {
    std::string_view token_bytes;
    auto regular_token_result = token_map_->tryGetString(token);
    if (regular_token_result) { // Found in regular tokens
      token_bytes = *regular_token_result;
    } else { // Not a regular token, check if it's a special token
      auto special_token_result = special_token_map_->tryGetString(token);
      if (special_token_result) { // It's a special token
        if (skip_special_tokens) {
          continue;
        }
        token_bytes = *special_token_result; // Don't skip, use its string
      } else { // Unknown token
        TK_LOG(Error, "unknown token: %" PRIu64 "\n", token);
        return Error::DecodeFailure;
      }
    }
    pieces.push_back(std::string(token_bytes));
  }

  auto decoded_pieces = _decode(pieces);

  std::string result_str;
  for (const auto& p : decoded_pieces) {
    result_str += p;
  }

  return result_str;
}

// -------------------------private method start--------------------------------

Error HFTokenizer::_encode(
    const std::string& input,
    std::vector<uint64_t>& ret,
    uint64_t& last_piece_token_len) const {
  std::string normalized_input = input;
  if (_normalizer) {
    normalized_input = _normalizer->normalize(input);
  }

  std::vector<std::string> pieces;
  if (_pretokenizer) {
    pieces = _pretokenizer->pre_tokenize(normalized_input);
  } else {
    pieces.push_back(normalized_input);
  }

  for (const auto& piece : pieces) {
    const auto result = token_map_->tryGetInteger(piece);
    if (result) {
      ret.push_back(*result);
      last_piece_token_len = 1;
      continue;
    }
    auto tokens_result = byte_pair_encode_(piece, *token_map_);
    if (!tokens_result.ok()) {
      return tokens_result.error();
    }
    auto piece_tokens = std::move(*tokens_result);
    ret.insert(ret.end(), piece_tokens.begin(), piece_tokens.end());
    last_piece_token_len = piece_tokens.size();
  }
  return Error::Ok;
}

void HFTokenizer::_decode(const std::string& input, std::string& ret) const {
  if (_decoder) {
    auto result = _decoder->decode({input});
    for (auto& piece : result) {
      ret += piece;
    }
  } else {
    ret += input;
  }
}

std::vector<std::string> HFTokenizer::_decode(
    const std::vector<std::string>& pieces) const {
  if (_decoder) {
    return _decoder->decode(pieces);
  }
  return pieces;
}

Result<std::vector<uint64_t>> HFTokenizer::byte_pair_encode_(
    const std::string& piece,
    const detail::TokenMap& token_map) const {
  if (piece.size() == 1) {
    const auto result = token_map.tryGetInteger(piece);
    if (result) {
      return std::vector<uint64_t>(1, *result);
    }
    if (byte_fallback_) {
      if (!unk_token_is_configured_) {
        return std::vector<uint64_t>(); // skip
      }
      auto find_id = [&](const std::string& key) -> std::optional<std::uint64_t> {
        auto r = token_map.tryGetInteger(key);
        if (!r) {
          r = special_token_map_->tryGetInteger(key);
        }
        return r;
      };
      std::optional<std::uint64_t> byte_id;
      {
        char hex[7];
        snprintf(hex, sizeof(hex), "<0x%02X>",
                 static_cast<unsigned char>(piece[0]));
        byte_id = find_id(std::string(hex));
      }
      if (!byte_id) {
        std::string key = unicode_byte_to_utf8(
            static_cast<unsigned char>(piece[0]));
        byte_id = find_id(key);
      }
      if (byte_id) {
        return std::vector<uint64_t>(1, *byte_id);
      }
      return std::vector<uint64_t>(1, unk_tok_);
    } else {
      if (unk_token_is_configured_) {
        return std::vector<uint64_t>(1, unk_tok_);
      }
      return std::vector<uint64_t>(); // skip
    }
  }

  // The HFTokenizer override of _byte_pair_merge merges via merge_map_ and
  // ignores the ranks argument, so token_map is passed only to satisfy the
  // signature.
  return _byte_pair_merge(
      piece,
      token_map,
      [this, &piece, &token_map](uint64_t start, uint64_t stop) {
        std::string_view key(piece.data() + start, stop - start);
        const auto result = token_map.tryGetInteger(key);
        if (result) {
          return *result;
        }
        if (byte_fallback_) {
          return UINT64_MAX;
        }
        if (unk_token_is_configured_) {
          return unk_tok_;
        }
        // When neither byte_fallback nor unk_token is configured,
        // still return UINT64_MAX rather than UINT64_MAX-1 so that
        // _byte_pair_merge can skip individual unknown characters
        // instead of discarding the entire BPE piece.
        return UINT64_MAX;
      });
}

std::vector<uint64_t> HFTokenizer::_byte_pair_merge(
    const std::string& piece,
    const detail::TokenMap& /*ranks*/,
    std::function<uint64_t(uint64_t, uint64_t)> func) const {
  HFWord word;
  size_t i = 0;
  while (i < piece.size()) {
    size_t char_start = i;
    size_t char_len = 1;
    unsigned char byte = static_cast<unsigned char>(piece[i]);
    if ((byte & 0x80) == 0) {
      char_len = 1;
    } else if ((byte & 0xE0) == 0xC0) {
      char_len = 2;
    } else if ((byte & 0xF0) == 0xE0) {
      char_len = 3;
    } else if ((byte & 0xF8) == 0xF0) {
      char_len = 4;
    }
    if (char_start + char_len > piece.size()) {
      char_len = piece.size() - char_start;
    }

    uint64_t token_id = func(char_start, char_start + char_len);
    if (token_id == UINT64_MAX) { // This is the sentinel for byte_fallback
      // When unk_token is not configured, silently skip unknown characters
      // (matching Python BPE behavior for models without unk_token, e.g.
      // Janus_Pro). Splitting into byte tokens would inject noise that the
      // BPE merge rules were never trained to handle.
      if (!unk_token_is_configured_) {
        // skip
      } else {
        for (size_t j = 0; j < char_len; ++j) {
          unsigned char byte = static_cast<unsigned char>(piece[char_start + j]);

          // Try SentencePiece <0x%02X> hex format first, then standard
          // BPE byte-level encoding (GPT2 bytes_to_unicode). Both formats
          // are used by different tokenizer.json producers for byte tokens.
          // Both token_map_ and special_token_map_ are consulted because
          // some formats store byte tokens in added_tokens (special map).
          std::optional<std::uint64_t> byte_id;
          auto find_id = [&](const std::string& key) {
            auto r = token_map_->tryGetInteger(key);
            if (!r) {
              r = special_token_map_->tryGetInteger(key);
            }
            return r;
          };
          {
            char hex[7];
            snprintf(hex, sizeof(hex), "<0x%02X>", byte);
            byte_id = find_id(std::string(hex));
          }
          if (!byte_id) {
            std::string key = unicode_byte_to_utf8(byte);
            byte_id = find_id(key);
          }

          if (byte_id) {
            word.add(*byte_id, 1);
          } else {
            word.add(unk_tok_, 1);
          }
        }
      }
    } else if (token_id == (UINT64_MAX - 1)) { // This is the sentinel for a
                                               // generic err
      return {}; // Return empty on error
    } else { // If it's not a sentinel, it's a valid token_id (could be 0, for
             // [UNK] oranother token)
      word.add(token_id, char_len); // Add any valid token_id
    }
    i += char_len;
  }

  if (merge_map_) {
    word.merge_all(*merge_map_);
  }
  return word.tokens;
}

Error HFTokenizer::parse_special_tokens(const json& parsed_json) {
  try {
    const auto& special_tokens = parsed_json.at("added_tokens");
    auto special_token_map_result = detail::build_token_map(
        special_tokens,
        [](const json& it) -> std::string { return it.at("content"); },
        [](const json& it) -> std::uint64_t { return it.at("id"); });
    if (!special_token_map_result.ok()) {
      return special_token_map_result.error();
    }
    auto special_token_map = std::move(*special_token_map_result);

    auto special_token_regex_result =
        detail::build_special_token_regex(special_token_map);
    if (!special_token_regex_result.ok()) {
      return special_token_regex_result.error();
    }
    special_token_regex_ = std::move(*special_token_regex_result);
    special_token_map_.emplace(std::move(special_token_map));
  } catch (const std::exception& e) {
    TK_LOG(Info, "Could not parse special tokens: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::parse_tokens(const json& parsed_json) {
  try {
    std::vector<std::pair<std::string, std::uint64_t>> token_pairs;
    const auto& vocab = parsed_json.at("/model/vocab"_json_pointer);
    for (const auto& entry : vocab.items()) {
      const std::string token = entry.key();
      const uint64_t token_id = entry.value();
      if (!special_token_map_->tryGetString(token_id)) {
        token_pairs.emplace_back(token, token_id);
      }
    }

    auto token_map_result = detail::build_token_map(std::move(token_pairs));
    if (!token_map_result.ok()) {
      return token_map_result.error();
    }
    token_map_.emplace(std::move(*token_map_result));
  } catch (const std::exception& e) {
    TK_LOG(Info, "Could not parse tokens: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_normalizer(const json& parsed_json) {
  try {
    if (parsed_json.contains("normalizer") &&
        !parsed_json.at("normalizer").is_null()) {
      const auto& normalizer_json = parsed_json.at("normalizer");
      _normalizer = NormalizerConfig().parse_json(normalizer_json).create();
    }
  } catch (const std::exception& e) {
    TK_LOG(Error, "Failed to setup normalizer: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_pretokenizer(const json& parsed_json) {
  try {
    if (parsed_json.contains("pre_tokenizer") &&
        !parsed_json.at("pre_tokenizer").is_null()) {
      const auto& pretokenizer_json = parsed_json.at("pre_tokenizer");
      _pretokenizer =
          PreTokenizerConfig().parse_json(pretokenizer_json).create();
    }
  } catch (const std::exception& e) {
    TK_LOG(Error, "Failed to setup pretokenizer: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_postprocessor(const json& parsed_json) {
  try {
    if (parsed_json.contains("post_processor") &&
        !parsed_json.at("post_processor").is_null()) {
      const auto& post_processor_json = parsed_json.at("post_processor");
      _postprocessor =
          PostProcessorConfig().parse_json(post_processor_json).create();
    }
  } catch (const std::exception& e) {
    TK_LOG(Error, "Failed to setup post_processor: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_decoder(const json& parsed_json) {
  try {
    if (parsed_json.contains("decoder") &&
        !parsed_json.at("decoder").is_null()) {
      _decoder =
          TokenDecoderConfig().parse_json(parsed_json.at("decoder")).create();
    }
  } catch (const std::exception& e) {
    TK_LOG(Error, "Failed to setup decoder: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::parse_merges(const json& parsed_json) {
  try {
    const auto& merges = parsed_json.at("/model/merges"_json_pointer);
    std::vector<std::pair<std::string, std::string>> merge_pairs;

    for (const auto& merge : merges) {
      std::string first, second;
      if (merge.is_string()) {
        std::string merge_str = merge.get<std::string>();
        if (merge_str.rfind("#version", 0) == 0) {
          continue;
        }
        auto space_pos = merge_str.find(' ');
        if (space_pos != std::string::npos) {
          first = merge_str.substr(0, space_pos);
          second = merge_str.substr(space_pos + 1);
        }
      } else if (merge.is_array() && merge.size() == 2) {
        first = merge[0].get<std::string>();
        second = merge[1].get<std::string>();
      }
      if (!first.empty() && !second.empty()) {
        merge_pairs.emplace_back(first, second);
      }
    }

    merge_map_ = std::make_unique<detail::MergeMap>();
    for (size_t i = 0; i < merge_pairs.size(); ++i) {
      const auto& merge_pair = merge_pairs[i];
      const auto& first = merge_pair.first;
      const auto& second = merge_pair.second;
      auto first_id = token_map_->tryGetInteger(first);
      auto second_id = token_map_->tryGetInteger(second);
      if (first_id && second_id) {
        std::string merged = first + second;
        auto merged_id = token_map_->tryGetInteger(merged);
        if (merged_id) {
          merge_map_->push_back(
              detail::MergeEntry{static_cast<uint32_t>(*first_id),
                                 static_cast<uint32_t>(*second_id),
                                 static_cast<uint32_t>(*merged_id),
                                 static_cast<uint32_t>(i)});
        }
      }
    }
    std::sort(merge_map_->begin(), merge_map_->end(),
              [](const detail::MergeEntry& a, const detail::MergeEntry& b) {
                return a.fid < b.fid ||
                       (a.fid == b.fid && a.sid < b.sid);
              });
  } catch (const std::exception& e) {
    TK_LOG(Error, "Could not parse merges: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_special_token_ids(
    const std::string& /*path*/,
    const json& parsed_json,
    const std::string& model_config_json,
    const std::string& special_tokens_map_json) {
  std::string config_bos_token;
  std::string config_eos_token;
  std::string config_unk_token;
  bool explicit_unk_null = false;

  try {
    const auto& model_json = parsed_json.at("model");
    if (model_json.contains("byte_fallback")) {
      byte_fallback_ = model_json.at("byte_fallback").get<bool>();
    }
    if (model_json.contains("unk_token")) {
      if (model_json.at("unk_token").is_null()) {
        explicit_unk_null = true;
      } else {
        config_unk_token = model_json.at("unk_token").get<std::string>();
      }
    }
  } catch (...) {
  }

  auto process_config_file = [&](const std::string& file_path) {
    if (file_path.empty()) {
      return;
    }
    std::ifstream f(file_path);
    if (!f) {
      return;
    }
    try {
      json j = json::parse(f);
      if (j.contains("bos_token")) {
        config_bos_token = extract_token_string(j["bos_token"]);
      }
      if (j.contains("eos_token")) {
        config_eos_token = extract_token_string(j["eos_token"]);
      }
      if (config_unk_token.empty() && !explicit_unk_null &&
          j.contains("unk_token")) {
        if (j["unk_token"].is_null()) {
          explicit_unk_null = true;
        } else {
          config_unk_token = extract_token_string(j["unk_token"]);
        }
      }
    } catch (...) {
    }
  };

  process_config_file(special_tokens_map_json);
  if (config_bos_token.empty() || config_eos_token.empty() ||
      (config_unk_token.empty() && !explicit_unk_null)) {
    process_config_file(model_config_json);
  }
  auto set_special =
      [&](const std::string& token_str, uint64_t& target_id, bool& found_flag) {
        if (!token_str.empty()) {
          auto id = special_token_map_->tryGetInteger(token_str);
          if (id) {
            target_id = *id;
            found_flag = true;
          }
        }
      };

  bool bos_found = false;
  bool eos_found = false;
  set_special(config_bos_token, bos_tok_, bos_found);
  set_special(config_eos_token, eos_tok_, eos_found);

  if (!config_unk_token.empty()) {
    auto id = special_token_map_->tryGetInteger(config_unk_token);
    if (id) {
      unk_tok_ = *id;
      unk_token_is_configured_ = true;
    }
  }

  if (!unk_token_is_configured_ && !explicit_unk_null) {
    for (const auto& name : {"<unk>", "[UNK]", "<|endoftext|>"}) {
      auto id = special_token_map_->tryGetInteger(name);
      if (id) {
        unk_tok_ = *id;
        unk_token_is_configured_ = true;
        break;
      }
    }
  }

  if (!bos_found || !eos_found) {
    std::vector<std::string_view> bos_c, eos_c;
    for (size_t i = 0; i < special_token_map_->size(); ++i) {
      const auto& element = special_token_map_->getElement(i);
      const auto& token = element.first;
      if (!bos_found &&
          (token.find("bos") != std::string::npos ||
           token.find("begin") != std::string::npos))
        bos_c.push_back(token);
      if (!eos_found &&
          (token.find("eos") != std::string::npos ||
           token.find("end") != std::string::npos))
        eos_c.push_back(token);
    }
    if (!bos_found && bos_c.size() == 1) {
      bos_tok_ = *special_token_map_->tryGetInteger(std::string(bos_c[0]));
      bos_found = true;
    }
    if (!eos_found && eos_c.size() == 1) {
      eos_tok_ = *special_token_map_->tryGetInteger(std::string(eos_c[0]));
      eos_found = true;
    }
    if (bos_found && !eos_found) {
      eos_tok_ = bos_tok_;
      eos_found = true;
    } else if (!bos_found && eos_found) {
      bos_tok_ = eos_tok_;
      bos_found = true;
    }
  }

  return Error::Ok;
}

} // namespace tokenizers
