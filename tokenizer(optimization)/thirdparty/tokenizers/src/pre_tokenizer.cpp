/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

// Local
#include <pytorch/tokenizers/pre_tokenizer.h>
#include <pytorch/tokenizers/regex.h>
#include <unicode.h>

// Standard
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Third Party
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace tokenizers {

// PreTokenizerConfig //////////////////////////////////////////////////////////

PreTokenizerConfig::PreTokenizerConfig(std::string type)
    : type(std::move(type)) {}

PreTokenizer::Ptr PreTokenizerConfig::create() const {
  // NOTE: These types must line up with the type strings found in the
  //  tokenizers library
  //  https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/mod.rs#L73
  if (type == "Split") {
    if (!pattern) {
      throw std::runtime_error(
          "Missing pattern for PreTokenizer of type Split");
    }

    // Validate behavior parameter, if missing set to default "Removed"
    std::string behavior_str = behavior ? *behavior : "Removed";
    if (behavior_str != "MergedWithPrevious" && behavior_str != "Isolated" &&
        behavior_str != "Removed") {
      throw std::runtime_error(
          "Unsupported behavior '" + behavior_str +
          "' for Split PreTokenizer. Only 'MergedWithPrevious', 'Removed' and 'Isolated' are supported.");
    }

    // Validate invert parameter
    const bool invert_flag = invert ? *invert : false;
    const bool delimiter_flag = is_delimiter ? *is_delimiter : false;
    if (invert_flag && delimiter_flag) {
      throw std::runtime_error(
          "invert=true is not supported for Split PreTokenizer with a String pattern.");
    }

    return PreTokenizer::Ptr(
        new RegexPreTokenizer(*pattern, delimiter_flag, behavior_str));
  }
  if (type == "Digits") {
    if (individual_digits) {
      return PreTokenizer::Ptr(new DigitsPreTokenizer(*individual_digits));
    }
    return PreTokenizer::Ptr(new DigitsPreTokenizer());
  }
  if (type == "ByteLevel") {
    const bool prefix_space = add_prefix_space.value_or(true);
    const bool regex = use_regex.value_or(true);
    if (pattern) {
      return PreTokenizer::Ptr(
          new ByteLevelPreTokenizer(prefix_space, *pattern, regex));
    }
    return PreTokenizer::Ptr(
        new ByteLevelPreTokenizer(prefix_space, "", regex));
  }
  if (type == "Sequence") {
    if (!pretokenizers || pretokenizers->empty()) {
      throw std::runtime_error(
          "Missing pretokenizers for PreTokenizer of type Sequence");
    }
    std::vector<PreTokenizer::Ptr> pretoks;
    std::transform(
        pretokenizers->begin(),
        pretokenizers->end(),
        std::back_inserter(pretoks),
        [](const PreTokenizerConfig& cfg) { return cfg.create(); });
    return PreTokenizer::Ptr(new SequencePreTokenizer(pretoks));
  }
  if (type == "Metaspace") {
    const std::string rep = replacement.value_or("\xE2\x96\x81"); // "▁"
    const std::string ps = prepend_scheme.value_or("always");
    const bool sp = metaspace_split.value_or(true);
    return PreTokenizer::Ptr(new MetaspacePreTokenizer(rep, ps, sp));
  }
  throw std::runtime_error("Unsupported PreTokenizer type: " + type);
}

PreTokenizerConfig& PreTokenizerConfig::parse_json(const json& json_config) {
  type = json_config.at("type");
  if (type == "Split") {
    try {
      pattern = json_config.at("pattern").at("Regex");
      is_delimiter = false;
    } catch (json::out_of_range&) {
      // "Regex" is not there, check "String", which is a delimiter
      std::string delimiter = json_config.at("pattern").at("String");
      // For string patterns, escape regex special characters to treat them as
      // literal strings (same as Rust's regex::escape)
      pattern = IRegex::escape(delimiter);
      is_delimiter = true;
    }

    // Parse behavior and invert fields
    try {
      behavior = json_config.at("behavior");
    } catch (json::out_of_range&) {
      // behavior is optional, default to empty string
    }

    try {
      invert = json_config.at("invert");
    } catch (json::out_of_range&) {
      // invert is optional, default to false
    }
  } else if (type == "Digits") {
    try {
      individual_digits = json_config.at("individual_digits");
    } catch (json::out_of_range&) {
    }
  } else if (type == "ByteLevel") {
    try {
      add_prefix_space = json_config.at("add_prefix_space");
    } catch (json::out_of_range&) {
    }
    try {
      use_regex = json_config.at("use_regex");
    } catch (json::out_of_range&) {
    }
    // TODO: trim_offsets
  } else if (type == "Metaspace") {
    try {
      replacement = json_config.at("replacement");
    } catch (json::out_of_range&) {
      replacement = "\xE2\x96\x81"; // "▁"
    }
    try {
      prepend_scheme = json_config.at("prepend_scheme");
    } catch (json::out_of_range&) {
      // Some older tokenizer.json use "add_prefix_space"
      try {
        bool aps = json_config.at("add_prefix_space");
        prepend_scheme = aps ? "always" : "never";
      } catch (json::out_of_range&) {
        prepend_scheme = "always";
      }
    }
    try {
      metaspace_split = json_config.at("split");
    } catch (json::out_of_range&) {
      metaspace_split = true;
    }
  } else if (type == "Sequence") {
    pretokenizers = std::vector<PreTokenizerConfig>();
    for (const auto& entry : json_config.at("pretokenizers")) {
      pretokenizers->push_back(PreTokenizerConfig().parse_json(entry));
    }
  } else {
    throw std::runtime_error("Unsupported PreTokenizer type: " + type);
  }
  return *this;
}

// RegexPreTokenizer ///////////////////////////////////////////////////////////

std::unique_ptr<IRegex> RegexPreTokenizer::create_regex_(
    const std::string& pattern) {
  assert(!pattern.empty());
  auto regex_result = create_regex(pattern);
  if (!regex_result.ok()) {
    throw std::runtime_error(
        "Error: " + std::to_string(static_cast<int>(regex_result.error())));
  }
  return std::move(regex_result.get());
}

std::vector<std::string> RegexPreTokenizer::pre_tokenize(
    const std::string& input) const {
  if (!regex_) {
    return {};
  }

  std::vector<std::string> results;
  auto matches = regex_->find_all(input);

  if (!is_delimiter_) {
    // When behavior is "Isolated", interleave matched and non-matched text
    // (matching HuggingFace's Split-with-Isolated semantics). The original
    // default behaviour (return matches only) is preserved for patterns
    // that don't set a behaviour — comprehensive regexes like GPT2 cover
    // every character so the distinction is moot for them.
    if (behavior_ == "Isolated") {
      size_t last_end = 0;
      for (const auto& match : matches) {
        if (match.start > last_end) {
          results.push_back(
              input.substr(last_end, match.start - last_end));
        }
        results.push_back(
            input.substr(match.start, match.end - match.start));
        last_end = match.end;
      }
      if (last_end < input.length()) {
        results.push_back(input.substr(last_end));
      }
    } else {
      // Original behavior: return the matches themselves
      for (const auto& match : matches) {
        results.push_back(
            input.substr(match.start, match.end - match.start));
      }
    }
  } else {
    // Delimiter behavior
    if (matches.empty()) {
      // No matches found, return the entire input
      results.push_back(input);
      return results;
    }

    if (behavior_ == "MergedWithPrevious") {
      // MergedWithPrevious: Include delimiter with previous token
      // Example: "the-final--countdown" with delimiter "-"
      // -> ["the-", "final-", "-", "countdown"]
      size_t last_end = 0;

      for (size_t i = 0; i < matches.size(); ++i) {
        const auto& match = matches[i];

        // Add text before the match plus the delimiter
        if (match.start > last_end) {
          std::string token = input.substr(last_end, match.end - last_end);
          results.push_back(token);
        } else {
          // Only delimiter, no preceding text
          std::string delimiter =
              input.substr(match.start, match.end - match.start);
          results.push_back(delimiter);
        }

        last_end = match.end;
      }

      // Add remaining text after the last match (if any)
      if (last_end < input.length()) {
        results.push_back(input.substr(last_end));
      }
    } else if (behavior_ == "Isolated") {
      // Isolated: Keep delimiters as separate tokens
      // Example: "the-final--countdown" with delimiter "-"
      // -> ["the", "-", "final", "-", "-", "countdown"]
      size_t last_end = 0;
      for (const auto& match : matches) {
        // Add text before the match (if any)
        if (match.start > last_end) {
          results.push_back(input.substr(last_end, match.start - last_end));
        }

        // Add the delimiter itself as a separate token
        std::string delimiter =
            input.substr(match.start, match.end - match.start);
        results.push_back(delimiter);

        last_end = match.end;
      }

      // Add remaining text after the last match (if any)
      if (last_end < input.length()) {
        results.push_back(input.substr(last_end));
      }
    } else if (behavior_ == "Removed" || behavior_.empty()) {
      // Default delimiter behavior (split on delimiters, remove delimiters)
      size_t last_end = 0;
      for (const auto& match : matches) {
        // Add text before the match (if any)
        if (match.start > last_end) {
          results.push_back(input.substr(last_end, match.start - last_end));
        }
        last_end = match.end;
      }

      // Add remaining text after the last match (if any)
      if (last_end < input.length()) {
        results.push_back(input.substr(last_end));
      }
    }
  }
  return results;
}

// ByteLevelPreTokenizer ///////////////////////////////////////////////////////

//////////////////
// Impl Details //
//////////////////
namespace {

// Standard GPT2 regex
// https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53
constexpr char GPT2_EXPR[] =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

// Byte-encode the whole input as a single piece, matching the byte-level
// alphabet mapping that unicode_regex_split applies via
// unicode_byte_encoding_process. Used when use_regex is false so the input is
// not split by the GPT2 regex.
std::string byte_encode(const std::string& input) {
  std::string result;
  // Each input byte maps to a byte-level codepoint that is at most 2 bytes in
  // UTF-8, so reserve up front to avoid repeated reallocations on the hot path.
  result.reserve(input.size() * 2);
  for (const unsigned char byte : input) {
    result += unicode_byte_to_utf8(byte);
  }
  return result;
}

} // namespace

//////////////////
// Construction //
//////////////////

ByteLevelPreTokenizer::ByteLevelPreTokenizer(
    bool add_prefix_space,
    const std::string& pattern,
    bool use_regex)
    : pattern_(pattern.empty() ? GPT2_EXPR : pattern),
      add_prefix_space_(add_prefix_space),
      use_regex_(use_regex) {}

std::vector<std::string> ByteLevelPreTokenizer::pre_tokenize(
    const std::string& input) const {
  // Add the prefix space if configured to do so.
  std::string formatted_input = input;
  if (add_prefix_space_ && !formatted_input.empty() &&
      formatted_input[0] != ' ') {
    formatted_input.insert(formatted_input.begin(), ' ');
  }

  // When use_regex is false, do not split with the GPT2 regex; byte-encode the
  // whole input as a single piece (matches HF ByteLevel use_regex=false).
  if (!use_regex_) {
    return {byte_encode(formatted_input)};
  }

  // For patterns containing Unicode character classes, bypass
  // unicode_regex_split.  Its std::regex fallback collapses \p{L} to
  // A-Za-z (ASCII-only), losing all CJK / Hiragana / Maths codepoints.
  bool has_unicode_class = (pattern_.find("\\p{") != std::string::npos);
  if (has_unicode_class) {
    // Compile the regex once and cache it.  ByteLevel inside a Sequence is
    // called once per upstream piece (Digits can split 100k text into ~29K
    // pieces), so compiling on every call was ~30k pcre2_compile calls.
    if (!regex_cached_) {
      auto regex_result = create_regex(pattern_);
      if (regex_result.ok()) {
        regex_cache_ = std::move(regex_result.get());
      }
      regex_cached_ = true;
    }
    if (regex_cache_) {
      auto matches = regex_cache_->find_all(formatted_input);
      if (!matches.empty()) {
        std::vector<std::string> results;
        results.reserve(matches.size());
        for (const auto& match : matches) {
          results.push_back(byte_encode(
              formatted_input.substr(match.start, match.end - match.start)));
        }
        return results;
      }
    }
    // Fall through to unicode_regex_split if PCRE2 also fails.
  }

  return unicode_regex_split(formatted_input, {pattern_});
}

// MetaspacePreTokenizer //////////////////////////////////////////////////////

MetaspacePreTokenizer::MetaspacePreTokenizer(
    const std::string& replacement,
    const std::string& prepend_scheme,
    bool split)
    : replacement_(replacement),
      prepend_scheme_(prepend_scheme),
      split_(split) {
  if (replacement_.empty()) {
    replacement_ = "\xE2\x96\x81";
  }
}

std::vector<std::string> MetaspacePreTokenizer::pre_tokenize(
    const std::string& input) const {
  std::string s = input;

  bool should_prepend =
      prepend_scheme_ == "always" || prepend_scheme_ == "first";
  if (should_prepend && !s.empty()) {
    if (s[0] != ' ' && s[0] != '\t' && s[0] != '\n' && s[0] != '\r') {
      s.insert(s.begin(), ' ');
    }
  }

  // Replace every whitespace character with the replacement string.
  std::string replaced;
  replaced.reserve(s.size() * 2);
  for (char c : s) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      replaced += replacement_;
    } else {
      replaced += c;
    }
  }

  if (!split_) {
    return {replaced};
  }

  // Split on replacement_ and re-attach it as a prefix to each following
  // piece.  Matches HuggingFace Metaspace semantics: the replacement char
  // acts as a boundary that belongs to the next token.
  //
  // Example (replacement = "▁"):
  //   "▁Hello▁world" → ["", "Hello", "world"] → ["▁Hello", "▁world"]
  //   "Hello▁world"  → ["Hello", "world"]      → ["Hello", "▁world"]
  //   "▁▁a"          → ["", "", "a"]           → ["▁", "▁a"]
  const auto rlen = replacement_.size();
  std::vector<std::string> tokens;
  size_t start = 0;
  for (size_t i = 0;; ++i) {
    size_t pos = replaced.find(replacement_, start);
    bool last = (pos == std::string::npos);
    size_t end = last ? replaced.size() : pos;
    std::string piece = replaced.substr(start, end - start);
    if (i == 0 && piece.empty()) {
      // Leading replacement produced an empty segment — skip.
    } else {
      if (i > 0) {
        piece = replacement_ + piece;
      }
      tokens.push_back(std::move(piece));
    }
    if (last) {
      break;
    }
    start = pos + rlen;
  }

  return tokens;
}

// SequencePreTokenizer ////////////////////////////////////////////////////////

SequencePreTokenizer::SequencePreTokenizer(
    std::vector<PreTokenizer::Ptr> pre_tokenizers)
    : pre_tokenizers_(std::move(pre_tokenizers)) {}

std::vector<std::string> SequencePreTokenizer::pre_tokenize(
    const std::string& input) const {
  std::vector<std::string> pieces{std::string(input)};
  for (const auto& pre_tokenizer : pre_tokenizers_) {
    std::vector<std::string> new_pieces;
    for (const auto& piece : pieces) {
      for (const auto& subpiece : pre_tokenizer->pre_tokenize(piece)) {
        new_pieces.push_back(subpiece);
      }
    }
    pieces = std::move(new_pieces);
  }
  return pieces;
}

} // namespace tokenizers
