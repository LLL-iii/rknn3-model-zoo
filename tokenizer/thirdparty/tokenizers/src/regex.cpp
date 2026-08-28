/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Default implementation for create_regex.  All patterns are dispatched to the
// fallback regex engine (PCRE2 via regex_lookahead.cpp), so no RE2 dependency
// is required.

#include <pytorch/tokenizers/regex.h>

namespace tokenizers {

// Default implementation that returns failure
static Result<std::unique_ptr<IRegex>> default_create_fallback_regex(
    const std::string& pattern) {
  (void)pattern;
  return tokenizers::Error::RegexFailure;
}

FallbackRegexFn fallback_regex = default_create_fallback_regex;

bool register_override_fallback_regex(FallbackRegexFn fn) {
  TK_LOG(Debug, "Registering override fallback regex");
  fallback_regex = fn;
  return true;
}

FallbackRegexFn get_fallback_regex() {
  return fallback_regex;
}

std::string IRegex::escape(const std::string& input) {
  std::string result;
  result.reserve(input.size() * 2); // Reserve space for potential escaping

  for (char c : input) {
    // Escape regex special characters to treat them as literal strings
    if (c == '\\' || c == '^' || c == '$' || c == '.' || c == '|' || c == '?' ||
        c == '*' || c == '+' || c == '(' || c == ')' || c == '[' || c == ']' ||
        c == '{' || c == '}') {
      result += '\\';
    }
    result += c;
  }

  return result;
}

Result<std::unique_ptr<IRegex>> create_regex(const std::string& pattern) {
  // All patterns go through PCRE2 (linked in via regex_lookahead).  PCRE2
  // supports lookahead, full Unicode properties (\p{...}), and is the single
  // regex engine after RE2 + abseil were removed to enable C++11 builds.
  auto res = get_fallback_regex()(pattern);
  if (res.ok()) {
    return res;
  }

  TK_LOG(
      Error,
      "Failed to compile regex with PCRE2. Link with `regex_lookahead` to enable support.");
  return tokenizers::Error::RegexFailure;
}
} // namespace tokenizers
