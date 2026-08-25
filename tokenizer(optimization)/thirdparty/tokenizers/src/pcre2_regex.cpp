/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <vector>

#include <pytorch/tokenizers/pcre2_regex.h>

namespace tokenizers {

// Helper function to check if error code is a UTF-8 validation error
// PCRE2 UTF-8 error codes range from PCRE2_ERROR_UTF8_ERR1 (-3) to
// PCRE2_ERROR_UTF8_ERR21 (-23)
static bool is_utf8_error(int error_code) {
  return error_code >= PCRE2_ERROR_UTF8_ERR21 &&
      error_code <= PCRE2_ERROR_UTF8_ERR1;
}

Error Pcre2Regex::compile(const std::string& pattern) {
  int error_code;
  PCRE2_SIZE error_offset;

  // Compile with UTF-8 + UCP.  PCRE2_UCP makes \s \d \w match Unicode
  // properties, matching the Rust regex crate's Unicode-aware \s semantics
  // that HuggingFace tokenizers use.
  regex_ = pcre2_compile(
      reinterpret_cast<PCRE2_SPTR>(pattern.c_str()),
      pattern.length(),
      PCRE2_UCP | PCRE2_UTF,
      &error_code,
      &error_offset,
      nullptr);

  if (regex_ == nullptr) {
    PCRE2_UCHAR error_buffer[256];
    pcre2_get_error_message(error_code, error_buffer, sizeof(error_buffer));

    // Check if this is a UTF-8 validation error
    if (is_utf8_error(error_code)) {
      TK_LOG(
          Info,
          "PCRE2 UTF-8 validation failed at offset %" PRId64
          ": %s. Retrying without UTF flags.",
          static_cast<int64_t>(error_offset),
          error_buffer);

      // Retry compilation without PCRE2_UTF flag
      regex_ = pcre2_compile(
          reinterpret_cast<PCRE2_SPTR>(pattern.c_str()),
          pattern.length(),
          PCRE2_UCP,
          &error_code,
          &error_offset,
          nullptr);

      if (regex_ == nullptr) {
        pcre2_get_error_message(error_code, error_buffer, sizeof(error_buffer));
        TK_LOG(
            Error,
            "PCRE2 compilation failed (without UTF) at offset %" PRId64 ": %s",
            static_cast<int64_t>(error_offset),
            error_buffer);
        return Error::RegexFailure;
      }
    } else {
      TK_LOG(
          Error,
          "PCRE2 compilation failed at offset %" PRId64 ": %s",
          static_cast<int64_t>(error_offset),
          error_buffer);
      return Error::RegexFailure;
    }
  }

  // JIT-compile for fast repeated matching (find_all on long Split inputs makes
  // one pcre2_match call per piece; JIT cuts each call from ~50-100us to a few
  // us).  If the platform can't JIT (no executable memory etc.), this returns a
  // nonzero code and matching falls back to the interpreter.
  int jit_rc = pcre2_jit_compile(regex_, PCRE2_JIT_COMPLETE);
  if (jit_rc != 0) {
    TK_LOG(Debug, "PCRE2 JIT unavailable (rc=%d), using interpreter", jit_rc);
  }

  return Error::Ok;
}

Pcre2Regex::~Pcre2Regex() {
  if (regex_) {
    pcre2_code_free(regex_);
  }
}

std::vector<Match> Pcre2Regex::find_all(const std::string& text) const {
  std::vector<Match> result;

  if (!regex_) {
    TK_LOG(Error, "Regex is not compiled or invalid, run compile() first");
    return result;
  }

  // Allocate match_data per call rather than sharing it across calls.
  // PCRE2 writes match offsets into match_data during pcre2_match(), so a
  // shared buffer races under concurrent find_all() invocations: it silently
  // corrupts results and can trigger heap-buffer-overflow inside PCRE2.
  pcre2_match_data* match_data =
      pcre2_match_data_create_from_pattern(regex_, nullptr);
  if (match_data == nullptr) {
    TK_LOG(Error, "Failed to create PCRE2 match data");
    return result;
  }

  PCRE2_SIZE* ovector;
  PCRE2_SPTR subject = reinterpret_cast<PCRE2_SPTR>(text.c_str());
  PCRE2_SIZE subject_length = text.length();
  PCRE2_SIZE offset = 0;

  // Try JIT first (compiled in compile()); falls back to the interpreter if
  // the JIT function returns PCRE2_ERROR_JIT_BADOPTION / not available.
  while (offset < subject_length) {
    int rc = pcre2_jit_match(
        regex_,
        subject,
        subject_length,
        offset,
        0, // Default options
        match_data,
        nullptr);
    if (rc == PCRE2_ERROR_JIT_BADOPTION) {
      rc = pcre2_match(
          regex_,
          subject,
          subject_length,
          offset,
          0, // Default options
          match_data,
          nullptr);
    }

    if (rc < 0) {
      if (rc == PCRE2_ERROR_NOMATCH) {
        break; // No more matches
      } else {
        // Error occurred
        PCRE2_UCHAR error_buffer[256];
        pcre2_get_error_message(rc, error_buffer, sizeof(error_buffer));
        std::cerr << "PCRE2 matching error: " << error_buffer << std::endl;
        break;
      }
    }

    ovector = pcre2_get_ovector_pointer(match_data);

    // Add the match to the result
    result.push_back({ovector[0], ovector[1]});

    // Move to the next position after the match
    offset = ovector[1];

    // If the match was empty, move forward by one character to avoid infinite
    // loop
    if (ovector[0] == ovector[1]) {
      offset++;
    }
  }

  pcre2_match_data_free(match_data);
  return result;
}

} // namespace tokenizers
