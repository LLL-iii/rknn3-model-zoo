#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct unicode_cpt_flags {
    enum {
        UNDEFINED       = 0x0001,
        NUMBER          = 0x0002,  // regex: \p{N}
        LETTER          = 0x0004,  // regex: \p{L}
        SEPARATOR       = 0x0008,  // regex: \p{Z}
        ACCENT_MARK     = 0x0010,  // regex: \p{M}
        PUNCTUATION     = 0x0020,  // regex: \p{P}
        SYMBOL          = 0x0040,  // regex: \p{S}
        CONTROL         = 0x0080,  // regex: \p{C}
        MASK_CATEGORIES = 0x00FF,
    };

    uint16_t is_undefined   : 1;
    uint16_t is_number      : 1;  // regex: \p{N}
    uint16_t is_letter      : 1;  // regex: \p{L}
    uint16_t is_separator   : 1;  // regex: \p{Z}
    uint16_t is_accent_mark : 1;  // regex: \p{M}
    uint16_t is_punctuation : 1;  // regex: \p{P}
    uint16_t is_symbol      : 1;  // regex: \p{S}
    uint16_t is_control     : 1;  // regex: \p{C}
    uint16_t is_whitespace  : 1;  // regex: \s
    uint16_t is_lowercase   : 1;
    uint16_t is_uppercase   : 1;
    uint16_t is_nfd         : 1;

    inline unicode_cpt_flags(const uint16_t flags = 0) {
        *reinterpret_cast<uint16_t*>(this) = flags;
    }

    inline uint16_t as_uint() const {
        return *reinterpret_cast<const uint16_t*>(this);
    }

    inline uint16_t category_flag() const {
        return this->as_uint() & MASK_CATEGORIES;
    }
};

// 连续大 buffer + 每个 word 长度。用于替代 vector<string>，减少 tokenizer 预分词阶段的小块分配。
struct regex_split_result {
    std::string buffer;
    std::vector<size_t> word_lengths;
};

size_t unicode_len_utf8(char src);

std::string unicode_cpt_to_utf8  (uint32_t cpt);
uint32_t    unicode_cpt_from_utf8(const std::string & utf8, size_t & offset);

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8);
void unicode_cpts_from_utf8_into(const std::string & utf8, std::vector<uint32_t> & result);

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts);

unicode_cpt_flags unicode_cpt_flags_from_cpt (uint32_t cpt);
// unicode_cpt_flags unicode_cpt_flags_from_utf8(const std::string & utf8);

std::string unicode_byte_to_utf8(uint8_t byte);
uint8_t     unicode_utf8_to_byte(const std::string & utf8);

uint32_t unicode_tolower(uint32_t cpt);

regex_split_result unicode_regex_split(
    const std::string & text,
    const std::vector<std::string> & regex_exprs,
    bool byte_encode = true);

void unicode_regex_split_into(
    const std::string & text,
    const std::vector<std::string> & regex_exprs,
    bool byte_encode,
    regex_split_result & result,
    std::vector<uint32_t> & cpts_buf,
    std::vector<size_t> & offsets_buf,
    std::vector<size_t> & tmp_offsets_buf);