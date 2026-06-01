#if defined(_MSC_VER)
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#endif

#include "unicode.h"
#include "unicode-data.h"

#include <algorithm>
#include <cassert>
#include <codecvt>
#include <cstddef>
#include <cstdint>
#include <locale>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <stdint.h>
#include <limits>

static inline size_t rk_align_up_size(size_t v, size_t align) {
    return (v + align - 1) & ~(align - 1);
}

// vector reserve 参数是元素个数，这里保证最终申请字节数是 256B 的倍数
static inline size_t rk_reserve_elems_256_bytes(size_t elem_count) {
    const size_t align_bytes = 256;
    const size_t elem_size = sizeof(size_t);

    if (elem_count == 0) {
        elem_count = 1;
    }

    if (elem_count > (std::numeric_limits<size_t>::max() / elem_size)) {
        return elem_count;
    }

    const size_t bytes = elem_count * elem_size;
    const size_t aligned_bytes = rk_align_up_size(bytes, align_bytes);

    return aligned_bytes / elem_size;
}


inline static void throw_runtime_error(std::string what_msg) {
    printf("%s\n", what_msg.c_str());
    ::abort();
};

size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static std::string unicode_cpts_to_utf8(const std::vector<uint32_t> & cps) {
    std::string result;
    for (size_t i = 0; i < cps.size(); ++i) {
        result.append(unicode_cpt_to_utf8(cps[i]));
    }
    return result;
}

uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}

//static std::vector<uint16_t> unicode_cpt_to_utf16(uint32_t cpt) {
//    std::vector<uint16_t> result;
//    if (/* 0x0000 <= cpt && */ cpt <= 0xffff) {
//        result.emplace_back(cpt);
//        return result;
//    }
//    if (0x10000 <= cpt && cpt <= 0x10ffff) {
//        result.emplace_back(0xd800 | ((cpt - 0x10000) >> 10));
//        result.emplace_back(0xdc00 | ((cpt - 0x10000) & 0x03ff));
//        return result;
//    }
//    throw std::invalid_argument("failed to convert codepoint to utf16");
//}

//static std::vector<uint16_t> unicode_cpts_to_utf16(const std::vector<uint32_t> & cps) {
//    std::vector<uint16_t> result;
//    for (size_t i = 0; i < cps.size(); ++i) {
//        auto temp = unicode_cpt_to_utf16(cps[i]);
//        result.insert(result.end(), temp.begin(), temp.end());
//    }
//    return result;
//}

//static uint32_t unicode_cpt_from_utf16(const std::vector<uint16_t> & utf16, size_t & offset) {
//    assert(offset < utf16.size());
//    if (((utf16[0] >> 10) << 10) != 0xd800) {
//        auto result = utf16[offset + 0];
//        offset += 1;
//        return result;
//    }
//
//    if (offset + 1 >= utf16.size() || !((utf16[1] & 0xdc00) == 0xdc00)) {
//        throw std::invalid_argument("invalid character");
//    }
//
//    auto result = 0x10000 + (((utf16[0] & 0x03ff) << 10) | (utf16[1] & 0x03ff));
//    offset += 2;
//    return result;
//}

//static std::vector<uint32_t> unicode_cpts_from_utf16(const std::vector<uint16_t> & utf16) {
//    std::vector<uint32_t> result;
//    size_t offset = 0;
//    while (offset < utf16.size()) {
//        result.push_back(unicode_cpt_from_utf16(utf16, offset));
//    }
//    return result;
//}

static std::vector<unicode_cpt_flags> unicode_cpt_flags_array() {
    std::vector<unicode_cpt_flags> cpt_flags(MAX_CODEPOINTS, unicode_cpt_flags::UNDEFINED);

    assert (unicode_ranges_flags.begin()[0].first == 0);
    assert (unicode_ranges_flags.begin()[unicode_ranges_flags.size()-1].first == MAX_CODEPOINTS);
    for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
        const auto range_ini = unicode_ranges_flags.begin()[i-1];  // codepoint_ini, flags
        const auto range_end = unicode_ranges_flags.begin()[i];    // codepoint_end, flags
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    for (auto cpt : unicode_set_whitespace) {
        cpt_flags[cpt].is_whitespace = true;
    }

    for (auto p : unicode_map_lowercase) {
        cpt_flags[p.second].is_lowercase = true;
    }

    for (auto p : unicode_map_uppercase) {
        cpt_flags[p.second].is_uppercase = true;
    }

    for (auto &range : unicode_ranges_nfd) {  // start, last, nfd
        cpt_flags[range.nfd].is_nfd = true;
    }

    return cpt_flags;
}

static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(ch) == map.end()) {
            map[ch] = unicode_cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return map;
}

static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(unicode_cpt_to_utf8(ch)) == map.end()) {
            map[unicode_cpt_to_utf8(256 + n)] = ch;
            ++n;
        }
    }
    return map;
}

static inline std::wstring unicode_wstring_from_utf8(const std::string & s) {
#if defined(__clang__)
    // disable C++17 deprecation warning for std::codecvt_utf8
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

#if defined(__clang__)
#    pragma clang diagnostic pop
#endif

    return conv.from_bytes(s);
}

static inline const std::unordered_map<uint8_t, std::string> & unicode_byte_to_utf8_map_cached() {
    static const std::unordered_map<uint8_t, std::string> map = unicode_byte_to_utf8_map();
    return map;
}

static inline void append_byte_to_utf8(std::string & out, uint8_t byte) {
    const auto & map = unicode_byte_to_utf8_map_cached();
    out.append(map.at(byte));
}

// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static void unicode_regex_split_custom_gpt2(
        const std::string & text,
        const std::vector<uint32_t> & cpts,
        const std::vector<size_t> & offsets,
        std::vector<size_t> & bpe_offsets) {
    (void) text;
    bpe_offsets.clear(); // store the offset of each word
    if (bpe_offsets.capacity() < offsets.size()) {
        bpe_offsets.reserve(offsets.size());
    }

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // regex: 's|'t|'re|'ve|'m|'ll|'d
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = _get_cpt(pos+1);
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = _get_cpt(pos+2);
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            // regex: <space>?\p{L}+
            if (flags2.is_letter) {
                pos += (cpt == ' ');
                while (flags2.is_letter) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }
            // regex: <space>?\p{N}+
            if (flags2.is_number) {
                pos += (cpt == ' ');
                while (flags2.is_number) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }
            // regex: <space>?[^\s\p{L}\p{N}]+
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                num_whitespaces++;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

}


// LLAMA3 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
static void unicode_regex_split_custom_llama3(
        const std::string & text,
        const std::vector<uint32_t> & cpts,
        const std::vector<size_t> & offsets,
        std::vector<size_t> & bpe_offsets) {
    (void) text;
    bpe_offsets.clear(); // store the offset of each word
    if (bpe_offsets.capacity() < offsets.size()) {
        bpe_offsets.reserve(offsets.size());
    }

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?\p{L}+
            if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
                if (flags.is_letter || _get_flags(pos+1).is_letter) {  // one or more letters
                    pos++;
                    while (_get_flags(pos).is_letter) {
                        pos++;
                    }
                    _add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}{1,3}
            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3 ) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // regex: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

}



// use std::wregex to split the text
static void unicode_regex_split_stl(
        const std::wstring & wtext,
        const std::wstring & regex_expr,
        const std::vector<size_t> & offsets,
        std::vector<size_t> & out_bpe_offsets) {
    // out_bpe_offsets must be a scratch buffer different from offsets. If the
    // same vector is passed, clear() would destroy the input split offsets.
    assert(&offsets != &out_bpe_offsets);

    static thread_local std::unordered_map<std::wstring, std::wregex> wregex_cache;

    std::unordered_map<std::wstring, std::wregex>::iterator cache_it = wregex_cache.find(regex_expr);
    if (cache_it == wregex_cache.end()) {
        cache_it = wregex_cache.emplace(regex_expr, std::wregex(regex_expr)).first;
    }

    const std::wregex & expr = cache_it->second;

    out_bpe_offsets.clear();
    const size_t raw_reserve = wtext.size() + offsets.size() + 8;
    const size_t target_capacity = rk_reserve_elems_256_bytes(raw_reserve);
    if (out_bpe_offsets.capacity() < target_capacity) {
        out_bpe_offsets.reserve(target_capacity);
    }

    size_t start = 0;

    for (size_t oi = 0; oi < offsets.size(); ++oi) {
        const size_t offset = offsets[oi];

        std::wcregex_iterator rit(wtext.data() + start, wtext.data() + start + offset, expr);
        std::wcregex_iterator end;

        int64_t start_idx = 0;

        while (rit != end) {
            const std::wcmatch & match = *rit;

            const int64_t pos = static_cast<int64_t>(match.position());
            const int64_t len = static_cast<int64_t>(match.length());

            if (pos > start_idx) {
                out_bpe_offsets.emplace_back(static_cast<size_t>(pos - start_idx));
            }

            if (len > 0) {
                out_bpe_offsets.emplace_back(static_cast<size_t>(len));
            }

            start_idx = pos + len;
            ++rit;
        }

        if (start_idx < static_cast<int64_t>(offset)) {
            out_bpe_offsets.emplace_back(static_cast<size_t>(static_cast<int64_t>(offset) - start_idx));
        }

        start += offset;
    }
}

// use std::regex to split the text
static void unicode_regex_split_stl(
        const std::string & text,
        const std::string & regex_expr,
        const std::vector<size_t> & offsets,
        std::vector<size_t> & out_bpe_offsets) {
    // out_bpe_offsets must be a scratch buffer different from offsets. If the
    // same vector is passed, clear() would destroy the input split offsets.
    assert(&offsets != &out_bpe_offsets);

    static thread_local std::unordered_map<std::string, std::regex> regex_cache;

    std::unordered_map<std::string, std::regex>::iterator cache_it = regex_cache.find(regex_expr);
    if (cache_it == regex_cache.end()) {
        cache_it = regex_cache.emplace(regex_expr, std::regex(regex_expr)).first;
    }

    const std::regex & expr = cache_it->second;

    out_bpe_offsets.clear();
    const size_t raw_reserve = text.size() + offsets.size() + 8;
    const size_t target_capacity = rk_reserve_elems_256_bytes(raw_reserve);
    if (out_bpe_offsets.capacity() < target_capacity) {
        out_bpe_offsets.reserve(target_capacity);
    }

    size_t start = 0;

    for (size_t oi = 0; oi < offsets.size(); ++oi) {
        const size_t offset = offsets[oi];

        std::cregex_iterator rit(text.data() + start, text.data() + start + offset, expr);
        std::cregex_iterator end;

        int64_t start_idx = 0;

        while (rit != end) {
            const std::cmatch & match = *rit;

            const int64_t pos = static_cast<int64_t>(match.position());
            const int64_t len = static_cast<int64_t>(match.length());

            if (pos > start_idx) {
                out_bpe_offsets.emplace_back(static_cast<size_t>(pos - start_idx));
            }

            if (len > 0) {
                out_bpe_offsets.emplace_back(static_cast<size_t>(len));
            }

            start_idx = pos + len;
            ++rit;
        }

        if (start_idx < static_cast<int64_t>(offset)) {
            out_bpe_offsets.emplace_back(static_cast<size_t>(static_cast<int64_t>(offset) - start_idx));
        }

        start += offset;
    }
}

static bool unicode_regex_split_custom(
        const std::string & text,
        const std::vector<uint32_t> & cpts,
        const std::string & regex_expr,
        const std::vector<size_t> & offsets,
        std::vector<size_t> & bpe_offsets) {
    if (regex_expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
        unicode_regex_split_custom_gpt2(text, cpts, offsets, bpe_offsets);
        return true;
    } else if (
            regex_expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
            regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        unicode_regex_split_custom_llama3(text, cpts, offsets, bpe_offsets);
        return true;
    }

    bpe_offsets.clear();
    return false;
}

//
// interface
//

std::string unicode_cpt_to_utf8(uint32_t cpt) {
    std::string result;

    if (/* 0x00 <= cpt && */ cpt <= 0x7f) {
        result.push_back(cpt);
        return result;
    }
    if (0x80 <= cpt && cpt <= 0x7ff) {
        result.push_back(0xc0 | ((cpt >> 6) & 0x1f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    if (0x800 <= cpt && cpt <= 0xffff) {
        result.push_back(0xe0 | ((cpt >> 12) & 0x0f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    if (0x10000 <= cpt && cpt <= 0x10ffff) {
        result.push_back(0xf0 | ((cpt >> 18) & 0x07));
        result.push_back(0x80 | ((cpt >> 12) & 0x3f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }

    throw std::invalid_argument("invalid codepoint");
}

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    auto comp = [] (const uint32_t cpt, const range_nfd & range) {
        return cpt < range.first;
    };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        auto it = std::upper_bound(unicode_ranges_nfd.begin(), unicode_ranges_nfd.end(), cpt, comp) - 1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}

void unicode_cpts_from_utf8_into(const std::string & utf8, std::vector<uint32_t> & result) {
    result.clear();

    // UTF-8 的 codepoint 数量不会超过字节数。不要用“刚刚好”的
    // reserve(utf8.size())，否则输入长度轻微波动时会反复扩容/释放旧块，
    // 在 tokenizer 高频调用场景下容易制造小块碎片。这里保留 session
    // buffer 的容量，并在需要增长时给 2x 或 +32 的富余量。
    if (result.capacity() < utf8.size()) {
        const size_t cur_cap = result.capacity();
        const size_t grow_cap = cur_cap > 0 ? cur_cap * 2 : 32;
        const size_t need_cap = utf8.size() + 32;
        const size_t new_cap = std::max(grow_cap, need_cap);
        result.reserve(new_cap);
    }

    size_t offset = 0;
    while (offset < utf8.size()) {
        try {
            result.push_back(unicode_cpt_from_utf8(utf8, offset));
        }
        catch (const std::invalid_argument & /*ex*/) {
            ++offset;
            result.emplace_back(0xFFFD); // replacement character
        }
    }
}

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    unicode_cpts_from_utf8_into(utf8, result);
    return result;
}

unicode_cpt_flags unicode_cpt_flags_from_cpt(const uint32_t cpt) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
}

// unicode_cpt_flags unicode_cpt_flags_from_utf8(const std::string & utf8) {
//     static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
//     if (utf8.empty()) {
//         return undef;  // undefined
//     }
//     size_t offset = 0;
//     return unicode_cpt_flags_from_cpt(unicode_cpt_from_utf8(utf8, offset));
// }

std::string unicode_byte_to_utf8(uint8_t byte) {
    const auto & map = unicode_byte_to_utf8_map_cached();
    return map.at(byte);
}

uint8_t unicode_utf8_to_byte(const std::string & utf8) {
    static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    return map.at(utf8);
}

uint32_t unicode_tolower(uint32_t cpt) {
    // binary search
    auto it = std::lower_bound(unicode_map_lowercase.begin(), unicode_map_lowercase.end(), cpt,
        [](const std::pair<uint32_t, uint32_t> & pair, uint32_t value) {
            return pair.first < value;
        });
    if (it != unicode_map_lowercase.end() && it->first == cpt) {
        return it->second;
    }
    return cpt;  // Return the original code point if no lowercase mapping is found
}

static inline size_t encode_cpt_to_utf8_buf(uint32_t cpt, char buf[4]) {
    if (cpt <= 0x7f) {
        buf[0] = static_cast<char>(cpt);
        return 1;
    } else if (cpt <= 0x7ff) {
        buf[0] = static_cast<char>(0xc0 | ((cpt >> 6) & 0x1f));
        buf[1] = static_cast<char>(0x80 | (cpt & 0x3f));
        return 2;
    } else if (cpt <= 0xffff) {
        buf[0] = static_cast<char>(0xe0 | ((cpt >> 12) & 0x0f));
        buf[1] = static_cast<char>(0x80 | ((cpt >> 6) & 0x3f));
        buf[2] = static_cast<char>(0x80 | (cpt & 0x3f));
        return 3;
    } else if (cpt <= 0x10ffff) {
        buf[0] = static_cast<char>(0xf0 | ((cpt >> 18) & 0x07));
        buf[1] = static_cast<char>(0x80 | ((cpt >> 12) & 0x3f));
        buf[2] = static_cast<char>(0x80 | ((cpt >> 6) & 0x3f));
        buf[3] = static_cast<char>(0x80 | (cpt & 0x3f));
        return 4;
    }

    // 遇到非法 codepoint 不走异常路径，直接写入 U+FFFD，避免 tokenizer 崩溃和异常分配。
    buf[0] = static_cast<char>(0xef);
    buf[1] = static_cast<char>(0xbf);
    buf[2] = static_cast<char>(0xbd);
    return 3;
}

static inline void append_cpt_to_utf8(std::string & result, uint32_t cpt) {
    char buf[4];
    const size_t n = encode_cpt_to_utf8_buf(cpt, buf);
    result.append(buf, n);
}

static inline void append_cpt_to_byte_encoded_utf8(std::string & out, uint32_t cpt) {
    char buf[4];
    const size_t n = encode_cpt_to_utf8_buf(cpt, buf);
    for (size_t i = 0; i < n; ++i) {
        append_byte_to_utf8(out, static_cast<uint8_t>(buf[i]));
    }
}

void unicode_regex_split_into(
        const std::string & text,
        const std::vector<std::string> & regex_exprs,
        bool byte_encode,
        regex_split_result & result,
        std::vector<uint32_t> & cpts_buf,
        std::vector<size_t> & offsets_buf,
        std::vector<size_t> & tmp_offsets_buf) {
    // unicode categories
    static const std::map<std::string, int> k_ucat_enum = {
        { "\\p{N}", unicode_cpt_flags::NUMBER },
        { "\\p{L}", unicode_cpt_flags::LETTER },
        { "\\p{P}", unicode_cpt_flags::PUNCTUATION },
        { "\\p{M}", unicode_cpt_flags::ACCENT_MARK },
        { "\\p{S}", unicode_cpt_flags::SYMBOL },
        { "\\p{Lu}", unicode_cpt_flags::LETTER }, // Uppercase letter
        { "\\p{Ll}", unicode_cpt_flags::LETTER }, // Lowercase letter
        { "\\p{Lt}", unicode_cpt_flags::LETTER }, // Titlecase letter
        { "\\p{Lm}", unicode_cpt_flags::LETTER }, // Modifier letter
        { "\\p{Lo}", unicode_cpt_flags::LETTER }, // Other letter
    };

    static const std::map<int, int> k_ucat_cpt = {
        { unicode_cpt_flags::NUMBER,      0xD1 },
        { unicode_cpt_flags::LETTER,      0xD2 },
        { unicode_cpt_flags::PUNCTUATION, 0xD3 },
        { unicode_cpt_flags::ACCENT_MARK, 0xD4 },
        { unicode_cpt_flags::SYMBOL,      0xD5 },
    };

    static const std::map<int, std::string> k_ucat_map = {
        { unicode_cpt_flags::NUMBER,      "\x30-\x39" }, // 0-9
        { unicode_cpt_flags::LETTER,      "\x41-\x5A\x61-\x7A" }, // A-Za-z
        { unicode_cpt_flags::PUNCTUATION, "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D" }, // !-#%-*,-/:-;?-@\[-\]_\{\}
        { unicode_cpt_flags::ACCENT_MARK, "" }, // no sub-128 codepoints
        { unicode_cpt_flags::SYMBOL,      "\\\x24\\\x2B\x3C-\x3E\x5E\x60\\\x7C" }, // $+<=>^`|
    };

    // compute collapsed codepoints only if needed by at least one regex
    bool need_collapse = false;
    for (const auto & regex_expr : regex_exprs) {
        // search for unicode categories
        for (const auto & ucat : k_ucat_enum) {
            if (std::string::npos != regex_expr.find(ucat.first)) {
                need_collapse = true;
                break;
            }
        }
    }

    unicode_cpts_from_utf8_into(text, cpts_buf);
    const std::vector<uint32_t> & cpts = cpts_buf;

    // generate a "collapsed" representation of the text, where all codepoints are replaced by a single byte
    // ref: https://github.com/ggml-org/llama.cpp/pull/6920#issuecomment-2081479935
    // 高频 tokenizer 路径中复用 string 容量，避免每次 resize 都重新申请/释放堆块。
    static thread_local std::string text_collapsed;
    text_collapsed.clear();
    if (need_collapse) {
        // collapse all unicode categories
        if (text_collapsed.capacity() < cpts.size()) {
            text_collapsed.reserve(cpts.size() + 32);
        }
        text_collapsed.resize(cpts.size());

        for (size_t i = 0; i < cpts.size(); ++i) {
            // keep single-byte codepoints as is
            if (cpts[i] < 128) {
                text_collapsed[i] = cpts[i];
                continue;
            }

            const auto flags = unicode_cpt_flags_from_cpt(cpts[i]);

            if (flags.is_whitespace) {
                //NOTE: C++ std::regex \s does not mach 0x85, Rust and Python regex does.
                //text_collapsed[i] = (char) 0x85;  // <Next Line> as whitespace fallback
                text_collapsed[i] = (char) 0x0B;    // <vertical tab> as whitespace fallback
            } else if (k_ucat_cpt.find(flags.category_flag()) != k_ucat_cpt.end()) {
                text_collapsed[i] = k_ucat_cpt.at(flags.category_flag());
            } else {
                text_collapsed[i] = (char) 0xD0; // fallback
            }
        }
    }

    offsets_buf.clear();
    offsets_buf.push_back(cpts.size());
    std::vector<size_t> & bpe_offsets = offsets_buf;
    std::vector<size_t> & tmp_offsets = tmp_offsets_buf;

    for (const auto & regex_expr : regex_exprs) {
        // first, see if we have an efficient custom regex implementation
        tmp_offsets.clear();
        if (unicode_regex_split_custom(text, cpts, regex_expr, bpe_offsets, tmp_offsets)) {
            bpe_offsets.swap(tmp_offsets);
            continue;
        }

        // fallback to general-purpose std::regex / std::wregex
        try {
            // if a unicode category is used in the regex, we use the collapsed text and replace the unicode category
            // with the corresponding collapsed representation
            bool use_collapsed = false;
            for (const auto & ucat : k_ucat_enum) {
                if (std::string::npos != regex_expr.find(ucat.first)) {
                    use_collapsed = true;
                    break;
                }
            }

            if (use_collapsed) {
                // sanity-check that the original regex does not contain any non-ASCII characters.
                // 这里只需要判断 ASCII，不需要再把 regex_expr 转成 codepoints，避免额外 vector 分配。
                for (size_t i = 0; i < regex_expr.size(); ++i) {
                    if (static_cast<unsigned char>(regex_expr[i]) >= 128) {
                        throw_runtime_error("Regex includes both unicode categories and non-ASCII characters - not supported");
                    }
                }

                // generate a collapsed representation of the regex. 复用容量，避免热路径反复
                // 构造/销毁 std::string 小堆块。
                static thread_local std::string regex_expr_collapsed;
                regex_expr_collapsed.clear();
                if (regex_expr_collapsed.capacity() < regex_expr.size() + 16) {
                    regex_expr_collapsed.reserve(regex_expr.size() + 16);
                }

                // track if we are inside [], because nested [] are not allowed
                bool inside = false;
                for (size_t i = 0; i < regex_expr.size(); ++i) {
                    if (regex_expr[i] == '[' && (i == 0 || regex_expr[i - 1] != '\\')) {
                        regex_expr_collapsed += '[';
                        inside = true;
                        continue;
                    }

                    if (inside && regex_expr[i] == ']' && regex_expr[i - 1] != '\\') {
                        regex_expr_collapsed += ']';
                        inside = false;
                        continue;
                    }

                    // Match \p{...} Unicode properties of varying lengths
                    if (regex_expr[i + 0] == '\\' && i + 3 < regex_expr.size() &&
                        regex_expr[i + 1] == 'p' &&
                        regex_expr[i + 2] == '{') {
                        // Find the closing brace
                        size_t closing_brace = regex_expr.find('}', i + 3);
                        if (closing_brace != std::string::npos && closing_brace <= i + 10) { // reasonable limit
                            const size_t pat_len = closing_brace - i + 1;
                            std::map<std::string, int>::const_iterator ucat_it = k_ucat_enum.end();
                            for (std::map<std::string, int>::const_iterator it = k_ucat_enum.begin(); it != k_ucat_enum.end(); ++it) {
                                if (it->first.size() == pat_len && regex_expr.compare(i, pat_len, it->first) == 0) {
                                    ucat_it = it;
                                    break;
                                }
                            }
                            if (ucat_it != k_ucat_enum.end()) {
                                if (!inside) {
                                    regex_expr_collapsed += '[';
                                }
                                regex_expr_collapsed += k_ucat_cpt.at(ucat_it->second);
                                regex_expr_collapsed += k_ucat_map.at(ucat_it->second);
                                if (!inside) {
                                    regex_expr_collapsed += ']';
                                }
                                i = closing_brace;
                                continue;
                            }
                        }
                    }

                    regex_expr_collapsed += regex_expr[i];
                }

                //printf("text_collapsed: %s\n", text_collapsed.c_str());
                //printf("regex_expr_collapsed: %s\n", regex_expr_collapsed.c_str());
                unicode_regex_split_stl(text_collapsed, regex_expr_collapsed, bpe_offsets, tmp_offsets);
                bpe_offsets.swap(tmp_offsets);
            } else {
                // no unicode category used, we can use std::wregex directly
                static thread_local std::wstring wregex_expr;
                static thread_local std::wstring wtext;

                wregex_expr = unicode_wstring_from_utf8(regex_expr);

                // std::wregex \s does not mach non-ASCII whitespaces, using 0x0B as fallback
                wtext.clear();
                if (wtext.capacity() < cpts.size()) {
                    wtext.reserve(cpts.size() + 32);
                }
                wtext.assign(cpts.begin(), cpts.end());
                for (size_t i = 0; i < wtext.size(); ++i) {
                    if (wtext[i] > 0x7F && unicode_cpt_flags_from_cpt(wtext[i]).is_whitespace) {
                        wtext[i] = 0x0B;
                    }
                }

                //printf("text: %s\n", text.c_str());
                //printf("regex_expr: %s\n", regex_expr.c_str());
                unicode_regex_split_stl(wtext, wregex_expr, bpe_offsets, tmp_offsets);
                bpe_offsets.swap(tmp_offsets);
            }
        } catch (std::regex_error & e) {
            fprintf(stderr, "Failed to process regex: '%s'\n", regex_expr.c_str());
            fprintf(stderr, "Regex error: %s\n", e.what());
            throw_runtime_error("Failed to process regex");
        }
    }

    result.buffer.clear();
    result.word_lengths.clear();

    // 单体 buffer + offsets：避免 vector<string> 中每个 word 各自分配/释放小堆块。
    // byte_encode 后 GPT-2 byte 映射通常最多膨胀到约 2 倍；这里仅在容量不足时扩容，容量由 session 复用。
    const size_t expected_buffer_size = text.size() * (byte_encode ? 2 : 1);
    if (result.buffer.capacity() < expected_buffer_size) {
        result.buffer.reserve(expected_buffer_size);
    }
    if (result.word_lengths.capacity() < bpe_offsets.size()) {
        result.word_lengths.reserve(bpe_offsets.size());
    }

    size_t start = 0;
    for (size_t k = 0; k < bpe_offsets.size(); ++k) {
        const size_t cp_len = bpe_offsets[k];
        const size_t end = start + cp_len;
        const size_t word_begin = result.buffer.size();

        if (byte_encode) {
            // 直接内联 unicode_byte_encoding_process：不再生成中间 vector<string> 和 per-word 临时 string。
            for (size_t i = start; i < end; ++i) {
                append_cpt_to_byte_encoded_utf8(result.buffer, cpts[i]);
            }
        } else {
            for (size_t i = start; i < end; ++i) {
                append_cpt_to_utf8(result.buffer, cpts[i]);
            }
        }

        result.word_lengths.push_back(result.buffer.size() - word_begin);
        start = end;
    }

}

regex_split_result unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs, bool byte_encode) {
    regex_split_result result;
    std::vector<uint32_t> cpts_buf;
    std::vector<size_t> offsets_buf;
    std::vector<size_t> tmp_offsets_buf;
    unicode_regex_split_into(text, regex_exprs, byte_encode, result, cpts_buf, offsets_buf, tmp_offsets_buf);
    return result;
}