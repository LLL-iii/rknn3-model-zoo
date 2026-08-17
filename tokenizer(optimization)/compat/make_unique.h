// Provides std::make_unique for C++11 builds (it was added in C++14).
// Force-included via -include in C++11 mode; harmless no-op in C++17 mode.
#ifndef TOKENIZER_COMPAT_MAKE_UNIQUE_H_
#define TOKENIZER_COMPAT_MAKE_UNIQUE_H_

#include <cstddef>
#include <memory>
#include <utility>

#if __cplusplus < 201402L
namespace std {

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename T>
std::unique_ptr<T> make_unique(std::size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

} // namespace std
#endif // __cplusplus < 201402L

#endif // TOKENIZER_COMPAT_MAKE_UNIQUE_H_
