// Cygwin stubs for absl::synchronization primitives that don't compile
// on Cygwin (pthread_t is a pointer, no futex, etc).
// The tokenizer is single-threaded so no-op implementations are safe.

#include <cstdlib>

#include "absl/base/internal/low_level_alloc.h"
#include "absl/synchronization/internal/create_thread_identity.h"
#include "absl/synchronization/internal/per_thread_sem.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace synchronization_internal {

static base_internal::ThreadIdentity g_tid_stub{};
base_internal::ThreadIdentity* CreateThreadIdentity() {
  return &g_tid_stub;
}

}  // namespace synchronization_internal

namespace base_internal {

void* LowLevelAlloc::Alloc(size_t n) { return ::malloc(n); }
void LowLevelAlloc::Free(void* p)  { ::free(p); }

}  // namespace base_internal
ABSL_NAMESPACE_END
}  // namespace absl

// PerThreadSem stubs — declared with ABSL_ATTRIBUTE_WEAK in per_thread_sem.cc
extern "C" {

void AbslInternalPerThreadSemPost(
    absl::base_internal::ThreadIdentity*) {}

bool AbslInternalPerThreadSemWait(
    absl::synchronization_internal::KernelTimeout) {
  return false;  // never timeout
}

}  // extern "C"
