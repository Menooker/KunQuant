#pragma once
//===- Aligned.hpp - cross-platform aligned-allocator macros ----------===//
//
// Centralises the `kunAlignedAlloc(alignment, size)` /
// `kunAlignedFree(ptr)` macros so PyBinding.cpp and Runtime.cpp share
// one definition.  Runtime.cpp may *override* these for the
// `CHECKED_PTR` mode (page-guarded debug allocator) — keep this
// header `#include`d **before** any such override.
//
// Use `KUN_MALLOC_ALIGNMENT` from `Kun/StateBuffer.hpp` for the
// standard buffer alignment (64 bytes on x86 AVX-512 builds, 16 on
// ARM/NEON).
//
//===---------------------------------------------------------------------===//

#include <cstdlib>

#ifdef _WIN32
#include <malloc.h>
#define kunAlignedAlloc(alignment, size) _aligned_malloc((size), (alignment))
#define kunAlignedFree(ptr)              _aligned_free(ptr)
#else
// POSIX `aligned_alloc(alignment, size)` requires size to be a
// multiple of alignment.  Callers (Buffer::alloc, etc.) pass
// `count * elem_size` which is already a multiple of
// KUN_MALLOC_ALIGNMENT in practice, so the macro is a thin pass-through.
#define kunAlignedAlloc(alignment, size) aligned_alloc((alignment), (size))
#define kunAlignedFree(ptr)              free(ptr)
#endif
