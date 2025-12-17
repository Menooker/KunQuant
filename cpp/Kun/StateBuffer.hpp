#pragma once

#include "Base.hpp"
#include "MathUtil.hpp"
#include <cstdint>
#include <memory>

namespace kun {

#ifdef __AVX__
#define KUN_MALLOC_ALIGNMENT 64 // AVX-512 alignment
#else
#define KUN_MALLOC_ALIGNMENT 16 // NEON alignment
#endif

struct StateBuffer {
    using DtorFn_t = void (*)(StateBuffer *obj);
    using CtorFn_t = void (*)(StateBuffer *obj);

    alignas(KUN_MALLOC_ALIGNMENT) size_t num_objs;
    uint32_t elem_size;
    uint32_t initialized;
    CtorFn_t ctor_fn;
    DtorFn_t dtor_fn;
    alignas(KUN_MALLOC_ALIGNMENT) char buf[0];

    KUN_API static StateBuffer *make(size_t num_objs, size_t elem_size,
                             CtorFn_t ctor_fn, DtorFn_t dtor_fn);

    // for std::unique_ptr
    struct Deleter {
        KUN_API void operator()(StateBuffer *buf);
    };

    template <typename T>
    T &get(size_t idx) {
        return *reinterpret_cast<T *>(buf + idx * sizeof(T));
    }

    void initialize() {
        initialized = 1;
        ctor_fn(this);
    }
    void destroy() {
        if (initialized) {
            dtor_fn(this);
        }
        initialized = 0;
    }

  private:
    StateBuffer() = default;
};

using StateBufferPtr = std::unique_ptr<StateBuffer, StateBuffer::Deleter>;

template <typename T>
StateBufferPtr makeStateBuffer(size_t num_stocks, size_t simd_len) {
    return StateBufferPtr(StateBuffer::make(
        divideAndCeil(num_stocks, simd_len), sizeof(T),
        [](StateBuffer *obj) {
            for (size_t i = 0; i < obj->num_objs; i++) {
                new (&obj->get<T>(i)) T();
            }
        },
        [](StateBuffer *obj) {
            for (size_t i = 0; i < obj->num_objs; i++) {
                obj->get<T>(i).~T();
            }
        }));
}
} // namespace kun