#pragma once

#if defined(__AVX__)
#include "x86/s64x8.hpp"
#else
#include "cpu/neon/s64x8.hpp"
#endif
