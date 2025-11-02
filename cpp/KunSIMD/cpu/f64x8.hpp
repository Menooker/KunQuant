#pragma once

#if defined(__AVX__)
#include "x86/f64x8.hpp"
#else
#include "cpu/neon/f64x8.hpp"
#endif
