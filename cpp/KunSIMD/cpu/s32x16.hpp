#pragma once

#if defined(__AVX__)
#include "x86/s32x16.hpp"
#else
#include "cpu/neon/s32x16.hpp"
#endif
