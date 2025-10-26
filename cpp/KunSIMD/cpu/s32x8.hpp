#pragma once

#if defined(__SSE__)
#include "x86/s32x8.hpp"
#else
#include "cpu/neon/s32x8.hpp"
#endif
