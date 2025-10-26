#pragma once

#if defined(__SSE__)
#include "x86/f32x16.hpp"
#else
#include "cpu/neon/f32x16.hpp"
#endif
