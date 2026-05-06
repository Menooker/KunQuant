#pragma once

#include "mlir/Pass/Pass.h"
#include <cstdint>
#include <memory>

namespace kungpu {

#define GEN_PASS_DECL
#include "KunGpu/Passes.h.inc"

// Default-args factory used by the pipeline registration and kun-opt.
std::unique_ptr<mlir::Pass> createWindowedTempMemoryPlanningPass();

// Parametric factory for use by callers that provide hardware config.
std::unique_ptr<mlir::Pass>
createWindowedTempMemoryPlanningPass(int64_t totalSmemSize,
                                     int64_t targetOccupancy,
                                     int64_t numThreadsPerBlock,
                                     int64_t vectorSize);

#define GEN_PASS_REGISTRATION
#include "KunGpu/Passes.h.inc"

} // namespace kungpu
