//===- KunGpuMemoryPlanning.cpp - Windowed-temp shared/local memory plan --===//
//
// Assigns each kungpu.windowed_temp op a "use shared memory" flag stored as
// the discardable attribute "kungpu.smem" (BoolAttr).  The pass itself does
// not mutate IR structure; the subsequent to-LLVM lowering consults the attr
// to pick an address space.
//
// Strategy: sort windowed_temp ops by ascending window size (smaller buffers
// fit more easily into shared memory) and greedily assign shared memory until
// the per-block budget is exhausted.
//
// Budget (from the enclosing kunir.func target_spec):
//   budget_per_block = target_spec.smem_size / target_spec.occupancy
//   num_threads      = target_spec.warps_per_cta * 32
//   bytes_per_buf    = N * num_threads * target_spec.vector_size * elem_bytes
//
//===----------------------------------------------------------------------===//

// MLIR and local headers must come before GEN_PASS_DEF_* so that ::kunir
// is fully declared when Passes.h.inc is expanded.
#include "KunGpu/KunGpuOps.h"
#include "KunGpu/Passes.h"
#include "KunIr/KunIrAttrs.h"
#include "KunIr/KunIrOps.h"
#include "KunIr/KunIrTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <limits>

// Pull in the generated PassBase scaffolding after all declarations are in scope.
#define GEN_PASS_DEF_WINDOWEDTEMPMEMORYPLANNING
#include "KunGpu/Passes.h.inc"

#define DEBUG_TYPE "kungpu-memory-planning"

using namespace mlir;
using namespace kunir;
using namespace kungpu;

namespace {

//===----------------------------------------------------------------------===//
// Helper: byte width of a floating-point element type
//===----------------------------------------------------------------------===//

static unsigned elemBytes(Type t) {
  if (auto ft = dyn_cast<FloatType>(t))
    return (ft.getWidth() + 7) / 8;
  return 4; // conservative fallback
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct WindowedTempMemoryPlanningPass
    : ::impl::WindowedTempMemoryPlanningBase<WindowedTempMemoryPlanningPass> {

  void runOnOperation() override {
    kunir::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // -----------------------------------------------------------------------
    // 1. Read hardware parameters from target_spec.
    //    smem_size is the total SM shared memory; divide by occupancy to get
    //    the per-block budget.
    // -----------------------------------------------------------------------
    auto ts = funcOp.getTargetSpec();
    int64_t occupancy       = ts.getOccupancy();
    int64_t budgetPerBlock  = (occupancy > 0) ? (ts.getSmemSize() / occupancy) : 0;
    int64_t numThreads      = ts.getWarpsPerCta() * 32;
    int64_t vectorSize      = ts.getVectorSize();

    // -----------------------------------------------------------------------
    // 2. Collect all windowed_temp ops in the function.
    // -----------------------------------------------------------------------
    SmallVector<WindowedTempOp> temps;
    funcOp.walk([&](WindowedTempOp op) { temps.push_back(op); });

    if (temps.empty())
      return;

    // -----------------------------------------------------------------------
    // 3. Sort by ascending window size (smaller N → higher smem priority).
    // -----------------------------------------------------------------------
    std::stable_sort(temps.begin(), temps.end(),
                     [](WindowedTempOp a, WindowedTempOp b) {
                       return llvm::cast<TsType>(a.getType()).getMaxLookback() <
                              llvm::cast<TsType>(b.getType()).getMaxLookback();
                     });

    // -----------------------------------------------------------------------
    // 4. Greedy assignment: place in shared memory while budget allows.
    // -----------------------------------------------------------------------
    int64_t usedSmem = 0;

    for (WindowedTempOp op : temps) {
      auto tsTy = llvm::cast<TsType>(op.getType());
      uint64_t N = tsTy.getMaxLookback();

      // Infinite-lookback buffers cannot be sized statically → always local.
      if (N == std::numeric_limits<uint64_t>::max()) {
        op->setAttr("kungpu.smem", BoolAttr::get(ctx, false));
        continue;
      }

      int64_t bytes = static_cast<int64_t>(N) * numThreads *
                      vectorSize * elemBytes(tsTy.getElementType());

      bool useSmem =
          (budgetPerBlock > 0) && (usedSmem + bytes <= budgetPerBlock);
      if (useSmem)
        usedSmem += bytes;

      op->setAttr("kungpu.smem", BoolAttr::get(ctx, useSmem));

      LLVM_DEBUG(llvm::dbgs()
                 << "[kungpu-memory-planning] windowed_temp N=" << N
                 << " bytes=" << bytes << " -> "
                 << (useSmem ? "smem" : "local") << "\n");
    }

    LLVM_DEBUG(llvm::dbgs() << "[kungpu-memory-planning] total smem used="
                            << usedSmem << " / budget=" << budgetPerBlock
                            << "\n");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public factory function
//===----------------------------------------------------------------------===//

namespace kungpu {

std::unique_ptr<mlir::Pass> createWindowedTempMemoryPlanningPass() {
  return std::make_unique<WindowedTempMemoryPlanningPass>();
}

} // namespace kungpu
