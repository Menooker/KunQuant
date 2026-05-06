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
// Budget:
//   smem_per_block  = total_smem_size / target_occupancy
//   bytes_per_buf   = N * num_threads_per_block * vector_size * elem_bytes
//
// Parameters are passed as plain integers (not IR attributes) because the
// hardware/launch-config attributes are not yet wired into the IR.
//
//===----------------------------------------------------------------------===//

// MLIR and local headers must come before GEN_PASS_DEF_* so that ::mlir
// and func::FuncOp are fully declared when Passes.h.inc is expanded.
#include "KunGpu/KunGpuOps.h"
#include "KunGpu/Passes.h"
#include "KunIr/KunIrTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

  // Parametric constructor — used by createWindowedTempMemoryPlanningPass().
  WindowedTempMemoryPlanningPass(int64_t totalSmemSize, int64_t targetOccupancy,
                                 int64_t numThreadsPerBlock, int64_t vectorSize)
      : totalSmemSize(totalSmemSize), targetOccupancy(targetOccupancy),
        numThreadsPerBlock(numThreadsPerBlock), vectorSize(vectorSize) {}

  // Default constructor — used by the pipeline registration factory.
  WindowedTempMemoryPlanningPass()
      : WindowedTempMemoryPlanningPass(
            /*totalSmemSize=*/49152, // 48 KB (typical Ampere)
            /*targetOccupancy=*/1,
            /*numThreadsPerBlock=*/32,
            /*vectorSize=*/1) {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // -----------------------------------------------------------------------
    // 1. Collect all windowed_temp ops in the function.
    // -----------------------------------------------------------------------
    SmallVector<WindowedTempOp> temps;
    funcOp.walk([&](WindowedTempOp op) { temps.push_back(op); });

    if (temps.empty())
      return;

    // -----------------------------------------------------------------------
    // 2. Compute per-block shared memory budget (bytes).
    // -----------------------------------------------------------------------
    int64_t budgetPerBlock =
        (targetOccupancy > 0) ? (totalSmemSize / targetOccupancy) : 0;

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

      int64_t bytes = static_cast<int64_t>(N) * numThreadsPerBlock *
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

  int64_t totalSmemSize;
  int64_t targetOccupancy;
  int64_t numThreadsPerBlock;
  int64_t vectorSize;
};

} // namespace

//===----------------------------------------------------------------------===//
// Public factory functions
//===----------------------------------------------------------------------===//

namespace kungpu {

std::unique_ptr<mlir::Pass> createWindowedTempMemoryPlanningPass() {
  return std::make_unique<WindowedTempMemoryPlanningPass>();
}

std::unique_ptr<mlir::Pass>
createWindowedTempMemoryPlanningPass(int64_t totalSmemSize,
                                     int64_t targetOccupancy,
                                     int64_t numThreadsPerBlock,
                                     int64_t vectorSize) {
  return std::make_unique<WindowedTempMemoryPlanningPass>(
      totalSmemSize, targetOccupancy, numThreadsPerBlock, vectorSize);
}

} // namespace kungpu
