//===- RuntimeUtil.h - private kun_cuda runtime helpers ------------------===//
//
// This header is private to libKunCudaRuntime.  It holds the pieces shared by
// the traditional sequential launcher and the CUDA Graph launcher.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "KunCuda/Runtime.h"

#include <cuda.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace kun_cuda {

// Runtime-resolved schedule + memory plan.  The public header forward-declares
// this type so Executable can keep it behind a pImpl.
struct GraphPlan {
  int numBuffers       = 0;
  int numGraphInputs   = 0;
  int numGraphOutputs  = 0;

  std::unordered_map<std::string, int> graphInputIdx;
  std::unordered_map<std::string, int> graphOutputIdx;

  std::vector<std::vector<int>> kernelInputBufs;
  std::vector<std::vector<int>> kernelOutputBufs;

  // producerKernel[bufIdx] = kernel that writes the buffer, or -1 for a graph
  // input.
  std::vector<int> producerKernel;

  // Topo order used by the sequential launcher and as a construction order for
  // CUDA Graph kernel nodes.
  std::vector<int> launchOrder;

  // Sequential-mode intermediate slot assignment.  Graph mode uses one
  // allocation node per logical intermediate instead.
  std::vector<int> intermediateBufToSlot;
  int peakIntermediateSlots = 0;
};

struct ChunkPlan {
  int64_t chunkSize = 0;
  unsigned numChunks = 1;
};

struct CudaGraphLaunchParams {
  int64_t timeLength = 0;
  int64_t numStocks = 0;
  int64_t mask = 0;
  int minChunkWarmupFactor = 0;
  double smFillFactor = 0.0;
  int devMaxSmemBytes = 0;
  int numSMs = 0;
  std::vector<uintptr_t> bufPtrs;
};

struct KernelLaunchDesc {
  KernelLaunchDesc() = default;
  KernelLaunchDesc(int kernelIndex, KernelKind kind, CUfunction fn,
                   bool isKernelNode,
                   int32_t timeLenI32, int32_t numStocksI32,
                   int32_t maskI32, int32_t chunkSizeI32,
                   int32_t warmupI32,
                   std::vector<CUdeviceptr> ptrs);
  KernelLaunchDesc(KernelLaunchDesc &&other) noexcept;
  KernelLaunchDesc &operator=(KernelLaunchDesc &&other) noexcept;
  KernelLaunchDesc(const KernelLaunchDesc &) = delete;
  KernelLaunchDesc &operator=(const KernelLaunchDesc &) = delete;

  void update(const GraphPlan &plan,
              const ExecutableData &data,
              const std::vector<CUfunction> &cuFuncs,
              int kernelIndex,
              const CudaGraphLaunchParams &launch);
  bool updateBuffer(const GraphPlan &plan,
                    int kernelIndex,
                    const std::vector<uintptr_t> &bufPtrs);

  int kernelIndex = -1;
  KernelKind kind = KernelKind::Jit;
  bool isKernelNode = true;

  int32_t timeLenI32 = 0;
  int32_t numStocksI32 = 0;
  int32_t maskI32 = 0;
  int32_t chunkSizeI32 = 0;
  int32_t warmupI32 = 0;

  CUDA_KERNEL_NODE_PARAMS params{};

private:
  std::vector<CUdeviceptr> ptrs_;
  std::vector<void *> argPtrs_;

  void rebuildKernelParamPointers();
};

struct CudaGraphLaunchState {
  ~CudaGraphLaunchState() noexcept;

  CUgraph graph = nullptr;
  CUgraphExec graphExec = nullptr;
  CUevent completionEvent = nullptr;
  bool hasLaunch = false;

  std::optional<CudaGraphLaunchParams> cachedLaunchParams;

  std::vector<uintptr_t> graphAllocBufPtrs;
  std::vector<CUgraphNode> allocNodes;
  std::vector<CUgraphNode> kernelNodes;
  std::vector<bool> kernelNodeIsKernel;
  std::vector<CUgraphNode> freeNodes;
  std::vector<KernelLaunchDesc> descs;
};

void checkCu(CUresult r, const char *what);

std::string joinNames(const std::vector<std::string> &v);

int firstIntermediateBuffer(const GraphPlan &plan) noexcept;

void validateLaunchInputs(const ExecutableData &data,
                          int64_t timeLength, int64_t numStocks,
                          int64_t mask);

std::vector<uintptr_t> resolveExternalBufferPointers(
    const GraphPlan &plan,
    const ExecutableData &data,
    const std::vector<std::pair<std::string, uintptr_t>> &args);

std::vector<uintptr_t> resolveBufferPointers(
    const GraphPlan &plan,
    const ExecutableData &data,
    const std::vector<std::pair<std::string, uintptr_t>> &args,
    const std::vector<uintptr_t> &slotBufs);

ChunkPlan computeChunkPlan(int64_t timeLength, int64_t numStocks,
                           int64_t warpsPerCta, int64_t vectorSize,
                           int64_t unreliableCount, int64_t mask,
                           int minChunkWarmupFactor,
                           double smFillFactor, int numSMs);

void launchKernelDesc(const KernelLaunchDesc &desc, CUstream stream);

} // namespace kun_cuda
