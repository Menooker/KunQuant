//===- RuntimeCudaGraph.cpp - CUDA Graph launcher for kun_cuda ------------===//
//
// CUDA Graph mode builds the real producer/consumer node DAG instead of
// enqueueing kernels in a topo-linear loop.  Intermediate buffers are
// graph-owned allocations:
//
//   alloc(intermediate) -> producer kernel -> all consumer kernels -> free
//
// User-visible graph inputs/outputs remain caller-owned pointers supplied at
// launch time.
//
//===----------------------------------------------------------------------===//

#include "KunCuda/Runtime.h"
#include "RuntimeUtil.h"

#include <cuda.h>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kun_cuda {

namespace {

void addUniqueNode(std::vector<CUgraphNode> &nodes, CUgraphNode n) {
  if (!n)
    return;
  if (std::find(nodes.begin(), nodes.end(), n) == nodes.end())
    nodes.push_back(n);
}

bool isIntermediate(const GraphPlan &plan, int bufIdx) {
  return bufIdx >= firstIntermediateBuffer(plan);
}

void ensureNoInFlight(CudaGraphLaunchState &state, const char *action) {
  if (!state.hasLaunch || !state.completionEvent)
    return;

  CUresult r = cuEventQuery(state.completionEvent);
  if (r == CUDA_SUCCESS) {
    state.hasLaunch = false;
    return;
  }
  if (r == CUDA_ERROR_NOT_READY)
    throw std::runtime_error(
        std::string("kun_cuda::launchOnStream(cuda_graph): previous CUDA "
                    "graph launch is still in flight; call synchronize() "
                    "before ") + action + " the executable's CUDA graph");
  checkCu(r, "cuEventQuery(cuda graph completion)");
}

size_t intermediateBytes(const ExecutableData &data,
                         int64_t timeLength, int64_t numStocks) {
  size_t bytes = static_cast<size_t>(timeLength) *
                 static_cast<size_t>(numStocks) *
                 bytesPerElem(data.dtype);
  return bytes == 0 ? 1 : bytes;
}

bool sameLaunchParamsExceptBuffers(const CudaGraphLaunchParams &a,
                                   const CudaGraphLaunchParams &b) {
  return a.timeLength == b.timeLength &&
         a.numStocks == b.numStocks &&
         a.mask == b.mask &&
         a.minChunkWarmupFactor == b.minChunkWarmupFactor &&
         a.smFillFactor == b.smFillFactor &&
         a.devMaxSmemBytes == b.devMaxSmemBytes &&
         a.numSMs == b.numSMs;
}

bool sameLaunchParams(const CudaGraphLaunchParams &a,
                      const CudaGraphLaunchParams &b) {
  return sameLaunchParamsExceptBuffers(a, b) && a.bufPtrs == b.bufPtrs;
}

CudaGraphLaunchParams makeLaunchParams(
    Executor *exec,
    int64_t timeLength, int64_t numStocks,
    int64_t mask,
    int minChunkWarmupFactor,
    double smFillFactor,
    std::vector<uintptr_t> bufPtrs) {
  CudaGraphLaunchParams params;
  params.timeLength = timeLength;
  params.numStocks = numStocks;
  params.mask = mask;
  params.minChunkWarmupFactor = minChunkWarmupFactor;
  params.smFillFactor = smFillFactor;
  params.devMaxSmemBytes = exec->devMaxSmemBytes();
  params.numSMs = exec->numSMs();
  params.bufPtrs = std::move(bufPtrs);
  return params;
}

std::vector<uintptr_t> resolveCudaGraphBufferPointers(
    const GraphPlan &plan,
    const ExecutableData &data,
    const CudaGraphLaunchState &state,
    const std::vector<std::pair<std::string, uintptr_t>> &args) {
  std::vector<uintptr_t> bufPtrs =
      resolveExternalBufferPointers(plan, data, args);
  for (int b = firstIntermediateBuffer(plan); b < plan.numBuffers; ++b)
    bufPtrs[b] = state.graphAllocBufPtrs[b];
  return bufPtrs;
}

CudaGraphLaunchParams makeLaunchParams(
    const GraphPlan &plan,
    const ExecutableData &data,
    const CudaGraphLaunchState &state,
    Executor *exec,
    int64_t timeLength, int64_t numStocks,
    const std::vector<std::pair<std::string, uintptr_t>> &args,
    int64_t mask,
    int minChunkWarmupFactor,
    double smFillFactor) {
  return makeLaunchParams(
      exec, timeLength, numStocks, mask, minChunkWarmupFactor, smFillFactor,
      resolveCudaGraphBufferPointers(plan, data, state, args));
}

CUgraphNode addAllocNode(CUgraph graph,
                         CUdevice device,
                         size_t bytes,
                         const std::vector<CUgraphNode> &deps,
                         CUdeviceptr &outPtr) {
  CUDA_MEM_ALLOC_NODE_PARAMS params{};
  params.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
  params.poolProps.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
  params.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  params.poolProps.location.id = device;
  params.bytesize = bytes;

  CUgraphNode node = nullptr;
  checkCu(cuGraphAddMemAllocNode(&node, graph,
                                 deps.empty() ? nullptr : deps.data(),
                                 deps.size(), &params),
          "cuGraphAddMemAllocNode");
  outPtr = params.dptr;
  return node;
}

std::vector<CUgraphNode> kernelInputDeps(const GraphPlan &plan,
                                         const CudaGraphLaunchState &state,
                                         int kIdx) {
  std::vector<CUgraphNode> deps;
  for (int b : plan.kernelInputBufs[kIdx]) {
    int producer = plan.producerKernel[b];
    if (producer >= 0)
      addUniqueNode(deps, state.kernelNodes[producer]);
  }
  return deps;
}

void addOutputAllocNodes(const GraphPlan &plan,
                         CudaGraphLaunchState &state,
                         CUdevice device,
                         size_t bytes,
                         int kIdx,
                         const std::vector<CUgraphNode> &inputDeps,
                         std::vector<uintptr_t> &bufPtrs) {
  for (int b : plan.kernelOutputBufs[kIdx]) {
    if (!isIntermediate(plan, b))
      continue;
    CUdeviceptr dptr = 0;
    CUgraphNode allocNode =
        addAllocNode(state.graph, device, bytes, inputDeps, dptr);
    state.allocNodes[b] = allocNode;
    state.graphAllocBufPtrs[b] = static_cast<uintptr_t>(dptr);
    bufPtrs[b] = static_cast<uintptr_t>(dptr);
  }
}

void addOneKernelNode(const GraphPlan &plan,
                      const ExecutableData &data,
                      const std::vector<CUfunction> &cuFuncs,
                      CudaGraphLaunchState &state,
                      const CudaGraphLaunchParams &launch,
                      int kIdx,
                      std::vector<CUgraphNode> deps) {
  for (int b : plan.kernelOutputBufs[kIdx])
    if (isIntermediate(plan, b))
      addUniqueNode(deps, state.allocNodes[b]);

  KernelLaunchDesc &stored = state.descs[kIdx];
  stored.update(plan, data, cuFuncs, kIdx, launch);

  CUgraphNode node = nullptr;
  if (stored.isKernelNode) {
    checkCu(cuGraphAddKernelNode(&node, state.graph,
                                 deps.empty() ? nullptr : deps.data(),
                                 deps.size(), &stored.params),
            "cuGraphAddKernelNode");
  } else {
    checkCu(cuGraphAddEmptyNode(&node, state.graph,
                                deps.empty() ? nullptr : deps.data(),
                                deps.size()),
            "cuGraphAddEmptyNode");
  }
  state.kernelNodes[kIdx] = node;
  state.kernelNodeIsKernel[kIdx] = stored.isKernelNode;
}

void addFreeNodes(const GraphPlan &plan,
                  CudaGraphLaunchState &state) {
  for (int b = firstIntermediateBuffer(plan); b < plan.numBuffers; ++b) {
    std::vector<CUgraphNode> deps;
    for (int kIdx = 0; kIdx < static_cast<int>(plan.kernelInputBufs.size());
         ++kIdx) {
      const auto &ins = plan.kernelInputBufs[kIdx];
      if (std::find(ins.begin(), ins.end(), b) != ins.end())
        addUniqueNode(deps, state.kernelNodes[kIdx]);
    }

    if (deps.empty()) {
      int producer = plan.producerKernel[b];
      if (producer >= 0)
        addUniqueNode(deps, state.kernelNodes[producer]);
    }

    CUgraphNode freeNode = nullptr;
    checkCu(cuGraphAddMemFreeNode(
                &freeNode, state.graph,
                deps.empty() ? nullptr : deps.data(),
                deps.size(),
                static_cast<CUdeviceptr>(state.graphAllocBufPtrs[b])),
            "cuGraphAddMemFreeNode");
    state.freeNodes[b] = freeNode;
  }
}

void buildCudaGraphState(const GraphPlan &plan,
                         const ExecutableData &data,
                         const std::vector<CUfunction> &cuFuncs,
                         CudaGraphLaunchState &state,
                         Executor *exec,
                         int64_t timeLength, int64_t numStocks,
                         const std::vector<std::pair<std::string, uintptr_t>> &args,
                         int64_t mask,
                         int minChunkWarmupFactor,
                         double smFillFactor) {
  checkCu(cuGraphCreate(&state.graph, 0), "cuGraphCreate");

  const int nBufs = plan.numBuffers;
  const int nKernels = static_cast<int>(data.kernels.size());
  state.graphAllocBufPtrs.assign(nBufs, 0);
  state.allocNodes.assign(nBufs, nullptr);
  state.kernelNodes.assign(nKernels, nullptr);
  state.kernelNodeIsKernel.assign(nKernels, false);
  state.freeNodes.assign(nBufs, nullptr);
  state.descs.resize(nKernels);

  CUdevice device = 0;
  checkCu(cuCtxGetDevice(&device), "cuCtxGetDevice");
  const size_t bytes = intermediateBytes(data, timeLength, numStocks);

  CudaGraphLaunchParams launch = makeLaunchParams(
      exec, timeLength, numStocks, mask, minChunkWarmupFactor, smFillFactor,
      resolveExternalBufferPointers(plan, data, args));

  for (int kIdx : plan.launchOrder) {
    std::vector<CUgraphNode> deps = kernelInputDeps(plan, state, kIdx);
    addOutputAllocNodes(plan, state, device, bytes, kIdx, deps,
                        launch.bufPtrs);
    addOneKernelNode(plan, data, cuFuncs, state, launch, kIdx,
                     std::move(deps));
  }

  addFreeNodes(plan, state);
  checkCu(cuGraphInstantiate(&state.graphExec, state.graph, 0),
          "cuGraphInstantiate");
  state.cachedLaunchParams = std::move(launch);
}

void updateCudaGraphKernelParams(
    const GraphPlan &plan,
    const ExecutableData &data,
    const std::vector<CUfunction> &cuFuncs,
    CudaGraphLaunchState &state,
    const CudaGraphLaunchParams &cached,
    const CudaGraphLaunchParams &launch) {
  const bool bufferOnly = sameLaunchParamsExceptBuffers(cached, launch);
  for (int kIdx : plan.launchOrder) {
    KernelLaunchDesc &desc = state.descs[kIdx];
    bool changed = false;
    if (bufferOnly) {
      changed = desc.updateBuffer(plan, kIdx, launch.bufPtrs);
    } else {
      desc.update(plan, data, cuFuncs, kIdx, launch);
      changed = true;
    }
    if (!changed)
      continue;
    if (state.kernelNodeIsKernel[desc.kernelIndex] != desc.isKernelNode)
      throw std::runtime_error(
          "kun_cuda::launchOnStream(cuda_graph): kernel/empty node shape "
          "changed without graph rebuild");
    if (!desc.isKernelNode)
      continue;
    checkCu(cuGraphExecKernelNodeSetParams(
                state.graphExec, state.kernelNodes[desc.kernelIndex],
                &desc.params),
            "cuGraphExecKernelNodeSetParams");
  }
}

} // namespace

CudaGraphLaunchState::~CudaGraphLaunchState() noexcept {
  if (hasLaunch && completionEvent)
    (void)cuEventSynchronize(completionEvent);
  if (graphExec)
    (void)cuGraphExecDestroy(graphExec);
  if (graph)
    (void)cuGraphDestroy(graph);
  if (completionEvent)
    (void)cuEventDestroy(completionEvent);
}

void Executable::launchCudaGraphOnStream(
    Executor *exec,
    int64_t timeLength, int64_t numStocks,
    const std::vector<std::pair<std::string, uintptr_t>> &args,
    int64_t mask,
    int minChunkWarmupFactor,
    double smFillFactor) {
  if (!cudaGraphState_)
    cudaGraphState_ = std::make_unique<CudaGraphLaunchState>();

  const bool needRebuild =
      !cudaGraphState_->graphExec ||
      !cudaGraphState_->cachedLaunchParams ||
      cudaGraphState_->cachedLaunchParams->timeLength != timeLength ||
      cudaGraphState_->cachedLaunchParams->numStocks != numStocks;

  if (needRebuild) {
    ensureNoInFlight(*cudaGraphState_, "rebuilding");
    resetCudaGraphState();
    cudaGraphState_ = std::make_unique<CudaGraphLaunchState>();
    buildCudaGraphState(*plan_, data_, cuFuncs_, *cudaGraphState_,
                        exec, timeLength, numStocks, args,
                        mask, minChunkWarmupFactor, smFillFactor);
  } else {
    ensureNoInFlight(*cudaGraphState_, "updating");
    CudaGraphLaunchParams launch = makeLaunchParams(
        *plan_, data_, *cudaGraphState_, exec, timeLength, numStocks, args,
        mask, minChunkWarmupFactor, smFillFactor);
    if (!sameLaunchParams(*cudaGraphState_->cachedLaunchParams, launch)) {
      updateCudaGraphKernelParams(*plan_, data_, cuFuncs_, *cudaGraphState_,
                                  *cudaGraphState_->cachedLaunchParams,
                                  launch);
      cudaGraphState_->cachedLaunchParams = std::move(launch);
    }
  }

  checkCu(cuGraphLaunch(cudaGraphState_->graphExec, exec->stream()),
          "cuGraphLaunch");
  if (!cudaGraphState_->completionEvent)
    checkCu(cuEventCreate(&cudaGraphState_->completionEvent,
                          CU_EVENT_DISABLE_TIMING),
            "cuEventCreate(cuda graph completion)");
  checkCu(cuEventRecord(cudaGraphState_->completionEvent, exec->stream()),
          "cuEventRecord(cuda graph completion)");
  cudaGraphState_->hasLaunch = true;
}

void Executable::resetCudaGraphState() noexcept {
  cudaGraphState_.reset();
}

} // namespace kun_cuda
