//===- Runtime.cpp - kun_cuda::Executable implementation ---------------===//

#include "KunCuda/Runtime.h"

#include <cuda.h>

#include <limits>
#include <sstream>
#include <stdexcept>

namespace kun_cuda {

namespace {

void checkCu(CUresult r, const char *what) {
  if (r == CUDA_SUCCESS)
    return;
  const char *s = nullptr;
  cuGetErrorString(r, &s);
  throw std::runtime_error(std::string(what) + ": " +
                            (s ? s : "unknown CUDA error"));
}

std::string joinNames(const std::vector<std::string> &v) {
  std::string r;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i)
      r += ", ";
    r += v[i];
  }
  return r;
}

} // namespace

Executable::Executable(ExecutableData &&data) : data_(std::move(data)) {
  // Require a primary context to already exist on the calling thread —
  // the caller's job to set one up (e.g. by allocating any device memory
  // through cupy / cudaMalloc).
  CUcontext cur = nullptr;
  checkCu(cuCtxGetCurrent(&cur), "cuCtxGetCurrent");
  if (!cur) {
    throw std::runtime_error(
        "kun_cuda::Executable: no current CUDA context.  Initialise the "
        "driver first (e.g. allocate any device memory via cupy or "
        "cudaMalloc) before constructing an Executable.");
  }
  checkCu(cuModuleLoadData(&cuModule_, data_.cubin.data()),
           "cuModuleLoadData");
  checkCu(cuModuleGetFunction(&cuFunc_, cuModule_, data_.kernelName.c_str()),
           "cuModuleGetFunction");
}

Executable::~Executable() {
  // Best-effort unload; we deliberately don't propagate driver errors out
  // of a destructor.
  if (cuModule_)
    cuModuleUnload(cuModule_);
}

void Executable::launch(
    int64_t timeLength, int64_t numStocks,
    const std::vector<std::pair<std::string, uintptr_t>> &args) {
  // 1.  Resolve full ordered argument list (inputs first, then outputs).
  std::vector<std::string> ordered;
  ordered.reserve(data_.inputNames.size() + data_.outputNames.size());
  for (auto &n : data_.inputNames)
    ordered.push_back(n);
  for (auto &n : data_.outputNames)
    ordered.push_back(n);
  if (ordered.empty())
    throw std::runtime_error("kun_cuda::launch: kernel has no I/O args");

  // 2.  Resolve each name to its device pointer — list is small, linear
  //     scan is fine.
  auto findArg = [&](const std::string &n) -> const uintptr_t * {
    for (auto &kv : args)
      if (kv.first == n)
        return &kv.second;
    return nullptr;
  };

  std::vector<uintptr_t> resolved;
  resolved.reserve(ordered.size());
  for (auto &n : ordered) {
    auto *a = findArg(n);
    if (!a) {
      throw std::runtime_error("kun_cuda::launch: missing argument '" + n +
                                "' (kernel expects: " + joinNames(ordered) +
                                ")");
    }
    resolved.push_back(*a);
  }

  // 3.  Caller is responsible for shape consistency; we only check that
  //     (T, S) fit in i32 since the kernel signature uses i32 i32.
  if (timeLength > std::numeric_limits<int32_t>::max() ||
      numStocks  > std::numeric_limits<int32_t>::max() ||
      timeLength < 0 || numStocks < 0) {
    throw std::runtime_error(
        "kun_cuda::launch: time_length / num_stocks out of i32 range "
        "(kernel signature uses i32, i32)");
  }

  // 4.  Build kernel argv: [i32 time_len, i32 num_stocks, ptr0, ptr1, ...]
  int32_t timeLenI32   = static_cast<int32_t>(timeLength);
  int32_t numStocksI32 = static_cast<int32_t>(numStocks);
  std::vector<CUdeviceptr> ptrs(resolved.size());
  for (size_t i = 0; i < resolved.size(); ++i)
    ptrs[i] = static_cast<CUdeviceptr>(resolved[i]);

  std::vector<void *> argPtrs;
  argPtrs.reserve(2 + ptrs.size());
  argPtrs.push_back(&timeLenI32);
  argPtrs.push_back(&numStocksI32);
  for (auto &p : ptrs)
    argPtrs.push_back(&p);

  // 5.  block / grid.
  unsigned blockX = static_cast<unsigned>(data_.warpsPerCta * 32);
  if (blockX == 0)
    throw std::runtime_error("kun_cuda::launch: warps_per_cta is 0");
  uint64_t stocksPerBlock =
      static_cast<uint64_t>(blockX) * static_cast<uint64_t>(data_.vectorSize);
  unsigned gridX = static_cast<unsigned>(
      (static_cast<uint64_t>(numStocks) + stocksPerBlock - 1) / stocksPerBlock);

  // sharedMemBytes = 0 — shared memory is static (declared as
  // `llvm.mlir.global addr_space=3` and allocated by ptxas into the
  // cubin's `.shared` section); the dynamic-smem launch parameter does
  // not apply.
  checkCu(cuLaunchKernel(cuFunc_, gridX, 1, 1, blockX, 1, 1,
                           /*sharedMemBytes=*/0, /*stream=*/nullptr,
                           argPtrs.data(), nullptr),
           "cuLaunchKernel");
  checkCu(cuCtxSynchronize(), "cuCtxSynchronize");
}

} // namespace kun_cuda
