//===- Runtime.h - kun_cuda runtime: ExecutableData + Executable -------===//
//
// Pure runtime piece, decoupled from the MLIR compiler and the Python
// binding.  The compiler produces an `ExecutableData` (cubin + metadata),
// the runtime turns that into a loaded `Executable` (cuModuleLoadData +
// cuModuleGetFunction) and launches it.
//
// This header forward-declares the two opaque CUDA Driver types it
// stores by pointer (CUmodule / CUfunction) so consumers don't need to
// pull in <cuda.h>.  These typedefs match cuda.h's verbatim — they have
// been ABI-stable for two decades.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

extern "C" {
typedef struct CUmod_st  *CUmodule;
typedef struct CUfunc_st *CUfunction;
} // extern "C"

namespace kun_cuda {

/// Everything needed to load + launch a compiled kunir kernel.  No CUDA
/// types — kept POD so the compiler library can populate it without
/// depending on cuda.h.
struct ExecutableData {
  std::vector<char> cubin;             ///< raw cubin bytes (ELF)
  std::string kernelName;              ///< symbol name in the cubin
  std::vector<std::string> inputNames; ///< from kungpu.input_names
  std::vector<std::string> outputNames;///< from kungpu.output_names
  int64_t warpsPerCta = 1;             ///< from kungpu.target_spec
  int64_t vectorSize  = 1;             ///< stocks-per-thread, from target_spec
};

/// RAII wrapper around a loaded cubin + resolved kernel function.
/// Construction calls `cuModuleLoadData` + `cuModuleGetFunction` on the
/// CUDA primary context of the calling thread (which must already exist).
/// Destruction calls `cuModuleUnload`.
class Executable {
public:
  /// Throws std::runtime_error on driver errors or missing CUDA context.
  /// Takes an rvalue — caller `std::move`s the data in.
  explicit Executable(ExecutableData &&data);
  ~Executable();

  // Non-copyable, non-movable — wrap in unique_ptr / shared_ptr if you
  // need transferable ownership.
  Executable(const Executable &)            = delete;
  Executable &operator=(const Executable &) = delete;
  Executable(Executable &&)                 = delete;
  Executable &operator=(Executable &&)      = delete;

  const ExecutableData &data() const noexcept { return data_; }
  const std::string &kernelName() const noexcept { return data_.kernelName; }
  const std::vector<std::string> &inputNames()  const noexcept { return data_.inputNames; }
  const std::vector<std::string> &outputNames() const noexcept { return data_.outputNames; }
  int64_t warpsPerCta() const noexcept { return data_.warpsPerCta; }
  int64_t vectorSize()  const noexcept { return data_.vectorSize; }

  /// Launch the kernel.  `timeLength` / `numStocks` describe the kernel
  /// invocation as a whole — the caller is responsible for verifying all
  /// device buffers are sized accordingly (TS layout: `(t, s)` at
  /// `ptr + (t*numStocks + s) * sizeof(T)`).
  ///
  /// `args` keys must equal `inputNames ++ outputNames` (order doesn't
  /// matter, names are looked up).  Grid configuration:
  ///
  ///   block_x = warps_per_cta * 32
  ///   grid_x  = ceil_div(numStocks, block_x * vector_size)
  ///
  /// Synchronous on the default stream.  Throws std::runtime_error on
  /// validation or driver errors.
  void launch(int64_t timeLength, int64_t numStocks,
              const std::vector<std::pair<std::string, uintptr_t>> &args);

private:
  ExecutableData data_;
  CUmodule cuModule_ = nullptr;
  CUfunction cuFunc_ = nullptr;
};

} // namespace kun_cuda
