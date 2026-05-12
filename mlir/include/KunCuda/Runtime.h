//===- Runtime.h - kun_cuda runtime: ExecutableData + Executable -------===//
//
// Pure runtime piece, decoupled from the MLIR compiler and the Python
// binding.  The compiler produces an `ExecutableData` (one cubin holding
// N kernels + per-kernel I/O *names* + the user's graph_inputs /
// graph_outputs lists).  The `Executable` ctor turns that into a loaded
// kernel set plus a fully resolved schedule:
//
//   names → buffer indices  ──→  topo sort  ──→  slot plan
//
// This split keeps the *compiler* concerned only with what's in the
// cubin, and lets the *runtime* own everything that's really a graph
// concern (dependency analysis, schedule, memory plan).  When we add
// CUDA-graph support later, all the input it needs already lives in the
// runtime: per-kernel buffer indices, the producer-kernel-of-each-buffer
// map, and the intermediate slot mapping.
//
// Buffer-table layout (assigned at Executable-construction time):
//   indices [0 .. numGraphInputs)             → graph inputs
//   indices [numGraphInputs .. firstInter)    → graph outputs
//   indices [firstInter .. numBuffers)        → intermediates
//
// Memory planning:
//   Intermediates share a pre-allocated slot pool sized to
//   `peakIntermediateSlots`.  Slot reuse is computed by refcount + LIFO
//   free pool over the topo-sorted schedule.  Slots are allocated lazily
//   on the first launch (and re-allocated if `(timeLength, numStocks)`
//   changes), then reused across subsequent launches with the same shape.
//
// This header forward-declares the two opaque CUDA Driver types so
// consumers don't need to pull in <cuda.h>.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

extern "C" {
typedef struct CUmod_st    *CUmodule;
typedef struct CUfunc_st   *CUfunction;
typedef struct CUstream_st *CUstream;
} // extern "C"

namespace kun_cuda {

/// Internal: the resolved schedule + memory plan.  Forward-declared so
/// the public header doesn't have to expose buffer-index tables,
/// producer maps, etc.  Fully defined in Runtime.cpp.
struct GraphPlan;

//===----------------------------------------------------------------------===//
// Compile-time output (all names — runtime resolves them to indices)
//===----------------------------------------------------------------------===//

/// Kernel dispatch kind.  `Jit` kernels live in the cubin produced by
/// the MLIR pipeline and are launched with the project-wide stock-major
/// grid (block_x = warps_per_cta * 32, grid_x = ceil(S / block_x)).
/// `ExtCsRank*` kernels are pre-compiled PTX bundled inside
/// libKunCudaRuntime; the executor lazy-loads them as a second
/// CUmodule and launches them with a time-major grid + dynamic shared
/// memory sized to the cross-section (one CTA per timestep).
enum class KernelKind : int32_t {
  Jit          = 0,
  ExtCsRankF32 = 1,
  ExtCsRankF64 = 2,
};

/// Per-kernel metadata, in name form.  This is what the compiler can
/// produce by walking a single lowered llvm.func — no graph topology
/// reasoning required.
struct KernelMeta {
  std::string kernelName;                    ///< symbol in the cubin (Jit) or in the bundled PTX (ExtCsRank*)
  KernelKind kind = KernelKind::Jit;         ///< picked by the MLIR pass; default is the regular path
  std::vector<std::string> inputNames;       ///< kungpu.input_names, in argv order
  std::vector<std::string> outputNames;      ///< kungpu.output_names, in argv order
};

/// What the compiler hands the runtime: a cubin + the kernels it
/// contains, declared purely by name.  `graphInputs` / `graphOutputs`
/// are user-supplied: they pick which named buffers cross the
/// graph-runtime boundary; everything else a kernel produces is treated
/// as an intermediate.
struct ExecutableData {
  std::vector<char> cubin;
  int64_t warpsPerCta = 1;          ///< from kungpu.target_spec (graph-wide)
  int64_t vectorSize  = 1;          ///< from kungpu.target_spec (graph-wide)
  std::vector<KernelMeta> kernels;  ///< unordered set; runtime topo-sorts
  std::vector<std::string> graphInputs;
  std::vector<std::string> graphOutputs;
};

//===----------------------------------------------------------------------===//
// Executable
//===----------------------------------------------------------------------===//

/// RAII wrapper around a loaded cubin + the resolved graph plan.
///
/// Construction:
///   1. Resolve names → buffer indices (graphInputs first, graphOutputs
///      next, intermediates last).
///   2. Build per-kernel int-index I/O lists and a producer-of-each-buffer
///      table.
///   3. Validate the graph (single producer; every consumer either a
///      graph input or has a producer; every graph output is produced).
///   4. Kahn topo sort over kernel-to-kernel edges.
///   5. Slot plan via refcount + LIFO free pool.
///   6. cuModuleLoadData + cuModuleGetFunction × N on the calling
///      thread's primary CUDA context (which must already exist).
///
/// Destruction calls `cuModuleUnload` and frees the slot pool.
class Executable {
public:
  /// Throws std::runtime_error on driver errors, missing CUDA context,
  /// or graph-validation failures.  Takes an rvalue — caller `std::move`s
  /// the data in.
  explicit Executable(ExecutableData &&data);
  ~Executable();

  // Non-copyable, non-movable — wrap in unique_ptr / shared_ptr if you
  // need transferable ownership.
  Executable(const Executable &)            = delete;
  Executable &operator=(const Executable &) = delete;
  Executable(Executable &&)                 = delete;
  Executable &operator=(Executable &&)      = delete;

  // ── Accessors (compile-time data) ─────────────────────────────────
  const ExecutableData &data() const noexcept { return data_; }
  const std::vector<std::string> &graphInputs()  const noexcept { return data_.graphInputs; }
  const std::vector<std::string> &graphOutputs() const noexcept { return data_.graphOutputs; }
  int64_t warpsPerCta() const noexcept { return data_.warpsPerCta; }
  int64_t vectorSize()  const noexcept { return data_.vectorSize; }
  size_t  numKernels()  const noexcept { return data_.kernels.size(); }

  // ── Accessors (runtime-resolved plan) ─────────────────────────────
  // Defined out-of-line so the header doesn't need GraphPlan's layout.

  /// Topo-sorted indices into `data().kernels` — the order the runtime
  /// launches kernels on the single CUDA stream.
  const std::vector<int> &launchOrder() const noexcept;
  /// Total buffer-table slots = numGraphInputs + numGraphOutputs +
  /// (number of distinct intermediates produced by kernels).
  int  numBuffers()            const noexcept;
  /// Number of physical intermediate buffers actually allocated by the
  /// runtime (after slot reuse).
  int  peakIntermediateSlots() const noexcept;

  /// Launch every kernel in `launchOrder` asynchronously on `stream`.
  /// **Does not synchronize** — the caller (typically `Executor::runGraph`
  /// + `Executor::synchronize`) owns waiting for completion.
  ///
  /// `args` keys must equal `graphInputs ++ graphOutputs` (order
  /// doesn't matter; the runtime hashes them into the buffer table).
  /// Intermediate buffers are owned by the executable and reused across
  /// launches with matching `(timeLength, numStocks)`.
  ///
  /// Grid configuration (per kernel — identical because warps_per_cta
  /// and vector_size are graph-wide):
  ///   block_x = warps_per_cta * 32
  ///   grid_x  = ceil_div(numStocks, block_x * vector_size)
  ///
  /// `devMaxSmemBytes` is the device's MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
  /// cached by the caller (`Executor`) so the runtime can validate
  /// `num_stocks * sizeof(T)` against the GPU's smem cap before
  /// invoking cuLaunchKernel for external cs_rank kernels.  Pass 0 if
  /// there are no external kernels in the executable (the check is a
  /// no-op in that case).
  ///
  /// Throws std::runtime_error on validation or driver errors.  This is
  /// a low-level entry point — most users go through `Executor::runGraph`.
  void launchOnStream(int64_t timeLength, int64_t numStocks,
                       const std::vector<std::pair<std::string, uintptr_t>> &args,
                       CUstream stream,
                       int devMaxSmemBytes);

private:
  /// Allocate (or re-allocate, if shape changed) the intermediate slot
  /// pool.  Each slot holds one `T × S` float32 array.
  void ensureSlotPool(int64_t timeLength, int64_t numStocks);
  /// Free all slot allocations.  Called from dtor and on shape change.
  void freeSlotPool();

  ExecutableData data_;
  std::unique_ptr<GraphPlan> plan_;          ///< pImpl — defined in Runtime.cpp

  CUmodule cuModule_ = nullptr;
  /// Module holding the pre-compiled cs_rank PTX.  Loaded at
  /// construction time iff any kernel has `kind != Jit`; null
  /// otherwise.
  CUmodule csRankModule_ = nullptr;
  std::vector<CUfunction> cuFuncs_;          ///< parallel to data_.kernels

  // Lazily allocated intermediate buffers, one CUdeviceptr per slot
  // (stored as uintptr_t to keep the header CUDA-free).
  std::vector<uintptr_t> slotBufs_;
  int64_t cachedT_ = -1;
  int64_t cachedS_ = -1;
};

//===----------------------------------------------------------------------===//
// Executor — wraps a CUDA stream and exposes the runGraph / synchronize
// pair, mirroring the CPU `kun::Executor` shape.
//
// Default constructor uses the CUDA default (NULL) stream.  The
// stream-injecting constructor lets callers reuse a stream they already
// own (e.g. `cupy.cuda.Stream`'s `.ptr`); the Executor does NOT take
// ownership and never destroys the stream.
//
// `runGraph` is asynchronous — it queues every kernel in the executable
// onto this stream and returns immediately.  Call `synchronize` (or wait
// on the stream by other means) before reading results back to host.
//
// Thread / Executable model: an `Executable`'s intermediate slot pool
// is mutable state shared by every `runGraph` call against it.  Driving
// the same Executable from two Executors concurrently is unsafe — pair
// them 1:1, or serialize the calls externally.
//===----------------------------------------------------------------------===//

class Executor {
public:
  /// Use the CUDA default stream.
  Executor();
  /// Reuse a stream the caller owns (e.g. cupy's `.ptr`).  We do not
  /// destroy it; lifetime is the caller's responsibility.
  explicit Executor(CUstream stream);
  ~Executor();

  Executor(const Executor &)            = delete;
  Executor &operator=(const Executor &) = delete;
  Executor(Executor &&)                 = delete;
  Executor &operator=(Executor &&)      = delete;

  /// Queue all kernels in `exe` on this executor's stream.  Async — does
  /// not synchronize.  Throws std::runtime_error on validation / driver
  /// errors.
  void runGraph(Executable &exe,
                int64_t timeLength, int64_t numStocks,
                const std::vector<std::pair<std::string, uintptr_t>> &args);

  /// Block until all queued work on this stream completes.
  void synchronize();

  /// Raw stream handle (default-stream Executor returns nullptr).
  CUstream stream() const noexcept { return stream_; }
  /// Cached MAX_SHARED_MEMORY_PER_BLOCK_OPTIN of the device this
  /// Executor's CUcontext is bound to, queried once at construction.
  /// Used to validate cs_rank dynamic-smem requests at launch time
  /// without a per-launch driver call.
  int devMaxSmemBytes() const noexcept { return devMaxSmemBytes_; }

private:
  CUstream stream_ = nullptr;
  int devMaxSmemBytes_ = 0;
};

} // namespace kun_cuda
