//===- Runtime.cpp - kun_cuda::Executable implementation ---------------===//
//
// The ctor pipeline is split into focused helpers — each step is small
// enough to read top-to-bottom on its own:
//
//   buildBufferIndices   — assign integer indices to every named buffer
//   resolveKernelIO      — translate per-kernel name lists to indices,
//                           build producer-of-each-buffer table
//   validateGraph        — check single producer, all consumers reachable,
//                           graph_outputs all produced, no self-dependency
//   topoSort             — Kahn's algorithm; rejects cycles
//   planSlots            — refcount + LIFO free pool over the topo order
//
// All helpers live in this file's anonymous namespace.  Future
// CUDA-graph support reuses the same plan: `kernelInputBufs` +
// `producerKernel` are exactly the dep edges cuGraph needs.
//
//===----------------------------------------------------------------------===//

#include "KunCuda/Runtime.h"

#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

// Pre-compiled cs_rank PTX, embedded by EmbedFile.cmake.  Exposes
// `kun_cs_rank_ptx[]` (bytes) and `kun_cs_rank_ptx_len`.
#include "cs_rank_ptx.inc"

namespace kun_cuda {

//===----------------------------------------------------------------------===//
// GraphPlan — pImpl payload, hidden from the public header
//===----------------------------------------------------------------------===//

/// Runtime-resolved schedule + memory plan.  All buffer references here
/// are integer indices into the flat buffer table.  Storing
/// `producerKernel` makes it cheap to re-derive kernel-to-kernel
/// dependency edges (needed for future cuGraph support: kernel K's deps
/// = {producerKernel[b] for b in kernelInputBufs[K], filtered to ≥ 0}).
struct GraphPlan {
  int numBuffers       = 0;
  int numGraphInputs   = 0;
  int numGraphOutputs  = 0;

  // Name → index for the user-facing args dict.  Other lookups happen
  // by integer indexing.
  std::unordered_map<std::string, int> graphInputIdx;
  std::unordered_map<std::string, int> graphOutputIdx;

  // Per-kernel I/O resolved to buffer indices.  Parallel to ExecutableData::kernels.
  std::vector<std::vector<int>> kernelInputBufs;
  std::vector<std::vector<int>> kernelOutputBufs;

  // producerKernel[bufIdx] = kernel that writes the buffer, or -1 if
  // the buffer is a graph input.
  std::vector<int> producerKernel;

  // Topo order — a single valid linearization for the v0 single-stream
  // launcher.
  std::vector<int> launchOrder;

  // Slot assignment: one entry per buffer index.  -1 if the buffer is a
  // graph input/output; otherwise a slot index in [0, peakIntermediateSlots).
  std::vector<int> intermediateBufToSlot;
  int peakIntermediateSlots = 0;
};

namespace {

//===----------------------------------------------------------------------===//
// CUDA driver helpers
//===----------------------------------------------------------------------===//

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
    if (i) r += ", ";
    r += v[i];
  }
  return r;
}

//===----------------------------------------------------------------------===//
// Plan-building helpers — small POD intermediates so each helper is
// independent and trivially testable.
//===----------------------------------------------------------------------===//

struct BufTable {
  int numBuffers      = 0;
  int numGraphInputs  = 0;
  int numGraphOutputs = 0;
  // Name → index for *every* buffer (graph IO + intermediates).  Used by
  // resolveKernelIO; the per-role maps below are kept around for the
  // launch-time user args dict lookup.
  std::unordered_map<std::string, int> nameToIdx;
  std::unordered_map<std::string, int> graphInputIdx;
  std::unordered_map<std::string, int> graphOutputIdx;
};

struct KernelIO {
  std::vector<std::vector<int>> kernelInputBufs;   // [kernel][argv pos]
  std::vector<std::vector<int>> kernelOutputBufs;
  // producerKernel[bufIdx] = kernel index that writes that buffer, or
  // -1 if it's a graph input.
  std::vector<int> producerKernel;
};

struct SlotPlan {
  std::vector<int> intermediateBufToSlot;
  int peakIntermediateSlots = 0;
};

/// Step 1 — assign buffer indices.  Layout:
///   [0 .. numGraphInputs)                        graph inputs
///   [numGraphInputs .. numGraphInputs+numGraphOutputs)  graph outputs
///   [..numBuffers)                               intermediates
/// Intermediates are everything a kernel produces that isn't a
/// graph_output; they get consecutive indices in first-seen order.
BufTable buildBufferIndices(const std::vector<std::string> &graphInputs,
                              const std::vector<std::string> &graphOutputs,
                              const std::vector<KernelMeta> &kernels) {
  BufTable t;

  for (const auto &n : graphInputs) {
    if (t.nameToIdx.count(n))
      throw std::runtime_error(
          "kun_cuda::Executable: duplicate name in graph_inputs: '" + n + "'");
    int idx = static_cast<int>(t.nameToIdx.size());
    t.nameToIdx[n] = idx;
    t.graphInputIdx[n] = idx;
  }
  t.numGraphInputs = static_cast<int>(t.nameToIdx.size());

  for (const auto &n : graphOutputs) {
    if (t.nameToIdx.count(n))
      throw std::runtime_error(
          "kun_cuda::Executable: name '" + n +
          "' appears in both graph_inputs and graph_outputs (or twice in "
          "one of them)");
    int idx = static_cast<int>(t.nameToIdx.size());
    t.nameToIdx[n] = idx;
    t.graphOutputIdx[n] = idx;
  }
  t.numGraphOutputs =
      static_cast<int>(t.nameToIdx.size()) - t.numGraphInputs;

  // Walk every kernel output and assign new indices to anything we
  // haven't seen yet (intermediates).  We don't validate single-producer
  // here — that's `validateGraph`'s job — but we do need to avoid
  // double-assigning if two kernels share an output name.
  for (const auto &k : kernels)
    for (const auto &outName : k.outputNames)
      if (!t.nameToIdx.count(outName))
        t.nameToIdx[outName] = static_cast<int>(t.nameToIdx.size());

  t.numBuffers = static_cast<int>(t.nameToIdx.size());
  return t;
}

/// Step 2 — resolve each kernel's I/O name list to int indices, plus
/// build the producer-of-each-buffer table.  Throws on a kernel input
/// that wasn't declared anywhere (neither graph input nor produced).
KernelIO resolveKernelIO(const std::vector<KernelMeta> &kernels,
                           const BufTable &tbl) {
  KernelIO kio;
  kio.kernelInputBufs.resize(kernels.size());
  kio.kernelOutputBufs.resize(kernels.size());
  kio.producerKernel.assign(tbl.numBuffers, -1);

  for (int kIdx = 0; kIdx < static_cast<int>(kernels.size()); ++kIdx) {
    const auto &k = kernels[kIdx];

    kio.kernelInputBufs[kIdx].reserve(k.inputNames.size());
    for (const auto &n : k.inputNames) {
      auto it = tbl.nameToIdx.find(n);
      if (it == tbl.nameToIdx.end())
        throw std::runtime_error(
            "kun_cuda::Executable: kernel '" + k.kernelName + "' consumes '" +
            n + "' which is neither a graph_input nor produced by any kernel");
      kio.kernelInputBufs[kIdx].push_back(it->second);
    }

    kio.kernelOutputBufs[kIdx].reserve(k.outputNames.size());
    for (const auto &n : k.outputNames) {
      // Index existence is guaranteed by buildBufferIndices.
      int b = tbl.nameToIdx.at(n);
      kio.kernelOutputBufs[kIdx].push_back(b);
      kio.producerKernel[b] = kIdx;  // last writer wins; validateGraph
                                      // catches multi-producer below.
    }
  }
  return kio;
}

/// Step 3 — graph-level validation.  Catches the cases buildBufferIndices /
/// resolveKernelIO can't, namely:
///   * two kernels claim to produce the same buffer
///   * a graph_output is declared but never produced
///   * a graph_input is also produced by a kernel (overlap is silly)
void validateGraph(const std::vector<KernelMeta> &kernels,
                     const std::vector<std::string> &graphOutputs,
                     const BufTable &tbl,
                     const KernelIO &kio) {
  // Multi-producer: count how many times each output name appears as a
  // kernel output.
  std::unordered_map<std::string, int> outCounts;
  std::unordered_map<std::string, std::string> firstProducer;
  for (const auto &k : kernels) {
    for (const auto &n : k.outputNames) {
      if (++outCounts[n] == 1)
        firstProducer[n] = k.kernelName;
      else if (outCounts[n] == 2)
        throw std::runtime_error(
            "kun_cuda::Executable: name '" + n +
            "' is produced by both kernel '" + firstProducer[n] +
            "' and kernel '" + k.kernelName + "'");
    }
  }

  // graph_outputs must be produced.
  for (const auto &n : graphOutputs) {
    int b = tbl.nameToIdx.at(n);
    if (kio.producerKernel[b] < 0)
      throw std::runtime_error(
          "kun_cuda::Executable: graph_output '" + n +
          "' is not produced by any kernel");
  }

  // graph_inputs must NOT be produced by any kernel — an input is by
  // definition supplied by the caller.
  for (const auto &kv : tbl.graphInputIdx) {
    if (kio.producerKernel[kv.second] >= 0)
      throw std::runtime_error(
          "kun_cuda::Executable: graph_input '" + kv.first +
          "' is also produced by a kernel; use a different name for the "
          "kernel output");
  }
}

/// Step 4 — Kahn topological sort over kernel-to-kernel edges.  An edge
/// `producer → consumer` exists whenever consumer reads any buffer that
/// producer writes.  Multi-edges between the same pair count as one.
/// Throws on cycle.
std::vector<int> topoSort(const KernelIO &kio, int numKernels) {
  std::vector<int> indeg(numKernels, 0);
  std::vector<std::vector<int>> succ(numKernels);

  // Build edges, deduped per (producer, consumer) pair.
  for (int kIdx = 0; kIdx < numKernels; ++kIdx) {
    std::vector<int> deps;
    for (int b : kio.kernelInputBufs[kIdx]) {
      int p = kio.producerKernel[b];
      if (p < 0) continue;                   // graph input
      if (p == kIdx)
        throw std::runtime_error(
            "kun_cuda::Executable: kernel index " + std::to_string(kIdx) +
            " depends on its own output");
      bool seen = false;
      for (int d : deps) if (d == p) { seen = true; break; }
      if (!seen) deps.push_back(p);
    }
    indeg[kIdx] = static_cast<int>(deps.size());
    for (int p : deps) succ[p].push_back(kIdx);
  }

  std::vector<int> order;
  order.reserve(numKernels);
  std::vector<int> ready;
  for (int i = 0; i < numKernels; ++i)
    if (indeg[i] == 0) ready.push_back(i);
  while (!ready.empty()) {
    int k = ready.back();
    ready.pop_back();
    order.push_back(k);
    for (int n : succ[k])
      if (--indeg[n] == 0)
        ready.push_back(n);
  }
  if (static_cast<int>(order.size()) != numKernels)
    throw std::runtime_error(
        "kun_cuda::Executable: cycle detected in kernel dependency graph");
  return order;
}

/// Step 5 — slot allocation for intermediates.  Refcount = number of
/// kernel-input slots that reference the buffer, plus +1 for graph
/// outputs (so we never try to recycle them).  Walking the topo order:
///   * before launching kernel K, allocate a fresh slot for each
///     intermediate output of K (drawn from the LIFO free pool when
///     possible),
///   * after, decrement refcounts on K's inputs; any intermediate that
///     hits zero returns its slot to the free pool.
SlotPlan planSlots(const std::vector<int> &launchOrder,
                    const BufTable &tbl,
                    const KernelIO &kio) {
  SlotPlan plan;
  plan.intermediateBufToSlot.assign(tbl.numBuffers, -1);
  const int firstIntermediate = tbl.numGraphInputs + tbl.numGraphOutputs;

  // Initial refcounts.
  std::vector<int> refcount(tbl.numBuffers, 0);
  for (const auto &ins : kio.kernelInputBufs)
    for (int b : ins)
      refcount[b]++;
  // graph_outputs are externally visible — pin them so we never try to
  // reuse them (they don't have slots anyway, but this keeps the loop
  // free of special cases).
  for (int i = tbl.numGraphInputs; i < firstIntermediate; ++i)
    refcount[i]++;

  std::vector<int> freePool;
  int nextNew = 0;

  auto allocSlot = [&]() -> int {
    if (!freePool.empty()) { int s = freePool.back(); freePool.pop_back(); return s; }
    int s = nextNew++;
    if (nextNew > plan.peakIntermediateSlots) plan.peakIntermediateSlots = nextNew;
    return s;
  };

  for (int kIdx : launchOrder) {
    // Allocate slots for this kernel's intermediate outputs.  Outputs
    // that ARE graph_outputs use caller-owned buffers and don't need a
    // slot.
    for (int b : kio.kernelOutputBufs[kIdx]) {
      if (b < firstIntermediate) continue;
      plan.intermediateBufToSlot[b] = allocSlot();
    }
    // Decrement refcounts on inputs; intermediate slots whose refcount
    // hits zero return to the free pool.
    for (int b : kio.kernelInputBufs[kIdx]) {
      if (--refcount[b] == 0 && b >= firstIntermediate) {
        int s = plan.intermediateBufToSlot[b];
        if (s >= 0) freePool.push_back(s);
      }
    }
  }
  return plan;
}

//===----------------------------------------------------------------------===//
// Launch helpers — pure functions used by launchOnStream below.
//===----------------------------------------------------------------------===//

/// Translate the user-supplied {name → device_ptr} args dict into a
/// flat buffer-index → pointer array, plug in the executable-owned
/// intermediate-slot pointers, and verify every graph_input /
/// graph_output the plan expects was provided.  Throws on unknown or
/// missing names.
static std::vector<uintptr_t> resolveBufferPointers(
    const GraphPlan &plan,
    const ExecutableData &data,
    const std::vector<std::pair<std::string, uintptr_t>> &args,
    const std::vector<uintptr_t> &slotBufs) {
  std::vector<uintptr_t> bufPtrs(plan.numBuffers, 0);
  std::vector<bool>      filled(plan.numBuffers, false);

  for (const auto &kv : args) {
    auto itIn  = plan.graphInputIdx.find(kv.first);
    auto itOut = plan.graphOutputIdx.find(kv.first);
    int idx = -1;
    if (itIn != plan.graphInputIdx.end())
      idx = itIn->second;
    else if (itOut != plan.graphOutputIdx.end())
      idx = itOut->second;
    else
      throw std::runtime_error(
          "kun_cuda::launchOnStream: unexpected argument '" + kv.first +
          "' (expected: " + joinNames(data.graphInputs) + " | " +
          joinNames(data.graphOutputs) + ")");
    bufPtrs[idx] = kv.second;
    filled[idx] = true;
  }

  // Confirm every graph_input + graph_output was supplied.
  for (int i = 0; i < plan.numGraphInputs + plan.numGraphOutputs; ++i) {
    if (filled[i]) continue;
    std::string missing;
    for (auto &kv : plan.graphInputIdx)  if (kv.second == i) missing = kv.first;
    if (missing.empty())
      for (auto &kv : plan.graphOutputIdx) if (kv.second == i) missing = kv.first;
    throw std::runtime_error(
        "kun_cuda::launchOnStream: missing argument '" + missing + "'");
  }

  // Intermediates: index into the pre-allocated slot pool.
  for (int i = plan.numGraphInputs + plan.numGraphOutputs;
        i < plan.numBuffers; ++i) {
    int slot = plan.intermediateBufToSlot[i];
    bufPtrs[i] = slotBufs[slot];
  }
  return bufPtrs;
}

/// Stock-major × time-chunk launch: block_x = warps_per_cta*32,
/// grid_x = ceil(numStocks / (block_x * vector_size)),
/// grid_y = numChunks, no dynamic smem.
static void launchJitKernel(CUfunction fn,
                              int64_t numStocks,
                              int64_t warpsPerCta, int64_t vectorSize,
                              unsigned numChunks,
                              void **args, CUstream stream) {
  unsigned blockX = static_cast<unsigned>(warpsPerCta * 32);
  uint64_t stocksPerBlock =
      static_cast<uint64_t>(blockX) * static_cast<uint64_t>(vectorSize);
  unsigned gridX = static_cast<unsigned>(
      (static_cast<uint64_t>(numStocks) + stocksPerBlock - 1) /
      stocksPerBlock);
  // sharedMemBytes = 0 — JIT'd kernels declare static smem via
  // llvm.mlir.global addr_space=3; the dynamic-smem launch parameter
  // does not apply.
  checkCu(cuLaunchKernel(fn, gridX, numChunks, 1, blockX, 1, 1,
                           /*sharedMemBytes=*/0, stream, args, nullptr),
           "cuLaunchKernel");
}

/// Chunk plan for a single JIT kernel.  `chunkSize` is the time-axis
/// width of every chunk (last chunk gets clipped to `timeLength` by
/// kungpu.time_ub at runtime, so we don't have to special-case that
/// here).  `numChunks` is the y-dim of the launch grid.
///
/// Decision tree (per kernel, since per-partition `unreliableCount`
/// varies):
///
///   1. target chunks   = ceil(smFillFactor * numSMs / stockTiles), ≥ 1
///   2. cap by warmup   = floor(T / (factor * unreliableCount))
///        — bounds the per-chunk overhead of chunks ≥ 1, which redo the
///          trailing `unreliableCount` time steps to prime windowed
///          rolling state.  mask is NOT included here: it's a one-time
///          chunk-0 skip, not a per-chunk overhead.
///   3. cap by mask     = floor((T - 1) / mask)
///        — chunks ≥ 1 write output[t - mask] for t ∈ [cy*chunk_size, …);
///          if chunk_size ≤ mask, chunk 1's first output index is
///          negative (out-of-bounds gmem write).  Enforce chunk_size >
///          mask by capping num_chunks here.
///   4. numChunks       = clamp(target, 1, min(cap_warmup, cap_mask))
///   5. chunkSize       = ceil(T / numChunks)
///
/// When both unreliable == 0 and mask == 0, the only cap is T itself.
/// When numSMs == 0 (Executor couldn't query the device) or
/// smFillFactor ≤ 0, fall back to single-chunk.
struct ChunkPlan {
  int64_t chunkSize;
  unsigned numChunks;
};
static ChunkPlan computeChunkPlan(int64_t timeLength, int64_t numStocks,
                                     int64_t warpsPerCta, int64_t vectorSize,
                                     int64_t unreliableCount, int64_t mask,
                                     int minChunkWarmupFactor,
                                     double smFillFactor, int numSMs) {
  if (timeLength <= 0)
    return {timeLength, 1u};
  if (numSMs <= 0 || smFillFactor <= 0.0)
    return {timeLength, 1u};

  int64_t blockX = warpsPerCta * 32;
  int64_t stocksPerBlock = blockX * vectorSize;
  int64_t stockTiles =
      (numStocks + stocksPerBlock - 1) / stocksPerBlock;
  if (stockTiles <= 0) stockTiles = 1;

  // Target chunks just to fill the GPU.  Round up so we don't under-fill.
  int64_t targetChunks = static_cast<int64_t>(
      std::ceil(smFillFactor * static_cast<double>(numSMs) /
                  static_cast<double>(stockTiles)));
  if (targetChunks < 1) targetChunks = 1;

  // Caps on numChunks.  Start at T (degenerate upper bound: ≥ 1 step per
  // chunk) and tighten with each constraint; clamp to ≥ 1 once at the end.
  int64_t cap = timeLength;

  // Per-chunk warmup overhead bound (chunks ≥ 1 only).
  if (unreliableCount > 0 && minChunkWarmupFactor > 0)
    cap = std::min<int64_t>(
        cap,
        timeLength /
            (static_cast<int64_t>(minChunkWarmupFactor) * unreliableCount));

  // chunkSize > mask: chunks ≥ 1 compute output index t - mask, which
  // must be ≥ 0 for their writes.  chunk_size = ceil(T / numChunks);
  // we want ceil(T / numChunks) > mask, equivalently numChunks ≤
  // (T - 1) / mask.
  if (mask > 0)
    cap = std::min<int64_t>(cap, (timeLength - 1) / mask);

  if (cap < 1) cap = 1;

  int64_t numChunks = std::min<int64_t>(targetChunks, cap);
  if (numChunks < 1) numChunks = 1;

  int64_t chunkSize = (timeLength + numChunks - 1) / numChunks;
  return {chunkSize, static_cast<unsigned>(numChunks)};
}

/// External cs_rank launch.
///
/// Block / grid both auto-tuned — cs_rank is cross-sectional, so the
/// graph-wide `warps_per_cta` hint doesn't apply.
///
///   blockX = clamp(round_up(numStocks, 32), 32, 1024)
///       Each thread owns roughly one stock; when numStocks > 1024 the
///       kernel falls back to its built-in `for (i = tid; i < S; i +=
///       blockDim.x)` stride loop.
///
///   gridX  = min(timeLength, ceil(smFillFactor * numSMs))
///       The kernel does a contiguous time-axis slice per CTA via a
///       grid-stride loop (see kernels/cs_rank.cu).  For small T the
///       min clamps to 1 CTA per timestep (matches the pre-tuning
///       launch shape); for large T fewer CTAs each do more time
///       steps, reducing launch / scheduling overhead.
///
///   smem   = numStocks * sizeof(T)  (one cross-section, reused across
///                                     the CTA's time slice)
///
/// Falls back to (gridX = timeLength, blockX = 32) when the executor
/// couldn't query `numSMs` from the device — degenerate "one CTA per
/// timestep, one warp per CTA" still works correctly.
static void launchExtCsRankKernel(CUfunction fn, KernelKind kind,
                                    const std::string &kernelName,
                                    int64_t timeLength, int64_t numStocks,
                                    int devMaxSmemBytes,
                                    double smFillFactor, int numSMs,
                                    void **args, CUstream stream) {
  size_t elemSize = (kind == KernelKind::ExtCsRankF64) ? 8u : 4u;
  uint64_t smemBytes64 =
      static_cast<uint64_t>(numStocks) * static_cast<uint64_t>(elemSize);

  if (devMaxSmemBytes <= 0)
    throw std::runtime_error(
        "kun_cuda::launchOnStream: external cs_rank kernel '" + kernelName +
        "' requires Executor's devMaxSmemBytes to be set; got 0.  "
        "Construct the Executable through Executor::runGraph, or pass "
        "devMaxSmemBytes when calling launchOnStream directly.");
  if (smemBytes64 > static_cast<uint64_t>(devMaxSmemBytes))
    throw std::runtime_error(
        "kun_cuda::launchOnStream: cs_rank dynamic smem "
        "(num_stocks=" + std::to_string(numStocks) +
        " * sizeof(T)=" + std::to_string(elemSize) + " = " +
        std::to_string(smemBytes64) +
        " bytes) exceeds this GPU's MAX_SHARED_MEMORY_PER_BLOCK_OPTIN (" +
        std::to_string(devMaxSmemBytes) +
        " bytes).  Reduce num_stocks or run on a GPU with a larger smem budget.");

  if (timeLength <= 0)
    return; // empty time chunk — nothing to launch

  constexpr int kWarp = 32;
  constexpr int kMaxBlock = 1024;
  int64_t blockX64 =
      ((std::max<int64_t>(numStocks, 1) + kWarp - 1) / kWarp) * kWarp;
  if (blockX64 > kMaxBlock) blockX64 = kMaxBlock;
  unsigned blockX = static_cast<unsigned>(blockX64);

  // Target gridX = sm_fill_factor * numSMs (capped at timeLength so we
  // never launch idle CTAs).  numSMs == 0 (device query failed) →
  // gridX = timeLength, one CTA per timestep.
  unsigned gridX;
  if (numSMs > 0 && smFillFactor > 0.0) {
    int64_t target = static_cast<int64_t>(
        std::ceil(smFillFactor * static_cast<double>(numSMs)));
    if (target < 1) target = 1;
    if (target > timeLength) target = timeLength;
    gridX = static_cast<unsigned>(target);
  } else {
    gridX = static_cast<unsigned>(timeLength);
  }

  unsigned smemBytes = static_cast<unsigned>(smemBytes64);
  checkCu(cuLaunchKernel(fn, gridX, 1, 1, blockX, 1, 1,
                           smemBytes, stream, args, nullptr),
           "cuLaunchKernel(cs_rank)");
}

//===----------------------------------------------------------------------===//
// Kernel-module / kernel-symbol helpers — read ExecutableData, mutate
// the CUmodule and CUfunction handles the ctor is populating.
//===----------------------------------------------------------------------===//

/// Load the JIT'd cubin if non-empty; otherwise sanity-check that no
/// kernel actually needs it (every `kind == Jit` requires a cubin).
static void loadJitCubin(const ExecutableData &data, CUmodule &outModule) {
  if (!data.cubin.empty()) {
    checkCu(cuModuleLoadData(&outModule, data.cubin.data()),
             "cuModuleLoadData");
    return;
  }
  for (const auto &k : data.kernels)
    if (k.kind == KernelKind::Jit)
      throw std::runtime_error(
          "kun_cuda::Executable: JIT kernel '" + k.kernelName +
          "' declared but no cubin supplied — this is a compile-side bug");
}

/// Lazy-load the bundled cs_rank PTX as a second CUmodule iff any
/// kernel uses it.  The driver JITs PTX → SASS on first load (cached
/// system-wide in ~/.nv/ComputeCache), so this is sub-ms after the
/// first run on a given GPU.
static void loadCsRankPtxIfNeeded(const std::vector<KernelMeta> &kernels,
                                    CUmodule &outModule) {
  for (const auto &k : kernels) {
    if (k.kind != KernelKind::Jit) {
      checkCu(cuModuleLoadData(&outModule, kun_cs_rank_ptx),
               "cuModuleLoadData(cs_rank.ptx)");
      return;
    }
  }
}

/// Pick the right CUmodule + symbol name for a kernel and resolve it.
static CUfunction resolveOneKernelSymbol(const KernelMeta &k,
                                          CUmodule jitModule,
                                          CUmodule csRankModule) {
  CUmodule mod = nullptr;
  const char *symbol = nullptr;
  switch (k.kind) {
    case KernelKind::Jit:
      mod = jitModule;
      symbol = k.kernelName.c_str();
      break;
    case KernelKind::ExtCsRankF32:
      mod = csRankModule;
      symbol = "kun_cs_rank_f32";
      break;
    case KernelKind::ExtCsRankF64:
      mod = csRankModule;
      symbol = "kun_cs_rank_f64";
      break;
  }
  CUfunction fn = nullptr;
  checkCu(cuModuleGetFunction(&fn, mod, symbol),
           "cuModuleGetFunction");
  return fn;
}

/// Opt every external (non-Jit) function into the device's full
/// dynamic-smem budget up-front.  The attribute is purely a permission
/// cap — raising it doesn't change the carveout or per-launch smem
/// cost, so we do it eagerly here rather than per-launch.  No-op if
/// there are no external kernels.
static void optInExternalSmemMax(const std::vector<KernelMeta> &kernels,
                                   const std::vector<CUfunction> &funcs) {
  bool anyExternal = false;
  for (const auto &k : kernels)
    if (k.kind != KernelKind::Jit) { anyExternal = true; break; }
  if (!anyExternal)
    return;

  CUdevice dev = 0;
  checkCu(cuCtxGetDevice(&dev), "cuCtxGetDevice");
  int maxOptIn = 0;
  checkCu(cuDeviceGetAttribute(
              &maxOptIn,
              CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev),
           "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)");
  for (size_t i = 0; i < funcs.size(); ++i) {
    if (kernels[i].kind == KernelKind::Jit) continue;
    checkCu(cuFuncSetAttribute(
                funcs[i],
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                maxOptIn),
             "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)");
  }
}

/// Per-kernel-kind I/O arity check.  External cs_rank kernels have a
/// fixed signature `(T_in, T_out)` — the kernel signature is set in
/// stone by `kernels/cs_rank.cu`, so we know the wiring is wrong (not
/// just unusual) the moment we see any other shape.  Static property
/// of the graph, so done at construction.
static void validateKernelIO(const std::vector<KernelMeta> &kernels,
                               const std::vector<std::vector<int>> &kernelInputBufs,
                               const std::vector<std::vector<int>> &kernelOutputBufs) {
  for (size_t i = 0; i < kernels.size(); ++i) {
    const auto &k    = kernels[i];
    const size_t nIn = kernelInputBufs[i].size();
    const size_t nOut = kernelOutputBufs[i].size();
    switch (k.kind) {
      case KernelKind::Jit:
        // JIT kernels can have any arity — they're whatever the MLIR
        // pipeline emitted.
        break;
      case KernelKind::ExtCsRankF32:
      case KernelKind::ExtCsRankF64:
        if (nIn != 1 || nOut != 1)
          throw std::runtime_error(
              "kun_cuda::Executable: cs_rank kernel '" + k.kernelName +
              "' must have exactly 1 input and 1 output (have " +
              std::to_string(nIn) + " / " + std::to_string(nOut) + ")");
        break;
    }
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Executable
//===----------------------------------------------------------------------===//

Executable::Executable(ExecutableData &&data) : data_(std::move(data)) {
  // Require a primary context to already exist on the calling thread —
  // the caller's job to set one up (e.g. by allocating any device memory
  // through cupy / cudaMalloc).
  CUcontext cur = nullptr;
  checkCu(cuCtxGetCurrent(&cur), "cuCtxGetCurrent");
  if (!cur)
    throw std::runtime_error(
        "kun_cuda::Executable: no current CUDA context.  Initialise the "
        "driver first (e.g. allocate any device memory via cupy or "
        "cudaMalloc) before constructing an Executable.");
  if (data_.kernels.empty())
    throw std::runtime_error(
        "kun_cuda::Executable: ExecutableData has no kernels");
  if (data_.graphInputs.empty())
    throw std::runtime_error(
        "kun_cuda::Executable: graph_inputs must be non-empty");
  if (data_.graphOutputs.empty())
    throw std::runtime_error(
        "kun_cuda::Executable: graph_outputs must be non-empty");

  // ── Build the runtime plan ───────────────────────────────────────
  BufTable tbl  = buildBufferIndices(data_.graphInputs, data_.graphOutputs,
                                       data_.kernels);
  KernelIO kio  = resolveKernelIO(data_.kernels, tbl);
  validateGraph(data_.kernels, data_.graphOutputs, tbl, kio);
  std::vector<int> order = topoSort(kio, static_cast<int>(data_.kernels.size()));
  SlotPlan slots = planSlots(order, tbl, kio);

  plan_ = std::make_unique<GraphPlan>();
  plan_->numBuffers          = tbl.numBuffers;
  plan_->numGraphInputs      = tbl.numGraphInputs;
  plan_->numGraphOutputs     = tbl.numGraphOutputs;
  plan_->graphInputIdx       = std::move(tbl.graphInputIdx);
  plan_->graphOutputIdx      = std::move(tbl.graphOutputIdx);
  plan_->kernelInputBufs     = std::move(kio.kernelInputBufs);
  plan_->kernelOutputBufs    = std::move(kio.kernelOutputBufs);
  plan_->producerKernel      = std::move(kio.producerKernel);
  plan_->launchOrder         = std::move(order);
  plan_->intermediateBufToSlot = std::move(slots.intermediateBufToSlot);
  plan_->peakIntermediateSlots = slots.peakIntermediateSlots;

  // ── Per-kernel I/O arity validation ──────────────────────────────
  // Catches mis-wired external kernels (which have a fixed signature)
  // at construction time, well before the launch path.
  validateKernelIO(data_.kernels,
                    plan_->kernelInputBufs, plan_->kernelOutputBufs);

  // ── Load cubin(s) + resolve every kernel symbol ──────────────────
  loadJitCubin(data_, cuModule_);
  loadCsRankPtxIfNeeded(data_.kernels, csRankModule_);

  cuFuncs_.resize(data_.kernels.size(), nullptr);
  for (size_t i = 0; i < data_.kernels.size(); ++i) {
    cuFuncs_[i] = resolveOneKernelSymbol(data_.kernels[i],
                                          cuModule_, csRankModule_);
  }

  // ── Opt external kernels into the device's full dynamic smem cap ──
  optInExternalSmemMax(data_.kernels, cuFuncs_);
}

Executable::~Executable() {
  // Best-effort cleanup; we deliberately don't propagate driver errors
  // out of a destructor.
  freeSlotPool();
  if (cuModule_)
    cuModuleUnload(cuModule_);
  if (csRankModule_)
    cuModuleUnload(csRankModule_);
}

void Executable::freeSlotPool() {
  for (uintptr_t p : slotBufs_)
    if (p) cuMemFree(static_cast<CUdeviceptr>(p));
  slotBufs_.clear();
  cachedT_ = -1;
  cachedS_ = -1;
}

void Executable::ensureSlotPool(int64_t timeLength, int64_t numStocks) {
  if (timeLength == cachedT_ && numStocks == cachedS_ &&
      static_cast<int>(slotBufs_.size()) == plan_->peakIntermediateSlots)
    return;
  freeSlotPool();
  if (plan_->peakIntermediateSlots == 0) {
    cachedT_ = timeLength;
    cachedS_ = numStocks;
    return;
  }
  size_t bytesPerSlot = static_cast<size_t>(timeLength) *
                          static_cast<size_t>(numStocks) * sizeof(float);
  slotBufs_.resize(plan_->peakIntermediateSlots, 0);
  for (int i = 0; i < plan_->peakIntermediateSlots; ++i) {
    CUdeviceptr p = 0;
    checkCu(cuMemAlloc(&p, bytesPerSlot), "cuMemAlloc(intermediate slot)");
    slotBufs_[i] = static_cast<uintptr_t>(p);
  }
  cachedT_ = timeLength;
  cachedS_ = numStocks;
}

//===----------------------------------------------------------------------===//
// Out-of-line plan accessors (header forward-declares GraphPlan)
//===----------------------------------------------------------------------===//

const std::vector<int> &Executable::launchOrder() const noexcept {
  return plan_->launchOrder;
}
int Executable::numBuffers() const noexcept { return plan_->numBuffers; }
int Executable::peakIntermediateSlots() const noexcept {
  return plan_->peakIntermediateSlots;
}

void Executable::launchOnStream(
    Executor *exec,
    int64_t timeLength, int64_t numStocks,
    const std::vector<std::pair<std::string, uintptr_t>> &args,
    int64_t mask,
    int minChunkWarmupFactor,
    double smFillFactor) {
  if (!exec)
    throw std::runtime_error(
        "kun_cuda::launchOnStream: Executor pointer is null");
  CUstream stream      = exec->stream();
  int devMaxSmemBytes  = exec->devMaxSmemBytes();
  int numSMs           = exec->numSMs();
  // ── Shape sanity (kernel signature is i32 across the board) ─────
  if (timeLength > std::numeric_limits<int32_t>::max() ||
      numStocks  > std::numeric_limits<int32_t>::max() ||
      timeLength < 0 || numStocks < 0)
    throw std::runtime_error(
        "kun_cuda::launchOnStream: time_length / num_stocks out of i32 "
        "range (kernel signature uses i32, i32)");
  if (mask < 0 || (timeLength > 0 && mask >= timeLength))
    throw std::runtime_error(
        "kun_cuda::launchOnStream: mask must be in [0, time_length), got "
        + std::to_string(mask) + " for time_length="
        + std::to_string(timeLength));
  if (data_.warpsPerCta <= 0)
    throw std::runtime_error(
        "kun_cuda::launchOnStream: warps_per_cta is " +
        std::to_string(data_.warpsPerCta));

  // ── Grow / reuse the intermediate slot pool for this shape ───────
  ensureSlotPool(timeLength, numStocks);

  // ── Map user args + slot pool into a flat buffer-index → ptr ─────
  const std::vector<uintptr_t> bufPtrs =
      resolveBufferPointers(*plan_, data_, args, slotBufs_);

  // ── Per-launch i32 scalars.  time_length / num_stocks / mask are
  //    shared across every kernel; chunk_size / warmup vary per kernel
  //    (chunk_size is derived from per-kernel unreliableCount). ──────
  int32_t timeLenI32   = static_cast<int32_t>(timeLength);
  int32_t numStocksI32 = static_cast<int32_t>(numStocks);
  int32_t maskI32      = static_cast<int32_t>(mask);

  for (int kIdx : plan_->launchOrder) {
    const auto &ins  = plan_->kernelInputBufs[kIdx];
    const auto &outs = plan_->kernelOutputBufs[kIdx];
    const auto &meta = data_.kernels[kIdx];

    std::vector<CUdeviceptr> ptrs;
    ptrs.reserve(ins.size() + outs.size());
    for (int b : ins)  ptrs.push_back(static_cast<CUdeviceptr>(bufPtrs[b]));
    for (int b : outs) ptrs.push_back(static_cast<CUdeviceptr>(bufPtrs[b]));

    if (meta.kind == KernelKind::Jit) {
      // JIT argv: (i32 T, i32 S, i32 mask, i32 chunk_size, i32 warmup,
      //            ptrs...).  Chunk plan is per-kernel because each
      //            kernel has its own unreliableCount.
      ChunkPlan plan = computeChunkPlan(
          timeLength, numStocks, data_.warpsPerCta, data_.vectorSize,
          meta.unreliableCount, mask, minChunkWarmupFactor,
          smFillFactor, numSMs);
      int32_t chunkSizeI32 = static_cast<int32_t>(plan.chunkSize);
      int32_t warmupI32    = static_cast<int32_t>(meta.unreliableCount);

      std::vector<void *> argPtrs;
      argPtrs.reserve(5 + ptrs.size());
      argPtrs.push_back(&timeLenI32);
      argPtrs.push_back(&numStocksI32);
      argPtrs.push_back(&maskI32);
      argPtrs.push_back(&chunkSizeI32);
      argPtrs.push_back(&warmupI32);
      for (auto &p : ptrs) argPtrs.push_back(&p);

      launchJitKernel(cuFuncs_[kIdx], numStocks,
                       data_.warpsPerCta, data_.vectorSize,
                       plan.numChunks, argPtrs.data(), stream);
    } else {
      // External cs_rank argv unchanged: (i32 T, i32 S, ptrs...).  These
      // kernels are cross-sectional, time-major, and don't multi-chunk
      // along time — the mask / chunk_size / warmup scalars don't apply.
      std::vector<void *> argPtrs;
      argPtrs.reserve(2 + ptrs.size());
      argPtrs.push_back(&timeLenI32);
      argPtrs.push_back(&numStocksI32);
      for (auto &p : ptrs) argPtrs.push_back(&p);
      launchExtCsRankKernel(cuFuncs_[kIdx], meta.kind, meta.kernelName,
                              timeLength, numStocks,
                              devMaxSmemBytes, smFillFactor, numSMs,
                              argPtrs.data(), stream);
    }
  }
}

//===----------------------------------------------------------------------===//
// Executor — thin CUstream wrapper, mirrors the CPU `kun::Executor` shape.
//===----------------------------------------------------------------------===//

namespace {
/// Query the current CUcontext's device for a single integer attribute.
/// Returns 0 if no context is current — callers gate use on 0 == "unknown".
int queryDevAttr(CUdevice_attribute attr) {
  CUcontext cur = nullptr;
  if (cuCtxGetCurrent(&cur) != CUDA_SUCCESS || !cur) return 0;
  CUdevice dev = 0;
  if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) return 0;
  int v = 0;
  if (cuDeviceGetAttribute(&v, attr, dev) != CUDA_SUCCESS) return 0;
  return v;
}
} // namespace

Executor::Executor()
    : stream_(nullptr),
      devMaxSmemBytes_(
          queryDevAttr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)),
      numSMs_(queryDevAttr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)) {}
Executor::Executor(CUstream stream)
    : stream_(stream),
      devMaxSmemBytes_(
          queryDevAttr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)),
      numSMs_(queryDevAttr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)) {}
Executor::~Executor() = default;

void Executor::runGraph(
    Executable &exe, int64_t timeLength, int64_t numStocks,
    const std::vector<std::pair<std::string, uintptr_t>> &args,
    int64_t mask, int minChunkWarmupFactor, double smFillFactor) {
  exe.launchOnStream(this, timeLength, numStocks, args,
                      mask, minChunkWarmupFactor, smFillFactor);
}

void Executor::synchronize() {
  checkCu(cuStreamSynchronize(stream_), "cuStreamSynchronize");
}

} // namespace kun_cuda
