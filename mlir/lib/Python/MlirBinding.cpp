//===- MlirBinding.cpp - Python bindings for the kunir → cubin flow ----===//
//
// Exposes:
//   KunMLIR.parse(text)            → ModuleOp     (loads MLIR text)
//   ModuleOp.to_string() / __str__  → str          (dumps the module)
//   KunMLIR.lower_to_ptx(mod, …)   → str          (kunir → PTX, debug only)
//   KunMLIR.compile(mod, …)        → Executable   (kunir → loadable kernel)
//   Executable.launch({name: cupy}) → None         (cuLaunchKernel + sync)
//
// `compile` is the main path; `lower_to_ptx` is for inspecting the
// intermediate PTX text that the upstream `gpu-module-to-binary` pass
// produces (with `format=isa`).  Both go through the same lowering
// pipeline — see PtxBackend.h.
//
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unique_ptr.h>

#include "PyModule.h"     // shared MLIRContext + ModuleOp wrapper
#include "IRBuilder.h"    // nanobind class for programmatic kunir construction
#include "dlpack.h"       // vendored DLPack ABI (consumer-only)

#include "KunCuda/Runtime.h"
#include "KunGpu/PtxBackend.h"

#include "llvm/ADT/StringRef.h"

#include <cuda.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

using kun_mlir_py::PyModule;

namespace {

//===----------------------------------------------------------------------===//
// PTX inspection (debug)
//===----------------------------------------------------------------------===//

static std::string pyLowerToPtx(PyModule &pm, const std::string &gpuArch,
                                  const std::string &targetTriple,
                                  const std::string &targetFeatures,
                                  unsigned optLevel,
                                  const std::string &toolkitPath) {
  kungpu::PtxCompileOptions opts;
  if (!gpuArch.empty())        opts.targetCpu      = gpuArch;
  if (!targetTriple.empty())   opts.targetTriple   = targetTriple;
  if (!targetFeatures.empty()) opts.targetFeatures = targetFeatures;
  opts.optLevel    = optLevel;
  opts.toolkitPath = toolkitPath;

  std::string ptx;
  if (failed(kungpu::compileKunIrToPtx(pm.module.get(), opts, ptx)))
    throw std::runtime_error("KunMLIR.lower_to_ptx failed");
  return ptx;
}

//===----------------------------------------------------------------------===//
// nanobind glue: read a Python GPU array via DLPack → device pointer + shape
//===----------------------------------------------------------------------===//

/// Result of reading one GPU array argument.  `ptr` is the device pointer
/// the kernel will consume; `(timeLength, numStocks)` is the resolved
/// 2-D shape used for cross-arg consistency checks.
struct CudaArrayInfo {
  uintptr_t ptr;
  int64_t   timeLength;   ///< shape[0]
  int64_t   numStocks;    ///< shape[1]
};

/// DLPack-spec encoding of the executor's CUDA stream, ready to hand to
/// `obj.__dlpack__(stream=…)`.  The protocol uses int sentinels for the
/// two default streams and the actual `CUstream` pointer otherwise:
///
///   None  ⇒  producer chooses (no sync)
///   1     ⇒  legacy default stream
///   2     ⇒  per-thread default stream
///   other ⇒  CUstream pointer cast to int
///
/// We're never "no sync" — every launch must serialise on the executor's
/// stream — so `stream_ == nullptr` (default-stream executor) maps to 1.
static nb::object dlpackStreamArg(CUstream stream) {
  if (stream == nullptr)
    return nb::int_(1);
  return nb::int_(reinterpret_cast<uintptr_t>(stream));
}

/// Throws if `(shape, stridesBytes)` doesn't describe a C-contiguous
/// 2-D buffer with `elemSize`-byte elements.  `stridesBytes == nullptr`
/// is the "default row-major" case (always contiguous).
static void requireRowMajorContiguous2D(const std::string &paramName,
                                          const int64_t *shape,
                                          const int64_t *stridesBytes,
                                          int64_t elemSize) {
  if (!stridesBytes)
    return;
  const int64_t innerStride = elemSize;
  const int64_t outerStride = elemSize * shape[1];
  if (stridesBytes[0] != outerStride || stridesBytes[1] != innerStride) {
    std::stringstream ss;
    ss << "'" << paramName << "' is not C-contiguous: strides=("
       << stridesBytes[0] << ", " << stridesBytes[1] << ") bytes, "
       << "expected (" << outerStride << ", " << innerStride
       << ") for shape (" << shape[0] << ", " << shape[1] << ")";
    throw std::runtime_error(ss.str());
  }
}

/// Read `__dlpack__(stream=…)` — the cross-framework GPU array protocol
/// implemented by CuPy / PyTorch / JAX / TensorFlow.  Validates every
/// field the kernel relies on and threads the executor's stream so the
/// producer can insert the needed cross-stream sync.
///
/// Memory lifecycle: `__dlpack__()` returns a PyCapsule named "dltensor"
/// owning a `DLManagedTensor`; when the capsule is GC'd, its destructor
/// calls the producer's `deleter`.  We grab the fields we need and let
/// the capsule fall out of scope at function exit — the underlying
/// tensor stays alive because the user is still holding `obj`.
static CudaArrayInfo readDLPack(nb::handle obj, const std::string &paramName,
                                  const nb::object &streamArg,
                                  kun_cuda::Datatype expectedDtype) {
  if (!nb::hasattr(obj, "__dlpack__"))
    throw std::runtime_error(
        "'" + paramName + "' does not implement __dlpack__ — pass a CuPy "
        "ndarray, a PyTorch CUDA tensor, a JAX device array, or any other "
        "object exporting the DLPack protocol.");

  nb::object capsule = obj.attr("__dlpack__")(nb::arg("stream") = streamArg);
  void *raw = PyCapsule_GetPointer(capsule.ptr(), "dltensor");
  if (!raw) {
    PyErr_Clear();
    throw std::runtime_error(
        "'" + paramName + "' __dlpack__() did not return a PyCapsule named "
        "'dltensor' (consumed capsule?  wrong producer?)");
  }
  const DLManagedTensor *mt = reinterpret_cast<const DLManagedTensor *>(raw);
  const DLTensor &t = mt->dl_tensor;

  // ── device: only CUDA (managed counts as CUDA-addressable) ──────────
  if (t.device.device_type != kDLCUDA &&
      t.device.device_type != kDLCUDAManaged)
    throw std::runtime_error(
        "'" + paramName + "' is on DLPack device type " +
        std::to_string(static_cast<int>(t.device.device_type)) +
        " — only CUDA (=2) and CUDAManaged (=13) are supported");

  // ── ndim ────────────────────────────────────────────────────────────
  if (t.ndim != 2)
    throw std::runtime_error(
        "'" + paramName + "' must be 2-D (got " +
        std::to_string(t.ndim) + "-D)");

  // ── dtype: kDLFloat, matches executable's element type ──────────────
  const uint8_t expectedBits =
      expectedDtype == kun_cuda::Datatype::Double ? 64 : 32;
  if (t.dtype.code != kDLFloat || t.dtype.bits != expectedBits ||
      t.dtype.lanes != 1)
    throw std::runtime_error(
        "'" + paramName + "' DLPack dtype is (code=" +
        std::to_string(static_cast<int>(t.dtype.code)) +
        ", bits=" + std::to_string(static_cast<int>(t.dtype.bits)) +
        ", lanes=" + std::to_string(static_cast<int>(t.dtype.lanes)) +
        ") — kernel expects float" + std::to_string(expectedBits) +
        " (kDLFloat, " + std::to_string(expectedBits) + ", 1)");

  // ── strides: NULL = row-major contiguous; else validate.  DLPack
  //    strides are in *elements*, not bytes — convert before checking.
  const int64_t elemBytes = static_cast<int64_t>(kun_cuda::bytesPerElem(expectedDtype));
  if (t.strides) {
    int64_t sb[2] = {t.strides[0] * elemBytes, t.strides[1] * elemBytes};
    requireRowMajorContiguous2D(paramName, t.shape, sb, elemBytes);
  }

  // ── data pointer (apply byte_offset before handing to kernel) ───────
  uintptr_t ptr = reinterpret_cast<uintptr_t>(t.data) + t.byte_offset;
  if (ptr == 0)
    throw std::runtime_error(
        "'" + paramName + "' DLPack data pointer is null");

  return CudaArrayInfo{ptr, t.shape[0], t.shape[1]};
}

/// Reject keys in `pyDict` that are not in `expectedNames`.  Used so the
/// error message points at the offending name instead of complaining
/// about a different missing key down the loop.
///
/// Fast path: when `pyDict.size() == expectedNames.size()` we skip the
/// per-key scan.  Either every expected name is present (no unexpected
/// key by definition) or one is missing — in the latter case the
/// downstream missing-key check still raises with a correct (if less
/// precise) error.
static void rejectUnexpectedKeys(const nb::dict &pyDict,
                                   const std::vector<std::string> &expectedNames,
                                   const char *kind) {
  if (pyDict.size() == expectedNames.size())
    return;
  for (auto kv : pyDict) {
    std::string key = nb::cast<std::string>(kv.first);
    bool known = false;
    for (auto &n : expectedNames) if (n == key) { known = true; break; }
    if (known) continue;
    std::string expected;
    for (size_t j = 0; j < expectedNames.size(); ++j) {
      if (j) expected += ", ";
      expected += expectedNames[j];
    }
    throw std::runtime_error(std::string("runGraph: unexpected ") + kind +
                              " '" + key + "' (expected: " + expected + ")");
  }
}

/// Walk `pyInputs` in `exe.graphInputs()` order, validate that every name
/// is present and that all arrays share the input shape (timeLength,
/// numStocks).  Caller specifies the required `timeLength` via
/// `requiredTimeLength` (== start + length); a value of -1 means "infer
/// from the first input" and the binding will treat that as the locked
/// shape.
struct CollectedInputs {
  int64_t timeLength;
  int64_t numStocks;
  std::vector<std::pair<std::string, uintptr_t>> args;
};

static CollectedInputs collectInputs(const kun_cuda::Executable &exe,
                                        const nb::dict &pyInputs,
                                        const nb::object &streamArg,
                                        int64_t requiredTimeLength) {
  const auto &inputNames = exe.graphInputs();
  rejectUnexpectedKeys(pyInputs, inputNames, "input");

  CollectedInputs out;
  out.timeLength = requiredTimeLength;
  out.numStocks  = -1;
  out.args.reserve(inputNames.size());

  for (const std::string &name : inputNames) {
    nb::object key = nb::str(name.c_str());
    if (!pyInputs.contains(key))
      throw std::runtime_error("runGraph: missing input '" + name + "'");
    CudaArrayInfo info = readDLPack(pyInputs[key], name, streamArg, exe.dtype());

    if (out.timeLength < 0) {
      out.timeLength = info.timeLength;
      out.numStocks  = info.numStocks;
    } else if (info.timeLength != out.timeLength ||
                 (out.numStocks >= 0 && info.numStocks != out.numStocks)) {
      std::stringstream ss;
      ss << "runGraph: input '" << name << "' has shape ("
         << info.timeLength << ", " << info.numStocks
         << "), expected (" << out.timeLength << ", "
         << (out.numStocks < 0 ? info.numStocks : out.numStocks) << ")";
      throw std::runtime_error(ss.str());
    }
    if (out.numStocks < 0)
      out.numStocks = info.numStocks;
    out.args.emplace_back(name, info.ptr);
  }
  return out;
}

/// Allocate a CUDA device buffer of `T*S` elements (`sizeof(elem) =
/// bytesPerElem(dt)`) and wrap it in an `nb::ndarray<>` (no framework
/// annotation) owning the allocation via a capsule.  Lifetime is tied
/// to the Python object: when the array's refcount drops to zero, the
/// capsule destructor frees via `cuMemFree`.
static nb::ndarray<> allocOwnedCudaArray2D(int64_t T, int64_t S,
                                              kun_cuda::Datatype dt) {
  const size_t elemBytes = kun_cuda::bytesPerElem(dt);
  size_t total = static_cast<size_t>(T) * static_cast<size_t>(S);
  CUdeviceptr p = 0;
  CUresult r = cuMemAlloc(&p, total * elemBytes);
  if (r != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorString(r, &msg);
    throw std::runtime_error(std::string("runGraph: cuMemAlloc failed: ") +
                              (msg ? msg : "(unknown)"));
  }
  nb::capsule owner(reinterpret_cast<void *>(p), [](void *q) noexcept {
    cuMemFree(reinterpret_cast<CUdeviceptr>(q));
  });
  CUdevice dev = 0;
  cuCtxGetDevice(&dev);
  size_t shape[2] = {static_cast<size_t>(T), static_cast<size_t>(S)};
  nb::dlpack::dtype npDtype =
      dt == kun_cuda::Datatype::Double ? nb::dtype<double>()
                                         : nb::dtype<float>();
  return nb::ndarray<>(reinterpret_cast<void *>(p), /*ndim=*/2, shape, owner,
                        /*strides=*/nullptr,
                        /*dtype=*/npDtype,
                        /*device_type=*/nb::device::cuda::value,
                        /*device_id=*/static_cast<int>(dev));
}

/// Walk `exe.graphOutputs()` in order: for each name, either pick the
/// caller-allocated buffer out of `pyOutputs` (validating shape) or
/// allocate a fresh CUDA buffer.  Appends `(name, devicePtr)` to `args`
/// and returns a `{name: ndarray}` dict of every output that Python
/// will see.
///
/// When `pyOutputs.is_none()` we short-circuit:  no dict cast, no
/// rejectUnexpectedKeys, no per-name `contains` probe — every output
/// is auto-allocated.  This is the common case (caller doesn't pre-
/// allocate outputs) and keeps it tight.
static nb::dict collectOutputs(
    const kun_cuda::Executable &exe,
    nb::object pyOutputs, int64_t length, int64_t numStocks,
    const nb::object &streamArg,
    std::vector<std::pair<std::string, uintptr_t>> &args) {
  const auto &outputNames = exe.graphOutputs();
  args.reserve(args.size() + outputNames.size());

  // Start with a null-PyObject* `nb::dict` — `nb::handle::inc_ref()` /
  // `dec_ref()` are `Py_XINCREF`/`Py_XDECREF` so it's safe to hold, and
  // we skip the `PyDict_New()` that bare `nb::dict()` would do.  Only
  // populate + extras-check when the caller passed a real dict; then
  // the handle's `operator bool()` doubles as the "user gave us
  // outputs" flag.
  nb::dict userOutputs = nb::steal<nb::dict>(nb::handle());
  if (!pyOutputs.is_none()) {
    userOutputs = nb::cast<nb::dict>(pyOutputs);
    rejectUnexpectedKeys(userOutputs, outputNames, "output");
  }

  nb::dict ret;
  for (const std::string &name : outputNames) {
    nb::object key = nb::str(name.c_str());
    uintptr_t base;
    if (userOutputs && userOutputs.contains(key)) {
      CudaArrayInfo info =
          readDLPack(userOutputs[key], name, streamArg, exe.dtype());
      if (info.timeLength != length || info.numStocks != numStocks) {
        std::stringstream ss;
        ss << "runGraph: output '" << name << "' has shape ("
           << info.timeLength << ", " << info.numStocks
           << "), expected (" << length << ", " << numStocks << ")";
        throw std::runtime_error(ss.str());
      }
      base = info.ptr;
      ret[key] = userOutputs[key];
    } else {
      nb::ndarray<> arr =
          allocOwnedCudaArray2D(length, numStocks, exe.dtype());
      base = reinterpret_cast<uintptr_t>(arr.data());
      ret[key] = nb::cast(std::move(arr));
    }
    args.emplace_back(name, base);
  }
  return ret;
}

/// Parse one Python `external_kernels=[...]` entry into a KernelMeta.
/// Expected dict shape:
///   {"name": str, "kind": str, "inputs": [str...], "outputs": [str...]}
/// where `kind` is one of "cs_rank_f32" / "cs_rank_f64".
static kun_cuda::KernelMeta parseExternalKernel(nb::handle obj) {
  nb::dict d = nb::cast<nb::dict>(obj);
  kun_cuda::KernelMeta km;
  km.kernelName = nb::cast<std::string>(d["name"]);
  std::string kind = nb::cast<std::string>(d["kind"]);
  if (kind == "cs_rank_f32")
    km.kind = kun_cuda::KernelKind::ExtCsRankF32;
  else if (kind == "cs_rank_f64")
    km.kind = kun_cuda::KernelKind::ExtCsRankF64;
  else
    throw std::runtime_error(
        "KunMLIR.compile: unknown external kernel kind '" + kind +
        "' (supported: cs_rank_f32, cs_rank_f64)");
  nb::iterable inputs  = nb::cast<nb::iterable>(d["inputs"]);
  nb::iterable outputs = nb::cast<nb::iterable>(d["outputs"]);
  for (nb::handle n : inputs)
    km.inputNames.push_back(nb::cast<std::string>(n));
  for (nb::handle n : outputs)
    km.outputNames.push_back(nb::cast<std::string>(n));
  return km;
}

static std::unique_ptr<kun_cuda::Executable>
pyCompile(PyModule &pm,
            const std::vector<std::string> &graphInputs,
            const std::vector<std::string> &graphOutputs,
            const std::string &gpuArch,
            const std::string &targetTriple,
            const std::string &targetFeatures, unsigned optLevel,
            const std::string &toolkitPath,
            nb::list externalKernels,
            int warpsPerCta,
            nb::dict outputUnreliable) {
  if (graphInputs.empty())
    throw std::runtime_error(
        "KunMLIR.compile: graph_inputs cannot be empty");
  if (graphOutputs.empty())
    throw std::runtime_error(
        "KunMLIR.compile: graph_outputs cannot be empty");

  kungpu::PtxCompileOptions opts;
  if (!gpuArch.empty())        opts.targetCpu      = gpuArch;
  if (!targetTriple.empty())   opts.targetTriple   = targetTriple;
  if (!targetFeatures.empty()) opts.targetFeatures = targetFeatures;
  opts.optLevel    = optLevel;
  opts.toolkitPath = toolkitPath;

  kun_cuda::ExecutableData data;
  if (failed(kungpu::compileKunIrToExecutable(pm.module.get(), opts, data)))
    throw std::runtime_error("KunMLIR.compile failed");

  // Append external (pre-compiled, runtime-dispatched) kernels.  The
  // MLIR pipeline never saw them; they're fabricated here from the
  // descriptor list the Python frontend collected.
  for (nb::handle obj : externalKernels)
    data.kernels.push_back(parseExternalKernel(obj));

  if (data.kernels.empty())
    throw std::runtime_error(
        "KunMLIR.compile: no kernels (neither MLIR-emitted nor "
        "external) — refusing to build an empty Executable");

  // No JIT kernels → `compileKunIrToExecutable` left warpsPerCta at
  // its default of 1.  Override with the caller-supplied value so the
  // external launch path's blockDim is right.  When there are JIT
  // kernels they fix warpsPerCta via their kungpu.target_spec, and we
  // trust that over the parameter (and ignore the parameter).
  bool anyJit = false;
  for (const auto &k : data.kernels)
    if (k.kind == kun_cuda::KernelKind::Jit) { anyJit = true; break; }
  if (!anyJit) {
    if (warpsPerCta <= 0)
      throw std::runtime_error(
          "KunMLIR.compile: warps_per_cta must be positive when every "
          "kernel is external; got " + std::to_string(warpsPerCta));
    data.warpsPerCta = warpsPerCta;
  }

  // Graph topology is a runtime concern — fill it in here, just before
  // handing off to Executable's ctor (which validates + plans).
  data.graphInputs  = graphInputs;
  data.graphOutputs = graphOutputs;
  for (auto item : outputUnreliable) {
    auto name = nb::cast<std::string>(item.first);
    auto val  = nb::cast<int64_t>(item.second);
    data.outputUnreliable[name] = val;
  }
  return std::make_unique<kun_cuda::Executable>(std::move(data));
}

} // namespace

NB_MODULE(KunMLIR, m) {
  m.doc() = "Bindings for the KunQuant MLIR compiler (kunir → PTX → CUBIN "
             "→ launch).";

  // Programmatic kunir construction (Value/Type opaque wrappers, IRBuilder).
  kun_mlir_py::registerIRBuilder(m);

  nb::class_<PyModule>(m, "ModuleOp")
      .def("to_string", &PyModule::toString,
            "Return the textual MLIR form of the module.")
      .def("__str__",  &PyModule::toString)
      .def("__repr__", [](const PyModule &m) {
        return "<KunMLIR.ModuleOp>\n" + m.toString();
      });

  m.def("parse", &PyModule::parse, nb::arg("text"),
         "Parse an MLIR text fragment into a ModuleOp.");

  m.def("lower_to_ptx", &pyLowerToPtx,
         nb::arg("module"),
         nb::arg("gpu_arch")       = "sm_80",
         nb::arg("target_triple")  = "nvptx64-nvidia-cuda",
         nb::arg("target_features") = "",
         nb::arg("opt_level")      = 3u,
         nb::arg("toolkit_path")   = "",
         "Lower kunir → PTX text via the upstream `gpu-module-to-binary` "
         "pass with `format=isa`.  Debug / inspection only — the main "
         "compile path goes straight to cubin.");

  nb::class_<kun_cuda::Executable>(m, "Executable")
      .def_prop_ro("input_names",   &kun_cuda::Executable::graphInputs,
            "Graph-level input names — match this against the keys of the "
            "args dict you pass to launch().")
      .def_prop_ro("output_names",  &kun_cuda::Executable::graphOutputs,
            "Graph-level output names — match this against the keys of the "
            "args dict you pass to launch().")
      .def_prop_ro("warps_per_cta", &kun_cuda::Executable::warpsPerCta)
      .def_prop_ro("vector_size",   &kun_cuda::Executable::vectorSize)
      .def_prop_ro("num_kernels",
            [](const kun_cuda::Executable &e) {
              return e.numKernels();
            })
      .def_prop_ro("kernel_names",
            [](const kun_cuda::Executable &e) {
              std::vector<std::string> r;
              r.reserve(e.data().kernels.size());
              for (auto &km : e.data().kernels)
                r.push_back(km.kernelName);
              return r;
            })
      .def_prop_ro("launch_order",  &kun_cuda::Executable::launchOrder,
            "Topo-sorted indices into kernel_names; the order kernels run "
            "on the single CUDA stream.")
      .def_prop_ro("peak_intermediate_slots",
            &kun_cuda::Executable::peakIntermediateSlots,
            "Number of intermediate buffers allocated by the runtime — "
            "shape `(time_length, num_stocks)` each.")
      .def_prop_ro("num_buffers",   &kun_cuda::Executable::numBuffers)
      .def_prop_ro("cubin",
            [](const kun_cuda::Executable &e) {
              const auto &b = e.data().cubin;
              return nb::bytes(b.data(), b.size());
            })
      .def("getOutputUnreliableCount",
            &kun_cuda::Executable::outputUnreliable,
            nb::rv_policy::reference_internal,
            "Return {output_name: unreliable_count} — leading time steps "
            "of each graph output to drop.");

  // ── Executor ────────────────────────────────────────────────────────
  // Mirrors the CPU `kun::Executor` shape: an opaque object that wraps a
  // CUDA stream and exposes run_graph / synchronize.  Constructor accepts
  // either a raw int (uintptr_t — e.g. the stream's `.ptr` from cupy) or
  // a duck-typed object with a `.ptr` attribute (so passing a
  // `cupy.cuda.Stream` directly Just Works).  None / no arg → default
  // CUDA stream.
  nb::class_<kun_cuda::Executor>(m, "Executor",
        "Wraps a CUDA stream + provides `run_graph(exe, args)` (async) "
        "and `synchronize()`.  Default constructor uses the CUDA default "
        "stream; pass a cupy stream (or its `.ptr` integer) to share one "
        "with caller-managed code.")
      .def("__init__", [](kun_cuda::Executor *self, nb::object stream_arg) {
            uintptr_t ptr = 0;
            if (!stream_arg.is_none()) {
              if (nb::hasattr(stream_arg, "ptr"))
                ptr = nb::cast<uintptr_t>(stream_arg.attr("ptr"));
              else
                ptr = nb::cast<uintptr_t>(stream_arg);
            }
            new (self) kun_cuda::Executor(reinterpret_cast<CUstream>(ptr));
          },
          nb::arg("stream") = nb::none(),
          "Build an Executor.  `stream=None` → default CUDA stream; "
          "otherwise expects either an int (uintptr_t handle) or a "
          "cupy.cuda.Stream-like object exposing `.ptr`.")
      .def_prop_ro("stream",
          [](const kun_cuda::Executor &e) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(e.stream());
          },
          "Raw stream handle as an int (0 ↔ CUDA default stream).")
      .def("runGraph",
          [](kun_cuda::Executor &e, kun_cuda::Executable &exe,
              nb::dict pyInputs, int64_t cur_time, int64_t length,
              nb::object pyOutputs, int64_t mask,
              int minChunkWarmupFactor, double smFillFactor) -> nb::dict {
            if (cur_time != 0)
              throw std::runtime_error(
                  "runGraph: cur_time != 0 not supported on GPU");
            if (length < 0)
              throw std::runtime_error("runGraph: length must be >= 0");

            // `length == 0` (default) → auto-infer from the first
            // input's row count; otherwise it's the engine's internal
            // time dim (== input rows == output rows).
            const bool inferLength = (length == 0);

            // Thread the executor's stream into __dlpack__(stream=…)
            // so producers (CuPy / PyTorch / JAX / TF) can insert the
            // cross-stream sync needed for data-readiness on our
            // launch stream.
            nb::object streamArg = dlpackStreamArg(e.stream());
            auto in = collectInputs(exe, pyInputs, streamArg,
                                       inferLength ? -1 : length);
            if (inferLength)
              length = in.timeLength;
            if (mask < 0 || mask >= length)
              throw std::runtime_error(
                  "runGraph: mask must be in [0, length)");

            // Kernel writes `output[t]` directly for `t ∈ [mask, length)`
            // (kungpu codegen no longer subtracts mask).  Rows `[0, mask)`
            // are left as whatever the user / allocator put there.
            const int64_t timeLength = length;

            // Build the args vector (inputs first, then outputs in
            // exe.graphOutputs order).  Auto-allocates any output the
            // caller didn't pre-allocate; returns the dict that goes
            // back to Python.
            std::vector<std::pair<std::string, uintptr_t>> args =
                std::move(in.args);
            nb::dict ret = collectOutputs(exe, pyOutputs, length,
                                            in.numStocks, streamArg, args);

            e.runGraph(exe, timeLength, in.numStocks, args,
                        mask, minChunkWarmupFactor, smFillFactor);
            return ret;
          },
          nb::arg("exe"), nb::arg("inputs"),
          nb::arg("cur_time") = 0, nb::arg("length") = 0,
          nb::arg("outputs") = nb::none(),
          nb::arg("mask") = 0,
          nb::arg("min_chunk_warmup_factor") = 4,
          nb::arg("sm_fill_factor") = 1.5,
          "Queue every kernel in `exe` onto this executor's stream.\n"
          "**Asynchronous** — call `.synchronize()` (or otherwise wait\n"
          "on the stream) before reading results back to host.\n"
          "\n"
          "`inputs` is a {name → cuda_array} dict whose keys must equal\n"
          "`exe.input_names`.  Arrays must be float32, 2-D, shape\n"
          "`(length, num_stocks)` (TS layout), and reside on the GPU.\n"
          "\n"
          "`cur_time` mirrors CPU `kr.runGraph`; GPU only accepts 0.\n"
          "\n"
          "`length` is input/output time dim.  Default 0 ⇒ auto-infer\n"
          "from the first input's row count.\n"
          "\n"
          "`outputs` is an optional {name → cuda_array} dict of\n"
          "caller-allocated output buffers (subset of\n"
          "`exe.output_names`).  Each must have shape `(length,\n"
          "num_stocks)` (same as input).  Names missing from `outputs`\n"
          "are auto-allocated by the binding (float32 CUDA buffers,\n"
          "capsule-owned).  Returns a dict of every output name → its\n"
          "buffer (user-supplied or freshly allocated).\n"
          "\n"
          "`mask` is the warmup-skip on graph outputs: the kernel only\n"
          "writes to output rows `[mask, length)`; rows `[0, mask)` are\n"
          "left untouched (whatever the user / allocator put there).\n"
          "Default 0.\n"
          "\n"
          "`min_chunk_warmup_factor` is the lower bound on "
          "`chunk_size / warmup` — keeps warmup-overlap overhead below "
          "`1 / factor` of total compute.  Default 4 (≤ 25% overhead).\n"
          "`sm_fill_factor` is the target `num_chunks * stock_tiles / "
          "numSMs`.  1.0 just fills the GPU; > 1 leaves scheduler "
          "slack.  Default 1.5.\n"
          "\n"
          "Named to match the CPU executor API "
          "(`KunRunner.runGraph(executor, mod, inputs, cur_time, length)`).")
      .def("synchronize", &kun_cuda::Executor::synchronize,
          "Block until every kernel queued on this stream completes.");

  m.def("compile", &pyCompile,
         nb::arg("module"),
         nb::arg("graph_inputs"),
         nb::arg("graph_outputs"),
         nb::arg("gpu_arch")       = "sm_80",
         nb::arg("target_triple")  = "nvptx64-nvidia-cuda",
         nb::arg("target_features") = "",
         nb::arg("opt_level")      = 3u,
         nb::arg("toolkit_path")   = "",
         nb::arg("external_kernels") = nb::list(),
         nb::arg("warps_per_cta")    = 0,
         nb::arg("output_unreliable") = nb::dict(),
         "Compile a kunir module all the way to a loaded Executable.\n"
         "\n"
         "Pipeline: kunir → LLVM dialect → upstream `gpu-module-to-binary`\n"
         "(format=bin) which handles libdevice linking + LLVM optimization\n"
         "+ PTX emission + ptxas, → cuModuleLoad on the resulting cubin.\n"
         "\n"
         "graph_inputs / graph_outputs name the buffers that flow in/out\n"
         "of the whole kernel graph; everything else produced by the\n"
         "kernels is treated as an intermediate and gets a runtime-managed\n"
         "slot.\n"
         "\n"
         "toolkit_path: optional path to the CUDA toolkit (where\n"
         "libdevice.10.bc and ptxas live).  Empty → search CUDA_HOME /\n"
         "CUDA_PATH / standard install locations.");
}
