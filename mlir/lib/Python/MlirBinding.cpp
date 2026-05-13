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
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unique_ptr.h>

#include "PyModule.h"     // shared MLIRContext + ModuleOp wrapper
#include "IRBuilder.h"    // nanobind class for programmatic kunir construction
#include "dlpack.h"       // vendored DLPack ABI (consumer-only)

#include "KunCuda/Runtime.h"
#include "KunGpu/PtxBackend.h"

#include "llvm/ADT/StringRef.h"

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
                                  const nb::object &streamArg) {
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

  // ── dtype: kDLFloat, 32-bit, 1 lane ─────────────────────────────────
  if (t.dtype.code != kDLFloat || t.dtype.bits != 32 || t.dtype.lanes != 1)
    throw std::runtime_error(
        "'" + paramName + "' DLPack dtype is (code=" +
        std::to_string(static_cast<int>(t.dtype.code)) +
        ", bits=" + std::to_string(static_cast<int>(t.dtype.bits)) +
        ", lanes=" + std::to_string(static_cast<int>(t.dtype.lanes)) +
        ") — need float32 (kDLFloat, 32, 1)");

  // ── strides: NULL = row-major contiguous; else validate.  DLPack
  //    strides are in *elements*, not bytes — convert before checking.
  if (t.strides) {
    int64_t sb[2] = {t.strides[0] * 4, t.strides[1] * 4};
    requireRowMajorContiguous2D(paramName, t.shape, sb, /*elemSize=*/4);
  }

  // ── data pointer (apply byte_offset before handing to kernel) ───────
  uintptr_t ptr = reinterpret_cast<uintptr_t>(t.data) + t.byte_offset;
  if (ptr == 0)
    throw std::runtime_error(
        "'" + paramName + "' DLPack data pointer is null");

  return CudaArrayInfo{ptr, t.shape[0], t.shape[1]};
}

/// Walk the user's {name → cuda_array} dict, validate that every named
/// arg is present and that all arrays share the same (timeLength,
/// numStocks).  Returns the common (T, S) plus a flat list of (name, ptr)
/// pairs.  Anything binding-side (CAI parsing, dtype/ndim/shape checks)
/// happens here so the runtime stays a thin launcher.
struct CollectedArgs {
  int64_t timeLength;
  int64_t numStocks;
  std::vector<std::pair<std::string, uintptr_t>> args;
};

static CollectedArgs collectArgs(const kun_cuda::Executable &exe,
                                   nb::dict pyArgs,
                                   const nb::object &streamArg) {
  // Graph inputs come first, then outputs — same as the buffer-table
  // layout the runtime expects.
  std::vector<std::string> ordered;
  ordered.reserve(exe.graphInputs().size() + exe.graphOutputs().size());
  for (auto &n : exe.graphInputs())  ordered.push_back(n);
  for (auto &n : exe.graphOutputs()) ordered.push_back(n);
  if (ordered.empty())
    throw std::runtime_error("launch: kernel has no I/O arguments");

  CollectedArgs out;
  out.args.reserve(ordered.size());

  // Reject extras up-front so the error message points at the offending
  // name (the per-name loop below would otherwise just complain about a
  // missing graph_input/output, which is misleading when the real issue
  // is a typo'd key).
  if (pyArgs.size() > ordered.size()) {
    for (auto kv : pyArgs) {
      std::string key = nb::cast<std::string>(kv.first);
      bool known = false;
      for (auto &n : ordered) if (n == key) { known = true; break; }
      if (!known) {
        std::string expected;
        for (size_t j = 0; j < ordered.size(); ++j) {
          if (j) expected += ", ";
          expected += ordered[j];
        }
        throw std::runtime_error(
            "launch: unexpected argument '" + key +
            "' (kernel expects: " + expected + ")");
      }
    }
  }

  bool first = true;
  for (size_t i = 0; i < ordered.size(); ++i) {
    const std::string &name = ordered[i];

    nb::object key = nb::str(name.c_str());
    if (!pyArgs.contains(key)) {
      std::string expected;
      for (size_t j = 0; j < ordered.size(); ++j) {
        if (j) expected += ", ";
        expected += ordered[j];
      }
      throw std::runtime_error("launch: missing argument '" + name +
                                "' (kernel expects: " + expected + ")");
    }
    CudaArrayInfo info = readDLPack(pyArgs[key], name, streamArg);
    if (first) {
      out.timeLength = info.timeLength;
      out.numStocks  = info.numStocks;
      first = false;
    } else if (info.timeLength != out.timeLength ||
                 info.numStocks  != out.numStocks) {
      std::stringstream ss;
      ss << "launch: shape mismatch on '" << name << "': expected ("
         << out.timeLength << ", " << out.numStocks
         << ") matching the first array, got ("
         << info.timeLength << ", " << info.numStocks << ")";
      throw std::runtime_error(ss.str());
    }
    out.args.emplace_back(name, info.ptr);
  }
  return out;
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
            int warpsPerCta) {
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
            });

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
              nb::dict pyArgs) {
            // Thread the executor's stream into __dlpack__(stream=…)
            // so producers (CuPy / PyTorch / JAX / TF) can insert the
            // cross-stream sync needed for data-readiness on our
            // launch stream.
            nb::object streamArg = dlpackStreamArg(e.stream());
            auto c = collectArgs(exe, pyArgs, streamArg);
            e.runGraph(exe, c.timeLength, c.numStocks, c.args);
          },
          nb::arg("exe"), nb::arg("args"),
          "Queue every kernel in `exe` onto this executor's stream.\n"
          "**Asynchronous** — call `.synchronize()` (or otherwise wait\n"
          "on the stream) before reading results back to host.\n"
          "\n"
          "`args` is a {name → cupy_array} dict; names must equal "
          "`exe.input_names ++ exe.output_names`.  Arrays must be "
          "float32, 2-D, shape `(time_length, num_stocks)` (TS layout), "
          "and reside on the GPU.\n"
          "\n"
          "Named to match the CPU executor API "
          "(`KunRunner.runGraph(executor, mod, ...)`).")
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
