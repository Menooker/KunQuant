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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "PyModule.h"     // shared MLIRContext + ModuleOp wrapper
#include "IRBuilder.h"    // pybind class for programmatic kunir construction

#include "KunCuda/Runtime.h"
#include "KunGpu/PtxBackend.h"

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

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
// pybind glue: read CAI dict → kun_cuda::DeviceArray, build name list
//===----------------------------------------------------------------------===//

/// Read CAI from one Python GPU array.  Validates dtype + ndim; shape is
/// returned to the caller for cross-array consistency checks.
struct CudaArrayInfo {
  uintptr_t ptr;
  int64_t timeLength;   ///< shape[0]
  int64_t numStocks;    ///< shape[1]
};

static CudaArrayInfo readCudaArray(py::handle obj,
                                     const std::string &paramName) {
  if (!py::hasattr(obj, "__cuda_array_interface__")) {
    throw std::runtime_error(
        "'" + paramName +
        "' has no __cuda_array_interface__ — pass a CuPy ndarray (or any "
        "GPU array implementing CAI).");
  }
  py::dict cai = obj.attr("__cuda_array_interface__").cast<py::dict>();

  py::tuple data = cai["data"].cast<py::tuple>();
  uintptr_t ptr  = data[0].cast<uintptr_t>();

  std::vector<int64_t> shape;
  for (py::handle s : cai["shape"].cast<py::tuple>())
    shape.push_back(s.cast<int64_t>());
  if (shape.size() != 2) {
    std::stringstream ss;
    ss << "'" << paramName << "' must be 2-D (got " << shape.size() << "-D)";
    throw std::runtime_error(ss.str());
  }

  std::string typestr = cai["typestr"].cast<std::string>();
  if (typestr != "<f4" && typestr != "|f4" && typestr != "=f4") {
    throw std::runtime_error("'" + paramName +
                              "' must be float32 little-endian (typestr "
                              "'<f4'); got '" +
                              typestr + "'");
  }
  return CudaArrayInfo{ptr, shape[0], shape[1]};
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
                                   py::dict pyArgs) {
  std::vector<std::string> ordered;
  ordered.reserve(exe.graphInputs().size() + exe.graphOutputs().size());
  for (auto &n : exe.graphInputs())  ordered.push_back(n);
  for (auto &n : exe.graphOutputs()) ordered.push_back(n);
  if (ordered.empty())
    throw std::runtime_error("launch: kernel has no I/O arguments");

  CollectedArgs out;
  out.args.reserve(ordered.size());

  bool first = true;
  for (const std::string &name : ordered) {
    py::object key = py::str(name);
    if (!pyArgs.contains(key)) {
      std::string expected;
      for (size_t i = 0; i < ordered.size(); ++i) {
        if (i) expected += ", ";
        expected += ordered[i];
      }
      throw std::runtime_error("launch: missing argument '" + name +
                                "' (kernel expects: " + expected + ")");
    }
    CudaArrayInfo info = readCudaArray(pyArgs[key], name);
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
static kun_cuda::KernelMeta parseExternalKernel(py::handle obj) {
  py::dict d = obj.cast<py::dict>();
  kun_cuda::KernelMeta km;
  km.kernelName = d["name"].cast<std::string>();
  std::string kind = d["kind"].cast<std::string>();
  if (kind == "cs_rank_f32")
    km.kind = kun_cuda::KernelKind::ExtCsRankF32;
  else if (kind == "cs_rank_f64")
    km.kind = kun_cuda::KernelKind::ExtCsRankF64;
  else
    throw std::runtime_error(
        "KunMLIR.compile: unknown external kernel kind '" + kind +
        "' (supported: cs_rank_f32, cs_rank_f64)");
  for (py::handle n : d["inputs"].cast<py::iterable>())
    km.inputNames.push_back(n.cast<std::string>());
  for (py::handle n : d["outputs"].cast<py::iterable>())
    km.outputNames.push_back(n.cast<std::string>());
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
            py::list externalKernels,
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
  for (py::handle obj : externalKernels)
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

PYBIND11_MODULE(KunMLIR, m) {
  m.doc() = "Bindings for the KunQuant MLIR compiler (kunir → PTX → CUBIN "
             "→ launch).";

  // Programmatic kunir construction (Value/Type opaque wrappers, IRBuilder).
  kun_mlir_py::registerIRBuilder(m);

  py::class_<PyModule>(m, "ModuleOp")
      .def("to_string", &PyModule::toString,
            "Return the textual MLIR form of the module.")
      .def("__str__",  &PyModule::toString)
      .def("__repr__", [](const PyModule &m) {
        return "<KunMLIR.ModuleOp>\n" + m.toString();
      });

  m.def("parse", &PyModule::parse, py::arg("text"),
         "Parse an MLIR text fragment into a ModuleOp.");

  m.def("lower_to_ptx", &pyLowerToPtx,
         py::arg("module"),
         py::arg("gpu_arch")       = "sm_80",
         py::arg("target_triple")  = "nvptx64-nvidia-cuda",
         py::arg("target_features") = "",
         py::arg("opt_level")      = 3u,
         py::arg("toolkit_path")   = "",
         "Lower kunir → PTX text via the upstream `gpu-module-to-binary` "
         "pass with `format=isa`.  Debug / inspection only — the main "
         "compile path goes straight to cubin.");

  py::class_<kun_cuda::Executable>(m, "Executable")
      .def_property_readonly("input_names",   &kun_cuda::Executable::graphInputs,
            "Graph-level input names — match this against the keys of the "
            "args dict you pass to launch().")
      .def_property_readonly("output_names",  &kun_cuda::Executable::graphOutputs,
            "Graph-level output names — match this against the keys of the "
            "args dict you pass to launch().")
      .def_property_readonly("warps_per_cta", &kun_cuda::Executable::warpsPerCta)
      .def_property_readonly("vector_size",   &kun_cuda::Executable::vectorSize)
      .def_property_readonly("num_kernels",
            [](const kun_cuda::Executable &e) {
              return e.numKernels();
            })
      .def_property_readonly("kernel_names",
            [](const kun_cuda::Executable &e) {
              std::vector<std::string> r;
              r.reserve(e.data().kernels.size());
              for (auto &km : e.data().kernels)
                r.push_back(km.kernelName);
              return r;
            })
      .def_property_readonly("launch_order",  &kun_cuda::Executable::launchOrder,
            "Topo-sorted indices into kernel_names; the order kernels run "
            "on the single CUDA stream.")
      .def_property_readonly("peak_intermediate_slots",
            &kun_cuda::Executable::peakIntermediateSlots,
            "Number of intermediate buffers allocated by the runtime — "
            "shape `(time_length, num_stocks)` each.")
      .def_property_readonly("num_buffers",   &kun_cuda::Executable::numBuffers)
      .def_property_readonly("cubin",
            [](const kun_cuda::Executable &e) {
              const auto &b = e.data().cubin;
              return py::bytes(b.data(), b.size());
            });

  // ── Executor ────────────────────────────────────────────────────────
  // Mirrors the CPU `kun::Executor` shape: an opaque object that wraps a
  // CUDA stream and exposes run_graph / synchronize.  Constructor accepts
  // either a raw int (uintptr_t — e.g. the stream's `.ptr` from cupy) or
  // a duck-typed object with a `.ptr` attribute (so passing a
  // `cupy.cuda.Stream` directly Just Works).  None / no arg → default
  // CUDA stream.
  py::class_<kun_cuda::Executor>(m, "Executor",
        "Wraps a CUDA stream + provides `run_graph(exe, args)` (async) "
        "and `synchronize()`.  Default constructor uses the CUDA default "
        "stream; pass a cupy stream (or its `.ptr` integer) to share one "
        "with caller-managed code.")
      .def(py::init([](py::object stream_arg) {
            uintptr_t ptr = 0;
            if (!stream_arg.is_none()) {
              if (py::hasattr(stream_arg, "ptr"))
                ptr = stream_arg.attr("ptr").cast<uintptr_t>();
              else
                ptr = stream_arg.cast<uintptr_t>();
            }
            return std::make_unique<kun_cuda::Executor>(
                reinterpret_cast<CUstream>(ptr));
          }),
          py::arg("stream") = py::none(),
          "Build an Executor.  `stream=None` → default CUDA stream; "
          "otherwise expects either an int (uintptr_t handle) or a "
          "cupy.cuda.Stream-like object exposing `.ptr`.")
      .def_property_readonly("stream",
          [](const kun_cuda::Executor &e) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(e.stream());
          },
          "Raw stream handle as an int (0 ↔ CUDA default stream).")
      .def("runGraph",
          [](kun_cuda::Executor &e, kun_cuda::Executable &exe,
              py::dict pyArgs) {
            auto c = collectArgs(exe, pyArgs);
            e.runGraph(exe, c.timeLength, c.numStocks, c.args);
          },
          py::arg("exe"), py::arg("args"),
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
         py::arg("module"),
         py::arg("graph_inputs"),
         py::arg("graph_outputs"),
         py::arg("gpu_arch")       = "sm_80",
         py::arg("target_triple")  = "nvptx64-nvidia-cuda",
         py::arg("target_features") = "",
         py::arg("opt_level")      = 3u,
         py::arg("toolkit_path")   = "",
         py::arg("external_kernels") = py::list(),
         py::arg("warps_per_cta")    = 0,
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
