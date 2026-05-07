//===- MlirBinding.cpp - Python bindings for the kunir → PTX flow ------===//
//
// Exposes:
//   kun_mlir.parse(text)            → ModuleOp     (loads MLIR text)
//   ModuleOp.to_string() / __str__  → str          (dumps the module)
//   kun_mlir.lower_to_ptx(mod, …)   → str          (kunir → PTX)
//   kun_mlir.ptx_to_cubin(ptx, …)   → bytes        (PTX → CUBIN via ptxas)
//   kun_mlir.compile(mod, …)        → Executable   (kunir → loadable kernel)
//   Executable.launch({name: cupy}) → None         (cuLaunchKernel + sync)
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"

// Dialect registrations
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "KunCuda/Runtime.h"
#include "KunGpu/KunGpuDialect.h"
#include "KunGpu/PtxBackend.h"
#include "KunIr/KunIrDialect.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

//===----------------------------------------------------------------------===//
// MLIR module wrapper
//===----------------------------------------------------------------------===//

class PyModule {
public:
  PyModule()
      : ctx(std::make_unique<mlir::MLIRContext>(makeRegistry(),
                                                  mlir::MLIRContext::Threading::DISABLED)) {
    ctx->loadAllAvailableDialects();
  }

  static mlir::DialectRegistry makeRegistry() {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::index::IndexDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::NVVM::NVVMDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<kunir::KunIrDialect>();
    registry.insert<kungpu::KunGpuDialect>();
    return registry;
  }

  static std::unique_ptr<PyModule> parse(const std::string &text) {
    auto pm = std::make_unique<PyModule>();
    pm->module = mlir::parseSourceString<mlir::ModuleOp>(text, pm->ctx.get());
    if (!pm->module)
      throw std::runtime_error("kun_mlir.parse: failed to parse MLIR text");
    return pm;
  }

  std::string toString() const {
    std::string out;
    llvm::raw_string_ostream os(out);
    module.get().print(os);
    os.flush();
    return out;
  }

  std::unique_ptr<mlir::MLIRContext> ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

//===----------------------------------------------------------------------===//
// One-shot helpers
//===----------------------------------------------------------------------===//

static std::string pyLowerToPtx(PyModule &pm, const std::string &targetCpu,
                                  const std::string &targetTriple,
                                  const std::string &targetFeatures,
                                  unsigned optLevel,
                                  unsigned sizeLevel) {
  kungpu::PtxCompileOptions opts;
  if (!targetCpu.empty())      opts.targetCpu      = targetCpu;
  if (!targetTriple.empty())   opts.targetTriple   = targetTriple;
  if (!targetFeatures.empty()) opts.targetFeatures = targetFeatures;
  opts.optLevel  = optLevel;
  opts.sizeLevel = sizeLevel;

  std::string ptx;
  if (failed(kungpu::compileKunIrToPtx(pm.module.get(), opts, ptx)))
    throw std::runtime_error("kun_mlir.lower_to_ptx failed");
  return ptx;
}

static py::bytes pyPtxToCubin(const std::string &ptx,
                                const std::string &gpuArch,
                                const std::vector<std::string> &extraArgs,
                                const std::string &ptxasPath) {
  kungpu::PtxToCubinOptions opts;
  if (!gpuArch.empty())   opts.gpuArch   = gpuArch;
  if (!ptxasPath.empty()) opts.ptxasPath = ptxasPath;
  opts.extraArgs = extraArgs;

  std::vector<char> cubin;
  std::string errMsg;
  if (failed(kungpu::compilePtxToCubin(ptx, opts, cubin, errMsg)))
    throw std::runtime_error(errMsg.empty() ? "kun_mlir.ptx_to_cubin failed"
                                              : errMsg);
  return py::bytes(cubin.data(), cubin.size());
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
  ordered.reserve(exe.inputNames().size() + exe.outputNames().size());
  for (auto &n : exe.inputNames())  ordered.push_back(n);
  for (auto &n : exe.outputNames()) ordered.push_back(n);
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

static std::unique_ptr<kun_cuda::Executable>
pyCompile(PyModule &pm, const std::string &targetCpu,
            const std::string &targetTriple,
            const std::string &targetFeatures, unsigned optLevel,
            unsigned sizeLevel, const std::string &ptxasPath) {
  kungpu::PtxCompileOptions popts;
  if (!targetCpu.empty())      popts.targetCpu      = targetCpu;
  if (!targetTriple.empty())   popts.targetTriple   = targetTriple;
  if (!targetFeatures.empty()) popts.targetFeatures = targetFeatures;
  popts.optLevel  = optLevel;
  popts.sizeLevel = sizeLevel;

  kungpu::PtxToCubinOptions copts;
  copts.gpuArch   = targetCpu.empty() ? "sm_80" : targetCpu;
  copts.ptxasPath = ptxasPath;

  kun_cuda::ExecutableData data;
  if (failed(kungpu::compileKunIrToExecutable(pm.module.get(), popts, copts,
                                                data)))
    throw std::runtime_error("kun_mlir.compile failed");
  return std::make_unique<kun_cuda::Executable>(std::move(data));
}

} // namespace

PYBIND11_MODULE(kun_mlir, m) {
  m.doc() = "Bindings for the KunQuant MLIR compiler (kunir → PTX → CUBIN "
             "→ launch).";

  py::class_<PyModule>(m, "ModuleOp")
      .def("to_string", &PyModule::toString,
            "Return the textual MLIR form of the module.")
      .def("__str__",  &PyModule::toString)
      .def("__repr__", [](const PyModule &m) {
        return "<kun_mlir.ModuleOp>\n" + m.toString();
      });

  m.def("parse", &PyModule::parse, py::arg("text"),
         "Parse an MLIR text fragment into a ModuleOp.");

  m.def("lower_to_ptx", &pyLowerToPtx,
         py::arg("module"),
         py::arg("target_cpu")     = "sm_80",
         py::arg("target_triple")  = "nvptx64-nvidia-cuda",
         py::arg("target_features") = "",
         py::arg("opt_level")      = 3u,
         py::arg("size_level")     = 0u,
         "Lower kunir → PTX text.  Returns a Python str.");

  m.def("ptx_to_cubin", &pyPtxToCubin,
         py::arg("ptx"),
         py::arg("gpu_arch")   = "sm_80",
         py::arg("extra_args") = std::vector<std::string>{},
         py::arg("ptxas_path") = "",
         "Assemble PTX → CUBIN via ptxas.  Returns bytes.");

  py::class_<kun_cuda::Executable>(m, "Executable")
      .def_property_readonly("kernel_name",   &kun_cuda::Executable::kernelName)
      .def_property_readonly("input_names",   &kun_cuda::Executable::inputNames)
      .def_property_readonly("output_names",  &kun_cuda::Executable::outputNames)
      .def_property_readonly("warps_per_cta", &kun_cuda::Executable::warpsPerCta)
      .def_property_readonly("vector_size",   &kun_cuda::Executable::vectorSize)
      .def_property_readonly("cubin",
            [](const kun_cuda::Executable &e) {
              const auto &b = e.data().cubin;
              return py::bytes(b.data(), b.size());
            })
      .def("launch",
            [](kun_cuda::Executable &e, py::dict pyArgs) {
              auto c = collectArgs(e, pyArgs);
              e.launch(c.timeLength, c.numStocks, c.args);
            },
            py::arg("args"),
            "Launch the kernel.  `args` is a {name → cupy_array} dict; "
            "names must match input_names ++ output_names.  All arrays "
            "must be float32, 2-D, shape (time_length, num_stocks) — TS "
            "layout — and reside on the GPU.");

  m.def("compile", &pyCompile,
         py::arg("module"),
         py::arg("target_cpu")     = "sm_80",
         py::arg("target_triple")  = "nvptx64-nvidia-cuda",
         py::arg("target_features") = "",
         py::arg("opt_level")      = 3u,
         py::arg("size_level")     = 0u,
         py::arg("ptxas_path")     = "",
         "Compile a kunir module all the way to a loaded Executable "
         "(kunir → LLVM dialect → LLVM IR → PTX → CUBIN → cuModuleLoad).");
}
