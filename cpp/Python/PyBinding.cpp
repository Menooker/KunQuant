#include <Kun/Aligned.hpp>
#include <Kun/Context.hpp>
#include <Kun/Module.hpp>
#include <Kun/IO.hpp>
#include <Kun/MathUtil.hpp>
#include <Kun/RunGraph.hpp>
#include <Kun/StateBuffer.hpp>  // KUN_MALLOC_ALIGNMENT
#include <KunSIMD/cpu/Table.hpp>
#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/function.h>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

// Shape / count types are `size_t` everywhere — nanobind's ndarray
// API uses unsigned counts for shape, signed `int64_t` for strides.
// The one place we need signed is `num_stocks` (-1 sentinel for
// "infer from inputs"); we keep that as `int64_t` and cast explicitly.

static std::string shapeToString(const std::vector<size_t> &shape) {
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape.size(); i++) {
        ss << shape[i];
        if (i + 1 != shape.size()) {
            ss << ", ";
        }
    }
    ss << ")";
    return ss.str();
}

using CpuArray = nb::ndarray<nb::device::cpu>;
using CpuArrayRO = nb::ndarray<nb::device::cpu, nb::ro>;

static std::vector<size_t> arrayShape(const CpuArrayRO &arr) {
    std::vector<size_t> r;
    r.reserve(arr.ndim());
    for (size_t i = 0; i < arr.ndim(); i++) {
        r.push_back(arr.shape(i));
    }
    return r;
}

static bool dtypeMatches(const CpuArrayRO &arr, kun::Datatype dtype) {
    if (dtype == kun::Datatype::Float) {
        return arr.dtype() == nb::dtype<float>();
    }
    return arr.dtype() == nb::dtype<double>();
}

// Mirrors the pybind version: matching dtype, ndim, shape, positive
// dimensions, row-major contiguous strides.  Differs from pybind only
// in that nanobind reports strides in *elements*, not bytes — so the
// expected-stride walk uses element counts.
static void expectContiguousShape(kun::Datatype dtype,
                                    const CpuArrayRO &arr,
                                    const char *name,
                                    const std::vector<size_t> &shape) {
    if (!dtypeMatches(arr, dtype)) {
        if (dtype == kun::Datatype::Float) {
            throw std::runtime_error(std::string("Expecting float buffer at ") + name);
        } else {
            throw std::runtime_error(std::string("Expecting double buffer at ") + name);
        }
    }
    std::vector<size_t> actual = arrayShape(arr);
    if (actual.size() != shape.size() || actual != shape) {
        std::stringstream ss;
        ss << "Bad shape at " << name << " expected " << shapeToString(shape)
           << " but got " << shapeToString(actual);
        throw std::runtime_error(ss.str());
    }
    for (auto s : actual) {
        if (s == 0) {
            throw std::runtime_error(std::string("Bad dimension number at ") + name);
        }
    }
    // Row-major contiguous: stride at axis i (in elements) =
    //   product of shape[i+1 .. ndim-1]; innermost is 1.
    int64_t expected = 1;
    for (int i = (int)arr.ndim() - 1; i >= 0; i--) {
        if (arr.stride(i) != expected) {
            throw std::runtime_error(std::string("Bad stride at ") + name);
        }
        expected *= (int64_t)shape[i];
    }
}

struct ModuleHandle {
    const kun::Module *modu;
    std::shared_ptr<kun::Library> lib;
    ModuleHandle(const kun::Module *modu,
                 const std::shared_ptr<kun::Library> &lib)
        : modu{modu}, lib{lib} {}
};
struct StreamContextWrapper {
    std::shared_ptr<kun::Library> lib;
    kun::StreamContext ctx;
    StreamContextWrapper(std::shared_ptr<kun::Executor> exec,
                         const ModuleHandle *m, size_t num_stocks,
                         kun::InputStreamBase *states = nullptr)
        : lib{m->lib}, ctx{std::move(exec), m->modu, num_stocks, states} {}
};

const void *checkInput(const CpuArrayRO &arr, const std::string &name,
                       kun::MemoryLayout mlayout, kun::Datatype dtype,
                       size_t &known_S, size_t &known_T,
                       size_t &knownNumStocks, size_t simd_len) {
    if (mlayout == kun::MemoryLayout::STs) {
        if (arr.ndim() != 3) {
            throw std::runtime_error("Bad STs shape at " + name);
        }
        auto S = arr.shape(0);
        auto T = arr.shape(1);
        if (known_S == 0) {
            known_S = S;
            known_T = T;
            knownNumStocks = known_S * simd_len;
        }
        expectContiguousShape(dtype, arr, name.c_str(),
                              {known_S, known_T,
                               simd_len});
    } else if (mlayout == kun::MemoryLayout::TS) {
        if (arr.ndim() != 2) {
            throw std::runtime_error("Bad TS shape at " + name);
        }
        auto S = arr.shape(1);
        auto T = arr.shape(0);
        if (known_S == 0) {
            known_S = S / simd_len;
            knownNumStocks = S;
        }
        if (known_T == 0) {
            known_T = T;
        }
        expectContiguousShape(dtype, arr, name.c_str(),
                              {known_T, knownNumStocks});
    } else {
        throw std::runtime_error("Unknown layout at " + name);
    }
    return arr.data();
}

static float *runtimeInputPtr(const void *ptr) {
    // The Kun runtime still types input buffers as float*, while the
    // Python binding intentionally accepts read-only arrays for inputs.
    return static_cast<float *>(const_cast<void *>(ptr));
}

kun::AggregrationKind getAggregrationKind(const std::string &name) {
    if (name == "sum")   { return kun::AggregrationKind::AGGREGRATION_SUM; }
    if (name == "min")   { return kun::AggregrationKind::AGGREGRATION_MIN; }
    if (name == "max")   { return kun::AggregrationKind::AGGREGRATION_MAX; }
    if (name == "first") { return kun::AggregrationKind::AGGREGRATION_FIRST; }
    if (name == "last")  { return kun::AggregrationKind::AGGREGRATION_LAST; }
    if (name == "count") { return kun::AggregrationKind::AGGREGRATION_COUNT; }
    if (name == "mean")  { return kun::AggregrationKind::AGGREGRATION_MEAN; }
    throw std::runtime_error("Unknown aggregration kind: " + name);
}

static CpuArray castWritableCpuArray(nb::handle obj, const char *name) {
    CpuArray arr;
    if (!nb::try_cast(obj, arr, false)) {
        throw std::runtime_error(std::string("Expecting writable CPU buffer at ") + name);
    }
    return arr;
}

// Capsule-owned numpy array, KUN_MALLOC_ALIGNMENT-aligned via
// kunAlignedAlloc.  Python's GC frees the buffer when the array dies.
template <typename T>
static nb::ndarray<nb::numpy, T>
allocOwnedNumpyArray(const size_t *shape, size_t ndim) {
    size_t total = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total *= shape[i];
    }
    T *data = static_cast<T *>(
        kunAlignedAlloc(KUN_MALLOC_ALIGNMENT,
                        kun::roundUp(total * sizeof(T), KUN_MALLOC_ALIGNMENT)));
    if (!data) {
        throw std::bad_alloc();
    }
    // Capsule destructor runs when Python's last ref to the array drops.
    nb::capsule owner(data, [](void *p) noexcept {
        kunAlignedFree(p);
    });
    return nb::ndarray<nb::numpy, T>(data, ndim, shape, owner);
}

} // namespace

NB_MODULE(KunRunner, m) {
    m.attr("__name__") = "KunQuant.runner.KunRunner";
    m.doc() = "Code Runner for KunQuant generated code";

    nb::class_<kun::Executor>(m, "Executor");
    m.def("createSingleThreadExecutor", &kun::createSingleThreadExecutor);
    m.def("createMultiThreadExecutor", &kun::createMultiThreadExecutor);
    m.def("getRuntimePath", []() -> std::string {
#ifdef _WIN32
        char path[MAX_PATH];
        HMODULE hm = NULL;
        if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                  GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCSTR)&kun_simd::LogLookupTable<float>::logr_table,
                              &hm) == 0) {
            fprintf(stderr, "GetModuleHandle failed, error = %d\n", (int)GetLastError());
            return std::string();
        }
        if (GetModuleFileName(hm, path, sizeof(path)) == 0) {
            fprintf(stderr, "GetModuleFileName failed, error = %d\n", (int)GetLastError());
            return std::string();
        }
        return path;
#else
        Dl_info info;
        if (dladdr(&kun_simd::LogLookupTable<float>::logr_table, &info)) {
            return info.dli_fname;
        }
#endif
        return std::string();
    });

    nb::class_<ModuleHandle>(m, "Module")
        .def_prop_ro("output_layout",
                       [](ModuleHandle &mod) -> const char * {
                           switch (mod.modu->output_layout) {
                           case kun::MemoryLayout::STs:    return "STs";
                           case kun::MemoryLayout::TS:     return "TS";
                           case kun::MemoryLayout::STREAM: return "STREAM";
                           }
                           return "?";
                       })
        .def_prop_ro("blocking_len",
                       [](ModuleHandle &mod) { return mod.modu->blocking_len; })
        .def("getOutputNames",
             [](ModuleHandle &m) {
                 auto &mod = *(m.modu);
                 std::vector<std::string> ret;
                 for (size_t i = 0; i < mod.num_buffers; i++) {
                     auto &buf = mod.buffers[i];
                     if (buf.kind == kun::BufferKind::OUTPUT) {
                         ret.emplace_back(buf.name);
                     }
                 }
                 return ret;
             })
        .def("getOutputUnreliableCount", [](ModuleHandle &m) {
            auto &mod = *(m.modu);
            nb::dict ret;
            for (size_t i = 0; i < mod.num_buffers; i++) {
                auto &buf = mod.buffers[i];
                if (buf.kind == kun::BufferKind::OUTPUT) {
                    ret[nb::cast(buf.name)] = buf.unreliable_count;
                }
            }
            return ret;
        });

    nb::class_<kun::Library>(m, "Library")
        .def_static("load", [](const char *filename) {
            auto lib = kun::Library::load(filename);
            if (!lib) {
                throw std::runtime_error("Cannot load library");
            }
            return lib;
        })
        .def("setCleanup",
             [](kun::Library &v, std::function<void()> f) {
                 v.dtor = [f](kun::Library *) { f(); };
             })
        .def("getModule",
             [](const std::shared_ptr<kun::Library> &v,
                const char *name) -> std::unique_ptr<ModuleHandle> {
                 if (auto m = v->getModule(name)) {
                     return std::unique_ptr<ModuleHandle>(new ModuleHandle(m, v));
                 }
                 throw std::runtime_error("Module name not found");
             });

    m.def(
        "runGraph",
        [](std::shared_ptr<kun::Executor> exec, ModuleHandle *m,
           const nb::dict inputs, size_t cur_time, size_t length,
           const nb::object outputs, bool skip_check,
           int64_t num_stocks) {
            auto mod = m->modu;
            std::unordered_map<std::string, float *> bufs;
            size_t known_S = 0, known_T = 0, knownNumStocks = 0;
            size_t simd_len = mod->blocking_len;
            for (auto kv : inputs) {
                auto name = nb::cast<std::string>(kv.first);
                auto arr  = nb::cast<CpuArrayRO>(kv.second, false);
                bufs[name] = runtimeInputPtr(arr.data());
                if (skip_check) {
                    if (known_S == 0 && strncmp(name.c_str(), "__init", 6)) {
                        if (mod->input_layout == kun::MemoryLayout::STs) {
                            known_S = arr.shape(0);
                            known_T = arr.shape(1);
                        } else if (mod->input_layout == kun::MemoryLayout::TS) {
                            auto S = arr.shape(1);
                            known_T = arr.shape(0);
                            known_S = S / simd_len;
                        }
                    }
                    continue;
                }
                if (!dtypeMatches(arr, mod->dtype)) {
                    if (mod->dtype == kun::Datatype::Float) {
                        throw std::runtime_error("Expecting float buffer at " + name);
                    } else {
                        throw std::runtime_error("Expecting double buffer at " + name);
                    }
                }
                if (!strncmp(name.c_str(), "__init", 6)) {
                    if (arr.ndim() != 1) {
                        throw std::runtime_error("Bad Init shape at " + name);
                    }
                    auto S = arr.shape(0);
                    if (!knownNumStocks) {
                        knownNumStocks = S;
                    }
                    expectContiguousShape(mod->dtype, arr, name.c_str(),
                                          {knownNumStocks});
                } else if (mod->input_layout == kun::MemoryLayout::STs) {
                    if (arr.ndim() != 3) {
                        throw std::runtime_error("Bad STs shape at " + name);
                    }
                    auto S = arr.shape(0);
                    auto T = arr.shape(1);
                    if (known_S == 0) {
                        known_S = S;
                        known_T = T;
                        if (!knownNumStocks) {
                            knownNumStocks = known_S * simd_len;
                        }
                    }
                    expectContiguousShape(
                        mod->dtype, arr, name.c_str(),
                        {known_S, known_T,
                         mod->blocking_len});
                } else if (mod->input_layout == kun::MemoryLayout::TS) {
                    if (arr.ndim() != 2) {
                        throw std::runtime_error("Bad TS shape at " + name);
                    }
                    auto S = arr.shape(1);
                    auto T = arr.shape(0);
                    if (known_S == 0) {
                        known_S = S / simd_len;
                        known_T = T;
                        if (!knownNumStocks) {
                            knownNumStocks = S;
                        }
                        if (mod->aligned && knownNumStocks % simd_len != 0) {
                            throw std::runtime_error("Bad shape at " + name);
                        }
                    }
                    expectContiguousShape(mod->dtype, arr, name.c_str(),
                                          {known_T, knownNumStocks});
                } else {
                    throw std::runtime_error("Unknown layout at " + name);
                }
            }
            if (num_stocks < 0) {
                num_stocks = (int64_t)knownNumStocks;
            }
            // From here on `num_stocks` is guaranteed >= 0 so casts to
            // size_t for comparison with the unsigned counts are safe.
            const size_t num_stocks_u = (size_t)num_stocks;
            if (!skip_check) {
                if (length > known_T) {
                    throw std::runtime_error("Bad parameter: length");
                }
                if (mod->input_layout == kun::MemoryLayout::STs) {
                    if (num_stocks_u > knownNumStocks ||
                        knownNumStocks <= knownNumStocks - simd_len) {
                        throw std::runtime_error(
                            "num_stocks does not match the shape of inputs");
                    }
                } else {
                    if (num_stocks_u != knownNumStocks) {
                        throw std::runtime_error(
                            "num_stocks does not match the shape of inputs");
                    }
                }
            }
            nb::dict ret;
            // Build expected_out_shape as size_t directly — no
            // signed/unsigned vector copy at the binding-to-alloc
            // boundary.  Pass-by-pointer all the way down.
            std::vector<size_t> expected_out_shape;
            if (mod->output_layout == kun::MemoryLayout::STs) {
                expected_out_shape = {known_S, length, simd_len};
            } else {
                expected_out_shape = {length, num_stocks_u};
            }
            for (size_t i = 0; i < mod->num_buffers; i++) {
                auto &buf = mod->buffers[i];
                if (buf.kind != kun::BufferKind::OUTPUT) {
                    continue;
                }
                nb::object outbuffer;
                if (!outputs.is_none()) {
                    nb::dict outputs_dict = nb::cast<nb::dict>(outputs);
                    if (outputs_dict.contains(nb::cast(buf.name))) {
                        outbuffer = nb::borrow(outputs_dict[nb::cast(buf.name)]);
                        CpuArray view = castWritableCpuArray(outbuffer, buf.name);
                        if (!skip_check) {
                            expectContiguousShape(mod->dtype, CpuArrayRO(view), buf.name,
                                                  expected_out_shape);
                        }
                        bufs[buf.name] = static_cast<float *>(view.data());
                        ret[nb::cast(buf.name)] = outbuffer;
                        continue;
                    }
                }
                // Allocate via the templated capsule-owned helper —
                // 64-byte aligned, Python-owned via capsule deleter.
                if (mod->dtype == kun::Datatype::Double) {
                    auto arr = allocOwnedNumpyArray<double>(
                        expected_out_shape.data(), expected_out_shape.size());
                    bufs[buf.name] = (float *)arr.data();
                    ret[nb::cast(buf.name)] = nb::cast(std::move(arr));
                } else {
                    auto arr = allocOwnedNumpyArray<float>(
                        expected_out_shape.data(), expected_out_shape.size());
                    bufs[buf.name] = arr.data();
                    ret[nb::cast(buf.name)] = nb::cast(std::move(arr));
                }
            }
            kun::runGraph(exec, mod, bufs, num_stocks_u, known_T, cur_time, length);
            return ret;
        },
        nb::arg("exec"), nb::arg("mod"), nb::arg("inputs"), nb::arg("cur_time"),
        nb::arg("length"), nb::arg("outputs") = nb::dict(),
        nb::arg("skip_check") = false, nb::arg("num_stocks") = -1);

    m.def(
        "corrWith",
        [](std::shared_ptr<kun::Executor> exec,
           const std::vector<CpuArrayRO> &inputs,
           CpuArrayRO corr_with,
           const std::vector<CpuArray> &outs,
           const char *layout, bool rank_inputs) {
            kun::MemoryLayout mlayout;
            if (!strcmp(layout, "TS")) {
                mlayout = kun::MemoryLayout::TS;
            } else if (!strcmp(layout, "STs")) {
                mlayout = kun::MemoryLayout::STs;
            } else {
                throw std::runtime_error(std::string("Unknown layout") + layout);
            }
            if (inputs.size() != outs.size()) {
                throw std::runtime_error(
                    "number of inputs and outputs should match");
            }

            size_t known_S = 0, known_T = 0, knownNumStocks = 0;
            size_t simd_len = KUN_DEFAULT_FLOAT_SIMD_LEN;
            std::vector<float *> bufinputs;
            std::vector<float *> bufoutputs;

            float *bufcorr_with = runtimeInputPtr(checkInput(
                corr_with, "corr_with", mlayout, kun::Datatype::Float,
                known_S, known_T, knownNumStocks, simd_len));
            int idx = -1;
            for (auto &arr : inputs) {
                idx += 1;
                bufinputs.push_back(runtimeInputPtr(checkInput(
                    arr, std::string("buffer_") + std::to_string(idx), mlayout,
                    kun::Datatype::Float, known_S, known_T, knownNumStocks,
                    simd_len)));
            }
            std::vector<size_t> expected_out_shape{known_T};
            for (size_t i = 0; i < outs.size(); i++) {
                expectContiguousShape(kun::Datatype::Float, CpuArrayRO(outs[i]), "",
                                      expected_out_shape);
                bufoutputs.push_back(static_cast<float *>(outs[i].data()));
            }
            kun::corrWith(exec, mlayout, rank_inputs, bufinputs, bufcorr_with,
                          bufoutputs, knownNumStocks, known_T, 0, known_T);
        },
        nb::arg("exec"), nb::arg("inputs"), nb::arg("corr_with"),
        nb::arg("outs"), nb::arg("layout") = "TS",
        nb::arg("rank_inputs") = false);

    m.def(
        "aggregrate",
        [](std::shared_ptr<kun::Executor> exec,
           const std::vector<CpuArrayRO> &inputs,
           const std::vector<CpuArrayRO> &labels,
           const std::vector<nb::dict> &outs) {
            if (inputs.size() != labels.size() || inputs.size() != outs.size()) {
                throw std::runtime_error(
                    "number of inputs, labels and outputs should match");
            }
            if (inputs.empty()) {
                return;
            }
            kun::Datatype dtype = (inputs[0].dtype() == nb::dtype<float>())
                                      ? kun::Datatype::Float
                                      : kun::Datatype::Double;
            size_t known_S = 0, known_T_input = 0, knownNumStocks = 0;
            size_t simd_len = (dtype == kun::Datatype::Float)
                                       ? KUN_DEFAULT_FLOAT_SIMD_LEN
                                       : KUN_DEFAULT_DOUBLE_SIMD_LEN;
            std::vector<float *> bufinputs;
            std::vector<float *> buflabels;
            std::vector<kun::AggregrationOutput> bufoutputs;
            bufinputs.reserve(inputs.size());
            buflabels.reserve(labels.size());
            bufoutputs.reserve(inputs.size());

            for (size_t i = 0; i < inputs.size(); i++) {
                bufinputs.push_back(runtimeInputPtr(checkInput(
                    inputs[i], std::string("buffer_") + std::to_string(i),
                    kun::MemoryLayout::TS, dtype, known_S, known_T_input,
                    knownNumStocks, simd_len)));
                expectContiguousShape(dtype, labels[i], "label",
                                      {known_T_input});
                buflabels.push_back(runtimeInputPtr(labels[i].data()));
                size_t known_T_output = 0;
                kun::AggregrationOutput output{};
                for (auto kv : outs[i]) {
                    auto name = nb::cast<std::string>(kv.first);
                    auto idx = getAggregrationKind(name);
                    auto value_arr = castWritableCpuArray(kv.second, name.c_str());
                    checkInput(CpuArrayRO(value_arr),
                               std::string("output_") + name + std::to_string(idx),
                               kun::MemoryLayout::TS, dtype, known_S, known_T_output,
                               knownNumStocks, simd_len);
                    output.buffers[idx] = static_cast<float *>(value_arr.data());
                }
                bufoutputs.emplace_back(output);
            }

            kun::aggregrate(exec, inputs.size(), bufinputs.data(),
                            buflabels.data(), dtype, bufoutputs.data(),
                            knownNumStocks, known_T_input, 0, known_T_input);
        },
        nb::arg("exec"), nb::arg("inputs"), nb::arg("labels"), nb::arg("outs"));

    nb::class_<StreamContextWrapper>(m, "StreamContext")
        .def(nb::init<std::shared_ptr<kun::Executor>, const ModuleHandle *,
                       size_t>())
        .def("__init__",
             [](StreamContextWrapper *self,
                std::shared_ptr<kun::Executor> exec,
                const ModuleHandle *mod, size_t stocks, nb::object init) {
                 if (nb::isinstance<nb::str>(init)) {
                     auto filename = nb::cast<std::string>(init);
                     kun::FileInputStream stream(filename);
                     new (self) StreamContextWrapper(std::move(exec), mod,
                                                       stocks, &stream);
                     return;
                 }
                 if (nb::isinstance<nb::bytes>(init)) {
                     nb::bytes b = nb::cast<nb::bytes>(init);
                     kun::MemoryInputStream stream{b.c_str(), b.size()};
                     new (self) StreamContextWrapper(std::move(exec), mod,
                                                       stocks, &stream);
                     return;
                 }
                 throw std::runtime_error(
                     "Bad type for init, expecting filename or bytes");
             })
        .def("queryBufferHandle",
             [](StreamContextWrapper &t, const char *name) {
                 return t.ctx.queryBufferHandle(name);
             })
        .def("getCurrentBuffer",
             [](nb::handle self, size_t handle) -> nb::object {
                 // Zero-copy view of the internal buffer.  Taking
                 // `self` as nb::handle skips the nb::find() instance
                 // lookup — `self` is already the Python wrapper, use
                 // it directly as the ndarray owner.
                 auto &ths = nb::cast<StreamContextWrapper &>(self).ctx;
                 if (ths.m->dtype == kun::Datatype::Double) {
                     auto *buf = ths.getCurrentBufferPtrDouble(handle);
                     return nb::cast(nb::ndarray<nb::numpy, const double>(
                         buf, {ths.ctx.stock_count}, self));
                 }
                 auto *buf = ths.getCurrentBufferPtrFloat(handle);
                 return nb::cast(nb::ndarray<nb::numpy, const float>(
                     buf, {ths.ctx.stock_count}, self));
             })
        .def("pushData",
             [](StreamContextWrapper &t, size_t handle, CpuArrayRO data) {
                 auto &ths = t.ctx;
                 expectContiguousShape(ths.m->dtype, data, "input data",
                                       {ths.ctx.stock_count});
                 if (ths.m->dtype == kun::Datatype::Float) {
                     ths.pushData(handle, (const float *)data.data());
                 } else {
                     ths.pushData(handle, (const double *)data.data());
                 }
             })
        .def(
            "serializeStates",
            [](StreamContextWrapper &t, nb::object fileNameOrNone) -> nb::object {
                if (nb::isinstance<nb::str>(fileNameOrNone)) {
                    auto filename = nb::cast<std::string>(fileNameOrNone);
                    kun::FileOutputStream stream(filename);
                    if (!t.ctx.serializeStates(&stream)) {
                        throw std::runtime_error("Failed to serialize states");
                    }
                    return nb::none();
                }
                kun::MemoryOutputStream stream;
                if (!t.ctx.serializeStates(&stream)) {
                    throw std::runtime_error("Failed to serialize states");
                }
                return nb::bytes(stream.getData(), stream.getSize());
            },
            nb::arg("fileNameOrNone") = nb::none())
        .def("run", [](StreamContextWrapper &t) { t.ctx.run(); });
}
