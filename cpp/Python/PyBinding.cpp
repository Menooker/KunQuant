#include <Kun/Context.hpp>
#include <Kun/Module.hpp>
#include <Kun/IO.hpp>
#include <Kun/RunGraph.hpp>
#include <KunSIMD/cpu/Table.hpp>
#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

static void expectContiguousShape(kun::Datatype dtype,
                                  const py::buffer_info &info, const char *name,
                                  const std::vector<py::ssize_t> &shape) {
    if (dtype == kun::Datatype::Float) {
        if (info.format != py::format_descriptor<float>::format())
            throw std::runtime_error(std::string("Expecting float buffer at ") +
                                     name);
    } else if (info.format != py::format_descriptor<double>::format()) {
        throw std::runtime_error(std::string("Expecting double buffer at ") +
                                 name);
    }
    // ST8s layout
    if (info.ndim != shape.size() || info.shape != shape) {
        throw std::runtime_error(std::string("Bad shape at ") + name);
    }
    for (auto s : info.shape) {
        if (s <= 0) {
            throw std::runtime_error(std::string("Bad dimension number at ") +
                                     name);
        }
    }
    py::ssize_t stride =
        dtype == kun::Datatype::Double ? sizeof(double) : sizeof(float);
    auto &strides = info.strides;
    for (int i = (int)info.ndim - 1; i >= 0; i--) {
        if (strides[i] != stride) {
            throw std::runtime_error(std::string("Bad stride at ") + name);
        }
        stride *= shape[i];
    }
}

namespace {
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
                         const ModuleHandle *m, size_t num_stocks, kun::InputStreamBase* states = nullptr)
        : lib{m->lib}, ctx{std::move(exec), m->modu, num_stocks, states}
           {}
};
} // namespace

PYBIND11_MODULE(KunRunner, m) {
    m.attr("__name__") = "KunQuant.runner.KunRunner";
    m.doc() = R"(Code Runner for KunQuant generated code)";

    py::class_<kun::Executor, std::shared_ptr<kun::Executor>>(m, "Executor");
    m.def("createSingleThreadExecutor", &kun::createSingleThreadExecutor);
    m.def("createMultiThreadExecutor", &kun::createMultiThreadExecutor);
    m.def("getRuntimePath", []() -> std::string {
#ifdef _WIN32
        char path[MAX_PATH];
        HMODULE hm = NULL;

        if (GetModuleHandleEx(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                (LPCSTR)&kun_simd::LogLookupTable<float>::logr_table,
                &hm) == 0) {
            int ret = GetLastError();
            fprintf(stderr, "GetModuleHandle failed, error = %d\n", ret);
            return std::string();
            // Return or however you want to handle an error.
        }
        if (GetModuleFileName(hm, path, sizeof(path)) == 0) {
            int ret = GetLastError();
            fprintf(stderr, "GetModuleFileName failed, error = %d\n", ret);
            return std::string();
            // Return or however you want to handle an error.
        }
        return path;
#else
    // On Windows, use GetMappedFileNameW
    Dl_info info;
    if (dladdr(&kun_simd::LogLookupTable<float>::logr_table, &info)) { return info.dli_fname; }
#endif
        return std::string();
    });
    py::class_<ModuleHandle>(m, "Module")
        .def_property_readonly("output_layout",
                               [](ModuleHandle &mod) {
                                   switch (mod.modu->output_layout) {
                                   case kun::MemoryLayout::STs:
                                       return "STs";
                                   case kun::MemoryLayout::TS:
                                       return "TS";
                                   case kun::MemoryLayout::STREAM:
                                       return "STREAM";
                                   }
                                   return "?";
                               })
        .def_property_readonly(
            "blocking_len",
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
            py::dict ret;
            for (size_t i = 0; i < mod.num_buffers; i++) {
                auto &buf = mod.buffers[i];
                if (buf.kind == kun::BufferKind::OUTPUT) {
                    ret[buf.name] = buf.unreliable_count;
                }
            }
            return ret;
        });
    py::class_<kun::Library, std::shared_ptr<kun::Library>>(m, "Library")
        .def_static("load", [](const char *filename) {
            auto lib = kun::Library::load(filename);
            if (!lib) {
                throw std::runtime_error("Cannot load library");
            }
            return lib;
        })
        .def("setCleanup",
             [](kun::Library &v, py::function f) {
                 v.dtor = [f](kun::Library *v) { f(); };
             })
        .def("getModule",
             [](const std::shared_ptr<kun::Library> &v,
                const char *name) -> std::unique_ptr<ModuleHandle> {
                 if (auto m = v->getModule(name)) {
                     return std::unique_ptr<ModuleHandle>(
                         new ModuleHandle(m, v));
                 }
                 throw std::runtime_error("Module name not found");
             });
    m.def(
        "runGraph",
        [](std::shared_ptr<kun::Executor> exec, ModuleHandle *m,
           const py::dict inputs, size_t cur_time, size_t length,
           const py::object outputs, bool skip_check, py::ssize_t num_stocks) {
            auto mod = m->modu;
            std::unordered_map<std::string, float *> bufs;
            py::ssize_t known_S = 0;
            py::ssize_t known_T = 0;
            py::ssize_t knownNumStocks = 0;
            py::ssize_t simd_len = mod->blocking_len;
            for (auto kv : inputs) {
                auto name = py::cast<std::string>(kv.first);
                auto buf_obj = py::cast<py::buffer>(kv.second);
                auto info = buf_obj.request();
                bufs[name] = (float *)info.ptr;
                if (skip_check) {
                    // if S is not known and it is not init buffer
                    if (known_S == 0 && strncmp(name.c_str(), "__init", 6)) {
                        if (mod->input_layout == kun::MemoryLayout::STs) {
                            auto S = info.shape[0];
                            auto T = info.shape[1];
                            known_S = S;
                            known_T = T;
                        } else if (mod->input_layout == kun::MemoryLayout::TS) {
                            auto S = info.shape[1];
                            auto T = info.shape[0];
                            known_S = S / simd_len;
                            known_T = T;
                        }
                    }
                    continue;
                }
                if (mod->dtype == kun::Datatype::Float) {
                    if (info.format != py::format_descriptor<float>::format())
                        throw std::runtime_error("Expecting float buffer at " +
                                                 name);
                } else if (info.format !=
                           py::format_descriptor<double>::format()) {
                    throw std::runtime_error("Expecting double buffer at " +
                                             name);
                }
                if (!strncmp(name.c_str(), "__init", 6)) {
                    if (info.ndim != 1) {
                        throw std::runtime_error("Bad Init shape at " + name);
                    }
                    auto S = info.shape[0];
                    if(!knownNumStocks)
                        knownNumStocks = S;
                    expectContiguousShape(mod->dtype, info, name.c_str(),
                                          {knownNumStocks});
                    
                } else if (mod->input_layout == kun::MemoryLayout::STs) {
                    // ST8t layout
                    if (info.ndim != 3) {
                        throw std::runtime_error("Bad STs shape at " + name);
                    }
                    auto S = info.shape[0];
                    auto T = info.shape[1];
                    if (known_S == 0) {
                        known_S = S;
                        known_T = T;
                        if (!knownNumStocks)
                            knownNumStocks = known_S * simd_len;
                    }
                    expectContiguousShape(
                        mod->dtype, info, name.c_str(),
                        {known_S, known_T, (py::ssize_t)mod->blocking_len});
                } else if (mod->input_layout == kun::MemoryLayout::TS) {
                    // TS layout
                    if (info.ndim != 2) {
                        throw std::runtime_error("Bad TS shape at " + name);
                    }
                    auto S = info.shape[1];
                    auto T = info.shape[0];
                    if (known_S == 0) {
                        known_S = S / simd_len;
                        known_T = T;
                        if (!knownNumStocks)
                            knownNumStocks = S;
                        if (mod->aligned) {
                            if (knownNumStocks % simd_len != 0) {
                                throw std::runtime_error("Bad shape at " +
                                                         name);
                            }
                        }
                    }
                    expectContiguousShape(mod->dtype, info, name.c_str(),
                                          {known_T, knownNumStocks});
                } else {
                    throw std::runtime_error("Unknown layout at " + name);
                }
            }
            if (num_stocks < 0) {
                num_stocks = knownNumStocks;
            }
            if (!skip_check) {
                if ((py::ssize_t)length > known_T) {
                    throw std::runtime_error("Bad parameter: length");
                }
                if (mod->input_layout == kun::MemoryLayout::STs) {
                    if (num_stocks > knownNumStocks ||
                        knownNumStocks <= knownNumStocks - simd_len) {
                        throw std::runtime_error(
                            "num_stocks does not match the shape of inputs");
                    }
                } else {
                    if (num_stocks != knownNumStocks) {
                        throw std::runtime_error(
                            "num_stocks does not match the shape of inputs");
                    }
                }
            }
            py::dict ret{};
            py::array::ShapeContainer expected_out_shape;
            if (mod->output_layout == kun::MemoryLayout::STs) {
                expected_out_shape = {known_S, (py::ssize_t)length, simd_len};
            } else {
                expected_out_shape = {(py::ssize_t)length, num_stocks};
            }
            for (size_t i = 0; i < mod->num_buffers; i++) {
                auto &buf = mod->buffers[i];
                if (buf.kind == kun::BufferKind::OUTPUT) {
                    py::array outbuffer;
                    if (!outputs.is_none() && outputs.contains(buf.name)) {
                        py::array v;
                        outbuffer = outputs[buf.name].cast<py::buffer>();
                        auto info = outbuffer.request(true);
                        if (!skip_check) {
                            expectContiguousShape(mod->dtype, info, buf.name,
                                                  *expected_out_shape);
                        }
                        bufs[buf.name] = (float *)info.ptr;
                    } else {
                        if (mod->dtype == kun::Datatype::Float) {
                            outbuffer = py::array_t<float, py::array::c_style>{
                                expected_out_shape};
                        } else {
                            outbuffer = py::array_t<double, py::array::c_style>{
                                expected_out_shape};
                        }
                        bufs[buf.name] = (float *)outbuffer.request().ptr;
                    }
                    ret[buf.name] = outbuffer;
                }
            }
            kun::runGraph(exec, mod, bufs, num_stocks, known_T, cur_time,
                          length);
            return ret;
        },
        py::arg("exec"), py::arg("mod"), py::arg("inputs"), py::arg("cur_time"),
        py::arg("length"), py::arg("outputs") = py::dict(),
        py::arg("skip_check") = false, py::arg("num_stocks") = -1);

    m.def(
        "corrWith",
        [](std::shared_ptr<kun::Executor> exec,
           const std::vector<py::buffer> &inputs, py::buffer corr_with,
           const std::vector<py::buffer> &outs, const char *layout,
           bool rank_inputs) {
            kun::MemoryLayout mlayout;
            if (!strcmp(layout, "TS")) {
                mlayout = kun::MemoryLayout::TS;
            } else if (!strcmp(layout, "STs")) {
                mlayout = kun::MemoryLayout::STs;
            } else {
                throw std::runtime_error(std::string("Unknown layout") +
                                         layout);
            }
            if (inputs.size() != outs.size())
                throw std::runtime_error(
                    "number of inputs and outputs should match");

            py::ssize_t known_S = 0;
            py::ssize_t known_T = 0;
            py::ssize_t knownNumStocks = 0;
            py::ssize_t simd_len = 8;
            std::vector<float *> bufinputs;
            std::vector<float *> bufoutputs;
            auto check_input = [&](py::buffer buf_obj,
                                   const std::string &name) -> float * {
                auto info = buf_obj.request();
                if (mlayout == kun::MemoryLayout::STs) {
                    // ST8t layout
                    if (info.ndim != 3) {
                        throw std::runtime_error("Bad STs shape at " + name);
                    }
                    auto S = info.shape[0];
                    auto T = info.shape[1];
                    if (known_S == 0) {
                        known_S = S;
                        known_T = T;
                        knownNumStocks = known_S * simd_len;
                    }
                    expectContiguousShape(kun::Datatype::Float, info,
                                          name.c_str(),
                                          {known_S, known_T, simd_len});
                } else if (mlayout == kun::MemoryLayout::TS) {
                    // TS layout
                    if (info.ndim != 2) {
                        throw std::runtime_error("Bad TS shape at " + name);
                    }
                    auto S = info.shape[1];
                    auto T = info.shape[0];
                    if (known_S == 0) {
                        known_S = S / simd_len;
                        known_T = T;
                        knownNumStocks = S;
                    }
                    expectContiguousShape(kun::Datatype::Float, info,
                                          name.c_str(),
                                          {known_T, knownNumStocks});
                } else {
                    throw std::runtime_error("Unknown layout at " + name);
                }
                return (float *)info.ptr;
            };
            float *bufcorr_with = check_input(corr_with, "corr_with");
            int idx = -1;
            for (auto buf_obj : inputs) {
                idx += 1;
                bufinputs.push_back(check_input(
                    buf_obj, std::string("buffer_") + std::to_string(idx)));
            }
            py::array::ShapeContainer expected_out_shape{known_T};
            for (size_t i = 0; i < outs.size(); i++) {
                auto &buf = outs[i];
                auto info = buf.request(true);
                expectContiguousShape(kun::Datatype::Float, info, "",
                                      *expected_out_shape);
                bufoutputs.push_back((float *)info.ptr);
            }
            kun::corrWith(exec, mlayout, rank_inputs, bufinputs, bufcorr_with,
                          bufoutputs, knownNumStocks, known_T, 0, known_T);
        },
        py::arg("exec"), py::arg("inputs"), py::arg("corr_with"),
        py::arg("outs"), py::arg("layout") = "TS",
        py::arg("rank_inputs") = false);

    py::class_<StreamContextWrapper>(m, "StreamContext")
        .def(py::init<std::shared_ptr<kun::Executor>, const ModuleHandle *,
                      size_t>())
        .def(py::init([](std::shared_ptr<kun::Executor> exec,
                         const ModuleHandle *mod, size_t stocks,
                         py::object init) {
            if (py::isinstance<std::string>(init)) {
                auto filename = py::cast<std::string>(init);
                kun::FileInputStream stream(filename);
                return new StreamContextWrapper(std::move(exec), mod, stocks,
                                                &stream);
            } else if (py::isinstance<py::bytes>(init)) {
                py::bytes b = py::cast<py::bytes>(init);
                char *data;
                py::ssize_t size;
                if (PYBIND11_BYTES_AS_STRING_AND_SIZE(b.ptr(), &data, &size))
                    throw std::runtime_error("Failed to get bytes data");
                kun::MemoryInputStream stream{data, (size_t)size};
                return new StreamContextWrapper(std::move(exec), mod, stocks,
                                                &stream);
            }
            throw std::runtime_error(
                "Bad type for init, expecting filename or bytes");
        }))
        .def("queryBufferHandle",
             [](StreamContextWrapper &t, const char *name) {
                 return t.ctx.queryBufferHandle(name);
             })
        .def("getCurrentBuffer",
             [](StreamContextWrapper &t, size_t handle) -> py::buffer {
                 auto &ths = t.ctx;
                 if (ths.m->dtype == kun::Datatype::Double) {
                     auto buf = ths.getCurrentBufferPtrDouble(handle);
                     return py::array_t<double, py::array::c_style>{
                         (py::ssize_t)ths.ctx.stock_count, buf};
                 }
                 auto buf = ths.getCurrentBufferPtrFloat(handle);
                 return py::array_t<float, py::array::c_style>{
                     (py::ssize_t)ths.ctx.stock_count, buf};
             })
        .def(
            "pushData",
            [](StreamContextWrapper &t, size_t handle, py::array data) {
                auto &ths = t.ctx;
                py::ssize_t ndim;
                if (ths.m->dtype == kun::Datatype::Float) {
                    if (!py::isinstance<py::array_t<float, py::array::c_style>>(
                            data)) {
                        throw std::runtime_error(
                            "Bad input type to push, expecting float");
                    }
                } else {
                    if (!py::isinstance<
                            py::array_t<double, py::array::c_style>>(data)) {
                        throw std::runtime_error(
                            "Bad input type to push, expecting float");
                    }
                }
                if (data.ndim() != 1 ||
                    data.shape()[0] != ths.ctx.stock_count) {
                    throw std::runtime_error(
                        "Bad dimension for input data to push");
                }
                if (ths.m->dtype == kun::Datatype::Float) {
                    ths.pushData(handle, (const float *)data.data());
                } else {
                    ths.pushData(handle, (const double *)data.data());
                }
            })
        .def(
            "serializeStates",
            [](StreamContextWrapper &t, py::object fileNameOrNone) -> py::object {
                if (py::isinstance<py::str>(fileNameOrNone)) {
                    auto filename = py::cast<std::string>(fileNameOrNone);
                    kun::FileOutputStream stream(filename);
                    if (!t.ctx.serializeStates(&stream)) {
                        throw std::runtime_error("Failed to serialize states");
                    }
                    return py::none();
                }
                kun::MemoryOutputStream stream;
                if (!t.ctx.serializeStates(&stream)) {
                    throw std::runtime_error("Failed to serialize states");
                }
                py::bytes b(stream.getData(), (py::ssize_t)stream.getSize());
                return b;
            },
            py::arg("fileNameOrNone") = py::none())
        .def("run", [](StreamContextWrapper &t) { t.ctx.run(); });
}