#include <Kun/Context.hpp>
#include <Kun/Module.hpp>
#include <Kun/RunGraph.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

static void expectContiguousShape(const py::buffer_info &info, const char *name,
                                  const std::vector<py::ssize_t> &shape) {
    if (info.format != py::format_descriptor<float>::format()) {
        throw std::runtime_error(std::string("Expecting float buffer at ") +
                                 name);
    }
    // ST8t layout
    if (info.ndim != shape.size() || info.shape != shape) {
        throw std::runtime_error(std::string("Bad shape at ") + name);
    }
    for (auto s : info.shape) {
        if (s <= 0) {
            throw std::runtime_error(std::string("Bad dimension number at ") +
                                     name);
        }
    }
    py::ssize_t stride = sizeof(float);
    auto &strides = info.strides;
    for (int i = (int)info.ndim - 1; i >= 0; i--) {
        if (strides[i] != stride) {
            throw std::runtime_error(std::string("Bad stride at ") + name);
        }
        stride *= shape[i];
    }
}

PYBIND11_MODULE(KunRunner, m) {
    m.doc() = R"(Code Runner for KunQuant generated code)";

    py::class_<kun::Executor, std::shared_ptr<kun::Executor>>(m, "Executor");
    m.def("createSingleThreadExecutor", &kun::createSingleThreadExecutor);
    m.def("createMultiThreadExecutor", &kun::createMultiThreadExecutor);

    py::class_<kun::Module>(m, "Module")
        .def_property_readonly("output_layout",
                               [](kun::Module &mod) {
                                   switch (mod.output_layout) {
                                   case kun::OutputLayout::ST8s:
                                       return "ST8s";
                                   case kun::OutputLayout::TS:
                                       return "TS";
                                   case kun::OutputLayout::STREAM:
                                       return "STREAM";
                                   }
                                   return "?";
                               })
        .def("getOutputNames",
             [](kun::Module &mod) {
                 std::vector<std::string> ret;
                 for (size_t i = 0; i < mod.num_buffers; i++) {
                     auto &buf = mod.buffers[i];
                     if (buf.kind == kun::BufferKind::OUTPUT) {
                         ret.emplace_back(buf.name);
                     }
                 }
                 return ret;
             })
        .def("getOutputUnreliableCount", [](kun::Module &mod) {
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
        .def_static("load", &kun::Library::load)
        .def("getModule", &kun::Library::getModule,
             py::return_value_policy::reference);
    m.def(
        "runGraph",
        [](std::shared_ptr<kun::Executor> exec, const kun::Module *mod,
           const py::dict inputs, size_t cur_time, size_t length,
           const py::object outputs) {
            std::unordered_map<std::string, float *> bufs;
            py::ssize_t known_S = 0;
            py::ssize_t known_T = 0;
            for (auto kv : inputs) {
                auto name = py::cast<std::string>(kv.first);
                auto buf_obj = py::cast<py::buffer>(kv.second);
                auto info = buf_obj.request();
                if (info.format != py::format_descriptor<float>::format()) {
                    throw std::runtime_error("Expecting float buffer at " +
                                             name);
                }
                if (mod->input_layout == kun::OutputLayout::ST8s) {
                    // ST8t layout
                    if (info.ndim != 3) {
                        throw std::runtime_error("Bad shape at " + name);
                    }
                    auto S = info.shape[0];
                    auto T = info.shape[1];
                    if (known_S == 0) {
                        known_S = S;
                        known_T = T;
                    }
                    expectContiguousShape(
                        info, name.c_str(),
                        {known_S, known_T, (py::ssize_t)kun::simd_len});
                } else if (mod->input_layout == kun::OutputLayout::TS) {
                    // TS layout
                    if (info.ndim != 2) {
                        throw std::runtime_error("Bad shape at " + name);
                    }
                    auto S = info.shape[0];
                    auto T = info.shape[1];
                    if (known_S == 0) {
                        known_S = S / (py::ssize_t)kun::simd_len;
                        known_T = T;
                    }
                    expectContiguousShape(
                        info, name.c_str(),
                        {S, T});
                } else {
                    throw std::runtime_error("Unknown layout at " + name);
                }
                bufs[name] = (float *)info.ptr;
            }
            if ((py::ssize_t)length > known_T) {
                throw std::runtime_error("Bad parameter: length");
            }
            py::dict ret{};
            py::array::ShapeContainer expected_out_shape;
            if (mod->output_layout == kun::OutputLayout::ST8s) {
                expected_out_shape = {known_S, (py::ssize_t)length,
                                      (py::ssize_t)kun::simd_len};
            } else {
                expected_out_shape = {(py::ssize_t)length,
                                      known_S * (py::ssize_t)kun::simd_len};
            }
            for (size_t i = 0; i < mod->num_buffers; i++) {
                auto &buf = mod->buffers[i];
                if (buf.kind == kun::BufferKind::OUTPUT) {
                    py::array_t<float, py::array::c_style> outbuffer;
                    if (!outputs.is_none() && outputs.contains(buf.name)) {
                        outbuffer =
                            outputs[buf.name]
                                .cast<py::array_t<float, py::array::c_style>>();
                        auto info = outbuffer.request();
                        expectContiguousShape(info, buf.name,
                                              *expected_out_shape);
                        bufs[buf.name] = (float *)info.ptr;
                    } else {
                        outbuffer = py::array_t<float, py::array::c_style>{
                            expected_out_shape};
                        bufs[buf.name] = (float *)outbuffer.request().ptr;
                    }
                    ret[buf.name] = outbuffer;
                }
            }
            kun::runGraph(exec, mod, bufs, known_S * kun::simd_len, known_T,
                          cur_time, length);
            return ret;
        },
        py::arg("exec"), py::arg("mod"), py::arg("inputs"), py::arg("cur_time"),
        py::arg("length"), py::arg("outputs") = py::dict());

    py::class_<kun::StreamContext>(m, "StreamContext")
        .def(py::init<std::shared_ptr<kun::Executor>, const kun::Module *,
                      size_t>())
        .def("queryBufferHandle", &kun::StreamContext::queryBufferHandle)
        .def("getCurrentBuffer",
             [](kun::StreamContext &ths, size_t handle) {
                 auto buf = ths.getCurrentBufferPtr(handle);
                 return py::array_t<float, py::array::c_style>{
                     (py::ssize_t)ths.ctx.stock_count, buf};
             })
        .def("pushData",
             [](kun::StreamContext &ths, size_t handle,
                py::array_t<float, py::array::c_style> data) {
                 auto info = data.request();
                 if (info.shape.size() != 1 &&
                     info.shape[0] != ths.ctx.stock_count) {
                     throw std::runtime_error("Bad input data to push");
                 }
                 ths.pushData(handle, (float *)info.ptr);
             })
        .def("run", &kun::StreamContext::run);
}