#include "Context.hpp"
#include "Module.hpp"
#include "RunGraph.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(KunRunner, m) {
    m.doc() = R"(Code Runner for KunQuant generated code)";

    py::class_<kun::Executor, std::shared_ptr<kun::Executor>>(m, "Executor");
    m.def("createSingleThreadExecutor", &kun::createSingleThreadExecutor);

    py::class_<kun::Module>(m, "Module");
    py::class_<kun::Library, std::shared_ptr<kun::Library>>(m, "Library")
        .def_static("load", &kun::Library::load)
        .def("getModule", &kun::Library::getModule,
             py::return_value_policy::reference);
    m.def("runGraph", [](std::shared_ptr<kun::Executor> exec,
                         const kun::Module *mod, const py::dict inputs,
                         size_t cur_time, size_t length, const py::dict outputs) {
        std::unordered_map<std::string, float *> bufs;
        py::ssize_t known_S = 0;
        py::ssize_t known_T = 0;
        for (auto kv : inputs) {
            auto name = py::cast<std::string>(kv.first);
            auto buf_obj = py::cast<py::buffer>(kv.second);
            auto info = buf_obj.request();
            if (info.format != py::format_descriptor<float>::format()) {
                throw std::runtime_error("Expecting float buffer at " + name);
            }
            // ST8t layout
            if (info.ndim != 3 || info.shape.back() != kun::simd_len) {
                throw std::runtime_error("Bad shape at " + name);
            }
            auto S = info.shape[0];
            auto T = info.shape[1];
            if (known_S == 0) {
                known_S = S;
                known_T = T;
            } else {
                if (S != known_S || T != known_T) {
                    throw std::runtime_error("Expecting same shape for " +
                                             name);
                }
            }
            if (known_S <= 0 || known_T <= 0) {
                throw std::runtime_error("Bad input shape" + name);
            }
            py::ssize_t expectedS0 = T * kun::simd_len * sizeof(float);
            py::ssize_t expectedS1 = kun::simd_len * sizeof(float);
            py::ssize_t expectedS2 = sizeof(float);
            if (info.strides !=
                std::vector<py::ssize_t>{expectedS0, expectedS1, expectedS2}) {
                throw std::runtime_error(
                    "Bad stride. Dense buffer is expected: " + name);
            }
            bufs[name] = (float *)info.ptr;
        }
        py::dict ret{};
        py::array::ShapeContainer expected_out_shape;
        if (mod->layout == kun::OutputLayout::ST8s) {
            expected_out_shape = {known_S, known_T, (py::ssize_t) kun::simd_len};
        } else {
            expected_out_shape = {known_T, known_S * (py::ssize_t) kun::simd_len};
        }
        for (size_t i = 0; i < mod->num_buffers; i++) {
            auto &buf = mod->buffers[i];
            if (buf.kind == kun::BufferKind::OUTPUT) {
                py::array_t<float, py::array::c_style> outbuffer;
                if (outputs.contains(buf.name)) {
                    outbuffer = outputs[buf.name].cast<py::array_t<float, py::array::c_style>>();
                    if (outbuffer.ndim() != expected_out_shape->size()) {
                        
                    }
                } else {
                    outbuffer = py::array_t<float, py::array::c_style>{expected_out_shape};
                }
                ret[buf.name] = outbuffer;
                bufs[buf.name] = (float *)outbuffer.request().ptr;
            }
        }
        kun::runGraph(exec, mod, bufs, known_S * kun::simd_len, known_T, cur_time, length);
        return ret;
    });
}