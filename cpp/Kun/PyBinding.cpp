#include "Context.hpp"
#include "Module.hpp"
#include "RunGraph.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

int add(int i, int j) { return i + j; }

namespace py = pybind11;

PYBIND11_MODULE(KunRunner, m) {
    m.doc() = R"(Code Runner for KunQuant generated code)";

    m.def("add", &add);
    py::class_<kun::Executor>(m, "Executor");
    m.def("createSingleThreadExecutor", &kun::createSingleThreadExecutor);

    py::class_<kun::Module>(m, "Module");
    py::class_<kun::Library>(m, "Library")
        .def_static("load", &kun::Library::load)
        .def("getModule", &kun::Library::getModule,
             py::return_value_policy::reference);
    m.def("runGraph",
          [](std::shared_ptr<kun::Executor> exec, const kun::Module *mod,
             const py::dict inputs, size_t cur_time) {
                std::unordered_map<std::string, float*> bufs;
                for(auto kv: inputs) {
                    auto name = py::cast<std::string>(kv.first);
                    auto buf_obj = py::cast<py::buffer>(kv.second);
                    auto info = buf_obj.request();
                    if (info.format != py::format_descriptor<float>::format()) {
                        throw std::runtime_error("Expecting float buffer at " + name);
                    }
                    // ST8t layout
                    if (info.ndim != 3 || info.shape.back() != 8) {
                        throw std::runtime_error("Bad shape at " + name);
                    }
                    // fix-me: other checks
                    bufs[name] = (float*)info.ptr;
                }
          });
}