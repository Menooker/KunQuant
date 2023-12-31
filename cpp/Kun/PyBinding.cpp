#include <pybind11/pybind11.h>
#include "Context.hpp"
#include "Module.hpp"

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(KunRunner, m) {
    m.doc() = R"(Code Runner for KunQuant generated code)";

    m.def("add", &add);
    py::class_<kun::Executor>(m, "Executor");
    m.def("createSingleThreadExecutor", &kun::createSingleThreadExecutor);

    py::class_<kun::Module>(m, "Module");
    py::class_<kun::Library>(m, "Library")
        .def_static("load", &kun::Library::load)
        .def("getModule", &kun::Library::getModule, py::return_value_policy::reference);
}