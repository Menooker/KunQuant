#include <pybind11/pybind11.h>
#include "Context.hpp"

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(KunRunner, m) {
    m.doc() = R"(Code Runner for KunQuant generated code)";

    m.def("add", &add);
    py::class_<kun::Executor>(m, "Executor");
}