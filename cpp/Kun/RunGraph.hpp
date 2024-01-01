#pragma once

#include "Context.hpp"
#include "Module.hpp"
#include <unordered_map>

namespace kun {
void runGraph(std::shared_ptr<Executor> exec, Module *m,
              std::unordered_map<std::string, float *> &buffers,
              size_t num_stocks, size_t total_time, size_t cur_time);

}