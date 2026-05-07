#pragma once

#include <memory>

namespace mlir { class Pass; }

namespace kunir {
void registerKunIrToKunGpuPass();
std::unique_ptr<::mlir::Pass> createKunIrToKunGpuPass();
} // namespace kunir
