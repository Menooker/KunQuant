//===- dlpack.h - Minimal vendored DLPack ABI (consumer-only) ----------===//
//
// Trimmed subset of dmlc/dlpack v0.8.  We only need to *consume* a
// `DLManagedTensor` produced by CuPy / PyTorch / JAX via the
// `__dlpack__()` protocol, so this header omits the producer-side
// helpers and the newer versioned form.  Vendored to keep the build
// dependency-free; full spec lives at https://github.com/dmlc/dlpack.
//
// Original license: Apache-2.0.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// `DLDeviceType` — where the tensor data sits.  We accept kDLCUDA and
/// kDLCUDAManaged; everything else (CPU, ROCm, Metal, …) is rejected.
typedef enum {
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAHost = 3,
  kDLOpenCL = 4,
  kDLVulkan = 7,
  kDLMetal = 8,
  kDLVPI = 9,
  kDLROCM = 10,
  kDLROCMHost = 11,
  kDLExtDev = 12,
  kDLCUDAManaged = 13,
  kDLOneAPI = 14,
  kDLWebGPU = 15,
  kDLHexagon = 16,
} DLDeviceType;

typedef struct {
  DLDeviceType device_type;
  int32_t      device_id;
} DLDevice;

/// `DLDataTypeCode` — element-kind dimension of the dtype triple.
typedef enum {
  kDLInt = 0,
  kDLUInt = 1,
  kDLFloat = 2,
  kDLOpaqueHandle = 3,
  kDLBfloat = 4,
  kDLComplex = 5,
  kDLBool = 6,
} DLDataTypeCode;

/// (code, bits, lanes).  E.g. f32 = {kDLFloat, 32, 1}; f64 = {kDLFloat, 64, 1}.
typedef struct {
  uint8_t  code;
  uint8_t  bits;
  uint16_t lanes;
} DLDataType;

/// Plain tensor descriptor — pointer + shape + dtype + device.
typedef struct {
  void       *data;
  DLDevice    device;
  int32_t     ndim;
  DLDataType  dtype;
  int64_t    *shape;
  int64_t    *strides;       ///< NULL → row-major contiguous
  uint64_t    byte_offset;
} DLTensor;

/// The wrapper exchanged via the unversioned PyCapsule named "dltensor".
/// The capsule's PyCapsule_Destructor calls `deleter(self)` when it is
/// GC'd — unless the consumer renamed the capsule to "used_dltensor",
/// in which case the destructor is a no-op and the consumer must call
/// `deleter` itself.
typedef struct DLManagedTensor {
  DLTensor dl_tensor;
  void    *manager_ctx;
  void   (*deleter)(struct DLManagedTensor *self);
} DLManagedTensor;

#ifdef __cplusplus
} // extern "C"
#endif
