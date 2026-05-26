from KunQuant.Driver import KunCompilerConfig
import numpy as np
import sys
import time
import os
import argparse
from typing import Dict
from KunQuant.jit import cfake
from KunQuant.runner import KunRunner as kr
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from KunQuant.predefined.Alpha158 import AllData
from KunQuant.jit.env import cpu_arch, get_cuda_compute_capability

isx86 = cpu_arch != "aarch64"


# Factor families the GPU backend can't compile yet (the underlying op
# has no kunir lowering).  We filter their Output ops out of the Function
# before compileit on the GPU path — the rest of alpha158 compiles fine.
#   QTLU / QTLD → WindowedQuantile → SkipList (CPU-only).
_GPU_SKIP_FACTOR_PREFIXES = ("QTLU", "QTLD")


def _filter_outputs_for_gpu(f: Function) -> None:
    """Remove Output ops whose name starts with a `_GPU_SKIP_FACTOR_PREFIXES`
    entry.  Mutates `f` in place via `set_ops`.  The dropped intermediate
    compute ops are GC'd as part of the downstream optimization pipeline
    (anything with no remaining user is dead)."""
    kept = []
    dropped = []
    for op in f.ops:
        if isinstance(op, Output):
            name = op.attrs.get("name", "")
            if any(name.startswith(p) for p in _GPU_SKIP_FACTOR_PREFIXES):
                dropped.append(name)
                continue
        kept.append(op)
    if dropped:
        print(f"[gpu] dropping {len(dropped)} unsupported outputs: "
              f"{sorted(set(n.rstrip('0123456789') for n in dropped))}")
    f.set_ops(kept)


def check_alpha158(avx512, keep, tempdir, gpu_arch=""):
    builder = Builder()
    with builder:
        pack_158 = AllData(low=Input("low"), high=Input("high"), close=Input(
            "close"), open=Input("open"), amount=Input("amount"), volume=Input("volume"))
        alpha158, names = pack_158.build({
            'kbar': {},  # whether to use some hard-code kbar features
            "price": {
                "windows": [0],
                "feature": [("OPEN", pack_158.open), ("HIGH", pack_158.high), ("LOW", pack_158.low), ("VWAP", pack_158.vwap)],
            },
            # 'volume': { # whether to use raw volume features
            #     'windows': [0, 1, 2, 3, 4], # use volume at n days ago
            # },
            'rolling': {  # whether to use rolling operator based features
                'windows': [5, 10, 20, 30, 60],  # rolling windows size
                # if include is None we will use default operators
                # 'exclude': ['RANK'], # rolling operator not to use
            }
        })
        for v, k in zip(alpha158, names):
            Output(v, k)
    print("Total names: ", len(names))
    f = Function(builder.ops)
    if gpu_arch:
        _filter_outputs_for_gpu(f)
        from KunQuant.jit import cuda as _cuda_jit
        target = [("alpha158", f, KunCompilerConfig(
            dtype='double', blocking_len=1, partition_factor=2,
            output_layout="TS", input_layout="TS",
            options={"opt_reduce": True, "fast_log": True,
                     'no_fast_stat': 'no_warn'}))]
        ccfg = _cuda_jit.CudaCompilerConfig(gpu_arch=gpu_arch)
        return _cuda_jit.compileit(target, "testalpha158", ccfg)

    if avx512:
        simd_len = 8
    elif isx86:
        simd_len = 4
    else:
        simd_len = 2
    target = [("alpha158", f, KunCompilerConfig(dtype='double', blocking_len=simd_len, partition_factor=4,
               output_layout="TS", input_layout="TS", options={"opt_reduce": True, "fast_log": True,
                                                                'no_fast_stat': 'no_warn'}))]
    if avx512:
        machine = cfake.X64CPUFlags(avx512=True, avx512dq=True, avx512vl=True)
    else:
        machine = cfake.NativeCPUFlags()
    return cfake.compileit(target, "testalpha158", cfake.CppCompilerConfig(machine=machine), tempdir=tempdir, keep_files=keep, load=not avx512)


num_stock = 8
num_time = 260


def load(inputs, ref):
    return dict(np.load(inputs)), dict(np.load(ref))


def ST_TS(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data.transpose()).astype('float64')


# ── Backend shims ───────────────────────────────────────────────────
#
# CPU and GPU have the same conceptual flow (prepare → execute → fetch),
# they differ only in the runtime calls and where the buffers live.
# Wrap each backend in a tiny object exposing the three methods so the
# `test()` body stays single-source.

class _CpuBackend:
    def __init__(self, lib: kr.Library, modname: str):
        self.modu = lib.getModule(modname)
        self.start_window = self.modu.getOutputUnreliableCount()
        self.outnames = self.modu.getOutputNames()
        self.executor = kr.createMultiThreadExecutor(8)
        # Pre-allocated NaN-filled output buffers; the CPU runtime writes
        # in place and we hand the same dict back to `_compare`.
        self._outbuffers = {}
        sharedbuf = np.empty((len(self.outnames), num_time, num_stock),
                              dtype="float64")
        sharedbuf[:] = np.nan
        for idx, name in enumerate(self.outnames):
            self._outbuffers[name] = sharedbuf[idx]

    def prepare_input(self, host_input):
        return host_input

    def execute(self, inputs):
        kr.runGraph(self.executor, self.modu, inputs, 0, num_time,
                     self._outbuffers)

    def fetch_output(self):
        return self._outbuffers


class _GpuBackend:
    def __init__(self, lib, modname: str):
        import cupy as cp
        from KunQuant.jit import KunMLIR as _kr_mlir
        self._cp = cp
        self.modu = lib.getModule(modname)
        self.start_window = self.modu.getOutputUnreliableCount()
        self.outnames = self.modu.output_names
        self.executor = _kr_mlir.Executor()
        self._raw = None

    def prepare_input(self, host_input):
        return {k: self._cp.asarray(v) for k, v in host_input.items()}

    def execute(self, inputs):
        self._raw = self.executor.runGraph(self.modu, inputs)
        self.executor.synchronize()

    def fetch_output(self):
        cp = self._cp
        out = {}
        for k in self.outnames:
            v = self._raw[k]
            arr = v if isinstance(v, cp.ndarray) else cp.from_dlpack(v)
            out[k] = cp.asnumpy(arr)
        return out


def _compare(outbuffers, ref, start_window, rtol, atol):
    for k, v in outbuffers.items():
        s = start_window[k]
        if not np.allclose(v[s:], ref[k][s:], rtol=rtol, atol=atol, equal_nan=True):
            print("Correctness check failed at " + k)
            for sid in range(num_stock):
                print("Check stock", sid)
                myout = v.transpose()[sid, s:]
                refv = ref[k].transpose()[sid, s:]
                if not np.allclose(myout, refv, rtol=rtol, atol=atol, equal_nan=True):
                    for j in range(num_time-s):
                        if not np.allclose(myout[j], refv[j], rtol=rtol, atol=atol, equal_nan=True):
                            print("j", j, myout[j], refv[j])
                    exit(1)


def test(backend, inputs: Dict[str, np.ndarray],
          ref: Dict[str, np.ndarray]) -> None:
    rtol = 1e-4
    atol = 1e-5
    print("Total num alphas", len(backend.outnames))
    host_input = {"high": ST_TS(inputs['dhigh']), "low": ST_TS(inputs['dlow']),
                  "close": ST_TS(inputs['dclose']), "open": ST_TS(inputs['dopen']),
                  "volume": ST_TS(inputs['dvol']), "amount": ST_TS(inputs['damount'])}
    be_input = backend.prepare_input(host_input)
    start = time.time()
    backend.execute(be_input)
    end = time.time()
    print(f"Exec takes: {end-start:.6f} seconds")
    _compare(backend.fetch_output(), ref, backend.start_window, rtol, atol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Run and check alpha158 again pre-computed result")
    parser.add_argument("--inputs", required=True, type=str,
                        help="The path to the input npz file")
    parser.add_argument("--ref", required=True, type=str,
                        help="The path to the reference output npz file")
    parser.add_argument("--action", required=True, type=str,
                        help="One of: compile_avx512, run_avx512, run_native, run_gpu")
    parser.add_argument("--gpu-arch", default="sm_80", type=str,
                        help="GPU compute capability for --action=run_gpu (e.g. sm_80)")
    args = parser.parse_args()
    if args.action == "compile_avx512":
        check_alpha158(True, True, "./build")
        exit(0)
    elif args.action == "run_avx512":
        lib = kr.Library.load(os.path.join("./build/testalpha158", "testalpha158.so"))
        inp, ref = load(args.inputs, args.ref)
        test(_CpuBackend(lib, "alpha158"), inp, ref)
    elif args.action == "run_gpu":
        # Touch the cupy allocator before compileit so the primary CUDA
        # context exists when KunMLIR.compile inherits it.
        import cupy as cp
        cp.cuda.Device(0).use()
        cp.zeros((1,), dtype=cp.float64)
        if args.gpu_arch == "auto":
            args.gpu_arch = get_cuda_compute_capability()
        lib = check_alpha158(False, False, None, gpu_arch=args.gpu_arch)
        inp, ref = load(args.inputs, args.ref)
        test(_GpuBackend(lib, "alpha158"), inp, ref)
    else:
        lib = check_alpha158(False, False, None)
        inp, ref = load(args.inputs, args.ref)
        test(_CpuBackend(lib, "alpha158"), inp, ref)
    print("done")
