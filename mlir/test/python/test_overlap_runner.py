#!/usr/bin/env python3
# RUN: %python %s
# RUN: %python %s --use-cuda-graph
# REQUIRES: cuda-device
"""Regression test for KunQuantMLIR.OverlapRunner.

The test submits more runs than there are runner slots, while changing both
the time length and the stock count. This exercises slot reuse, cached device
output reallocation, CUDA graph state rebuild/update, and the host output block
that is returned as per-output NumPy slices.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass

import numpy as np


SAMPLE_KUNIR = textwrap.dedent("""
gpu.module @kungpu_kernels {
  kunir.func @overlap_runner_kernel(%a: !kunir.ts<f32, inf>, %b: !kunir.ts<f32, inf>)
      inputs {%a = "a", %b = "b"}
      outputs {"sum", "diff"}
      target {occupancy = 1, warps_per_cta = 4, smem_size = 49152, vector_size = 1} unreliable_count = 0
      -> (!kunir.ts<f32, 1>, !kunir.ts<f32, 1>) {
    %sum = kunir.add %a, %b : !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
    %diff = kunir.sub %a, %b : !kunir.ts<f32, inf>, !kunir.ts<f32, inf>
    kunir.return %sum, %diff : !kunir.ts<f32, 1>, !kunir.ts<f32, 1>
  }
}
""").strip()


@dataclass
class Case:
    label: str
    time_length: int
    num_stocks: int
    length_arg: int


def make_inputs(case: Case, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "a": rng.standard_normal(
            (case.time_length, case.num_stocks), dtype=np.float32),
        "b": rng.standard_normal(
            (case.time_length, case.num_stocks), dtype=np.float32),
    }


def expected_outputs(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "sum": inputs["a"] + inputs["b"],
        "diff": inputs["a"] - inputs["b"],
    }


def check_outputs(case: Case, actual: dict[str, np.ndarray],
                  expected: dict[str, np.ndarray]) -> None:
    expected_shape = (case.time_length, case.num_stocks)
    assert set(actual) == {"sum", "diff"}, actual.keys()
    for name in ("sum", "diff"):
        arr = actual[name]
        assert arr.shape == expected_shape, (case, name, arr.shape)
        assert arr.dtype == np.float32, (case, name, arr.dtype)
        assert arr.flags.c_contiguous, (case, name, arr.strides)
        assert not arr.flags.owndata, (case, name)
        np.testing.assert_allclose(arr, expected[name], rtol=1e-6, atol=1e-6)


def build_cases(base_time: int, base_stocks: int) -> list[Case]:
    return [
        Case("infer-initial", base_time, base_stocks, 0),
        Case("explicit-length-change", base_time + 7, base_stocks,
             base_time + 7),
        Case("stock-count-change", max(8, base_time - 5),
             base_stocks + 37, max(8, base_time - 5)),
        Case("infer-both-change", base_time + 3, base_stocks + 79, 0),
        Case("explicit-shorter-shape", max(8, base_time // 2),
             max(8, base_stocks - 11), max(8, base_time // 2)),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    ap.add_argument("-T", "--time-length", type=int, default=32)
    ap.add_argument("-S", "--num-stocks", type=int, default=257)
    ap.add_argument("--use-cuda-graph", action="store_true")
    args = ap.parse_args()

    from KunQuant.jit import KunMLIR
    from KunQuant.jit.cuda import find_cuda_toolkit
    from KunQuant.jit.env import get_cuda_compute_capability
    from KunQuantMLIR.OverlapRunner import OverlapRunner

    import cupy as cp

    args.target = args.target or get_cuda_compute_capability()
    cp.cuda.Device(0).use()
    _ = cp.zeros((1,), dtype=cp.float32)

    mod = KunMLIR.parse(SAMPLE_KUNIR)
    exe = KunMLIR.compile(mod,
                          graph_inputs=["a", "b"],
                          graph_outputs=["sum", "diff"],
                          gpu_arch=args.target, opt_level=3,
                          toolkit_path=find_cuda_toolkit())
    assert exe.output_names == ["sum", "diff"], exe.output_names

    compute_stream = cp.cuda.Stream(non_blocking=True)
    executor = KunMLIR.Executor(stream=compute_stream)
    runner = OverlapRunner(exe, executor, num_slots=2)

    print("=== overlap runner ===")
    print(f"  target={args.target}  use_cuda_graph={args.use_cuda_graph}")
    print(f"  executor.stream={hex(executor.stream)}")

    pending = []
    for i, case in enumerate(build_cases(args.time_length, args.num_stocks)):
        inputs = make_inputs(case, seed=100 + i)
        result = runner.submit(inputs,
                               length=case.length_arg,
                               use_cuda_graph=args.use_cuda_graph)
        pending.append((case, inputs, result))
        print(f"  submitted {case.label}: T={case.time_length}, "
              f"S={case.num_stocks}, length={case.length_arg}")

    for case, inputs, result in pending:
        actual = result.wait()
        check_outputs(case, actual, expected_outputs(inputs))
        print(f"  ok {case.label}")

    runner.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
