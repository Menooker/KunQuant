"""Pipelined CUDA runner for overlapping copies with KunMLIR launches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

try:
    import cupy as cp
except ImportError as exc:
    raise ImportError(
        "KunQuantMLIR.OverlapRunner requires CuPy for asynchronous CUDA "
        "H2D/D2H copies. Install a CuPy build matching your CUDA runtime, "
        "for example cupy-cuda12x."
    ) from exc

from KunQuantMLIR import KunMLIR


CudaStream = Union[cp.cuda.Stream, cp.cuda.ExternalStream]


class DLPackProvider(Protocol):
    def __dlpack__(self) -> object:
        ...


def _stream_from_executor(executor: KunMLIR.Executor) -> CudaStream:
    if executor.stream == 0:
        return cp.cuda.Stream.null
    return cp.cuda.ExternalStream(executor.stream)


@dataclass
class PendingResult:
    """Host outputs from an asynchronous D2H copy.

    The NumPy arrays are intentionally kept alive by this object until the
    completion event has passed. Call ``wait()`` before reading them.
    """

    _output_names: List[str]
    _host_output_block: np.ndarray
    _done_event: cp.cuda.Event
    _outputs: Optional[Dict[str, np.ndarray]] = None

    def wait(self) -> Dict[str, np.ndarray]:
        self._done_event.synchronize()
        if self._outputs is None:
            self._outputs = {
                name: self._host_output_block[i]
                for i, name in enumerate(self._output_names)
            }
        return self._outputs


@dataclass
class _Slot:
    index: int
    dev_inputs: Dict[str, cp.ndarray]
    dev_outputs: Optional[Dict[str, cp.ndarray]]
    output_length: Optional[int]
    output_num_stocks: Optional[int]
    graph_executable: Optional[KunMLIR.Executable]
    h2d_event: cp.cuda.Event
    compute_event: cp.cuda.Event
    free_event: Optional[cp.cuda.Event]
    h2d_sources: Optional[Dict[str, np.ndarray]]
    host_output_block: Optional[np.ndarray]


class OverlapRunner:
    """Submit KunMLIR ``runGraph`` calls through a copy/compute pipeline.

    The runner owns three non-blocking streams shared by all slots:
    one for H2D input copies, one for the KunMLIR executor, and one for
    D2H output copies. Slots only own reusable device input/output buffers
    and references that must stay alive while async copies are in flight.
    """

    def __init__(self, executable: KunMLIR.Executable,
                 executor: KunMLIR.Executor,
                 num_slots: int = 3) -> None:
        if num_slots < 2:
            raise ValueError("OverlapRunner requires at least two slots")

        self.executable = executable
        self.executor = executor
        self.compute_stream = _stream_from_executor(executor)
        self.output_names = list(executable.output_names)

        self.h2d_stream = cp.cuda.Stream(non_blocking=True)
        self.d2h_stream = cp.cuda.Stream(non_blocking=True)

        self._slots: List[_Slot] = [
            _Slot(
                index=i,
                dev_inputs={},
                dev_outputs=None,
                output_length=None,
                output_num_stocks=None,
                graph_executable=None,
                h2d_event=cp.cuda.Event(),
                compute_event=cp.cuda.Event(),
                free_event=None,
                h2d_sources=None,
                host_output_block=None,
            )
            for i in range(num_slots)
        ]
        self._next_slot = 0

    @property
    def num_slots(self) -> int:
        return len(self._slots)

    def submit(self, inputs: Dict[str, np.ndarray], cur_time: int = 0,
               length: int = 0, mask: int = 0,
               min_chunk_warmup_factor: int = 4,
               sm_fill_factor: float = 1.5,
               use_cuda_graph: bool = False) -> PendingResult:
        slot = self._slots[self._next_slot]
        self._next_slot = (self._next_slot + 1) % len(self._slots)

        if slot.free_event is not None:
            slot.free_event.synchronize()

        host_inputs = self._prepare_host_inputs(inputs)
        slot.h2d_sources = host_inputs
        run_inputs, inputs_resized = self._copy_inputs_to_device(
            slot, host_inputs)
        output_length, num_stocks, output_dtype = self._output_spec(
            host_inputs, length)
        dev_outputs = self._cached_outputs_for_run(
            slot, output_length, num_stocks, output_dtype, inputs_resized)

        h2d_done = slot.h2d_event
        h2d_done.record(self.h2d_stream)
        slot.free_event = h2d_done

        self.compute_stream.wait_event(h2d_done)
        executable = self._executable_for_slot(slot, use_cuda_graph)
        try:
            ret = self.executor.runGraph(
                executable,
                run_inputs,
                cur_time=cur_time,
                length=length,
                outputs=dev_outputs,
                mask=mask,
                min_chunk_warmup_factor=min_chunk_warmup_factor,
                sm_fill_factor=sm_fill_factor,
                use_cuda_graph=use_cuda_graph,
            )
        except Exception:
            compute_done = slot.compute_event
            compute_done.record(self.compute_stream)
            slot.free_event = compute_done
            raise

        compute_done = slot.compute_event
        compute_done.record(self.compute_stream)
        slot.free_event = compute_done
        slot.dev_outputs = self._to_cupy_outputs(ret)

        self.d2h_stream.wait_event(compute_done)
        host_output_block: Optional[np.ndarray] = None
        try:
            host_output_block = self._allocate_pinned_output_block(
                slot.dev_outputs)
            for i, name in enumerate(self.output_names):
                dev_output = slot.dev_outputs[name]
                cp.asnumpy(dev_output, stream=self.d2h_stream,
                           out=host_output_block[i], blocking=False)
        finally:
            d2h_done = cp.cuda.Event()
            d2h_done.record(self.d2h_stream)
            slot.free_event = d2h_done
            slot.host_output_block = host_output_block

        result = PendingResult(self.output_names, host_output_block, d2h_done)
        return result

    def synchronize(self) -> None:
        for slot in self._slots:
            if slot.free_event is not None:
                slot.free_event.synchronize()
            slot.h2d_sources = None
            slot.host_output_block = None

    def _prepare_host_inputs(self, inputs: Dict[str, np.ndarray]
                             ) -> Dict[str, np.ndarray]:
        host_inputs: Dict[str, np.ndarray] = {}
        for name, value in inputs.items():
            arr = np.asarray(value)
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            host_inputs[name] = arr
        return host_inputs

    def _copy_inputs_to_device(self, slot: _Slot,
                               host_inputs: Dict[str, np.ndarray]
                               ) -> Tuple[Dict[str, cp.ndarray], bool]:
        run_inputs: Dict[str, cp.ndarray] = {}
        resized = False
        for name, host in host_inputs.items():
            dev = slot.dev_inputs.get(name)
            if dev is None or dev.shape != host.shape or dev.dtype != host.dtype:
                dev = cp.empty(host.shape, dtype=host.dtype)
                slot.dev_inputs[name] = dev
                resized = True
            dev.set(host, stream=self.h2d_stream)
            run_inputs[name] = dev
        return run_inputs, resized

    def _output_spec(self, host_inputs: Dict[str, np.ndarray],
                     length: int) -> Tuple[int, int, np.dtype]:
        if not host_inputs:
            raise ValueError("OverlapRunner requires at least one input")
        first = next(iter(host_inputs.values()))
        if first.ndim != 2:
            raise ValueError("OverlapRunner expects 2-D TS inputs")
        output_length = first.shape[0] if length == 0 else length
        return output_length, first.shape[1], first.dtype

    def _cached_outputs_for_run(self, slot: _Slot,
                                length: int,
                                num_stocks: int,
                                dtype: np.dtype,
                                inputs_resized: bool
                                ) -> Dict[str, cp.ndarray]:
        needs_alloc = (
            slot.dev_outputs is None or
            slot.output_length != length or
            slot.output_num_stocks != num_stocks or
            inputs_resized
        )
        if needs_alloc:
            slot.dev_outputs = {
                name: cp.empty((length, num_stocks), dtype=dtype)
                for name in self.output_names
            }
            slot.output_length = length
            slot.output_num_stocks = num_stocks
        return slot.dev_outputs

    def _allocate_pinned_output_block(
        self, dev_outputs: Dict[str, cp.ndarray]
    ) -> np.ndarray:
        if not self.output_names:
            raise ValueError("OverlapRunner requires at least one output")
        first = dev_outputs[self.output_names[0]]
        shape = first.shape
        dtype = np.dtype(first.dtype)
        pinned = cp.cuda.alloc_pinned_memory(len(self.output_names) *
                                            first.nbytes)
        return np.frombuffer(
            pinned, dtype=dtype, count=len(self.output_names) * first.size
        ).reshape((len(self.output_names),) + shape)

    def _to_cupy_outputs(self, outputs: Dict[str, Union[cp.ndarray,
                                                        DLPackProvider]]
                         ) -> Dict[str, cp.ndarray]:
        ret: Dict[str, cp.ndarray] = {}
        for name, value in outputs.items():
            if isinstance(value, cp.ndarray):
                ret[name] = value
            else:
                ret[name] = cp.from_dlpack(value)
        return ret

    def _executable_for_slot(self, slot: _Slot,
                             use_cuda_graph: bool) -> KunMLIR.Executable:
        if not use_cuda_graph:
            return self.executable
        if slot.graph_executable is None:
            slot.graph_executable = self.executable.clone()
        return slot.graph_executable
