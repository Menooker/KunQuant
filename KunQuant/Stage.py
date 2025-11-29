from KunQuant.ops import ReduceOp
from KunQuant.Op import OpBase, Output, ReductionOp, SinkOpTrait, Input
from typing import List, Dict
from collections import OrderedDict
import typing
from dataclasses import dataclass


@dataclass
class OpInfo:
    idx: int
    uses: typing.OrderedDict[OpBase, int]


class Function:
    ops: List[OpBase]
    op_to_id: Dict[OpBase, OpInfo]
    # whether to check the windowed op's inputs are WindowedDataSource.
    # We allow non-WindowedDataSource as windowed op's inputs before decompose pass
    strict_window: bool
    name: str

    def get_op_idx(self, op: OpBase):
        return self.op_to_id[op].idx

    @staticmethod
    def _topo_insert_op(op: OpBase, out: List[OpBase], all_ops, met, additional_dep) -> None:
        if op not in all_ops:
            raise RuntimeError("dependent Op not in the function: " + str(op))
        if op in met:
            return
        if op in additional_dep:
            for inp in additional_dep[op]:
                Function._topo_insert_op(inp, out, all_ops, met, additional_dep)
        for inp in op.inputs:
            Function._topo_insert_op(inp, out, all_ops, met, additional_dep)
        out.append(op)
        met.add(op)

    @staticmethod
    def topo_sort_ops(ops: List[OpBase]) -> List[OpBase]:
        all_ops = set(ops)
        met = set()
        out = []
        additional_dep = dict()
        for op in ops:
            parent = op.get_parent()
            if not parent and isinstance(op, ReductionOp):
                parent = op.get_loop()
            if parent:
                if parent not in additional_dep:
                    additional_dep[parent] = set()
                for inp in op.inputs:
                    if inp != parent and inp.get_parent() != parent:
                        additional_dep[parent].add(inp)

        for op in ops:
            Function._topo_insert_op(op, out, all_ops, met, additional_dep)
        return out

    @staticmethod
    def _garbage_collect(ops: List[OpBase], op_to_id: Dict[OpBase, OpInfo]) -> List[OpBase]:
        need_del: List[OpBase] = []
        for op in ops:
            if not isinstance(op, SinkOpTrait) and len(op_to_id[op].uses) == 0:
                need_del.append(op)
        if len(need_del) == 0:
            return ops
        while len(need_del):
            del_op = need_del.pop()
            for inp in del_op.inputs:
                op_info = op_to_id[inp]
                if del_op in op_info.uses:
                    del op_info.uses[del_op]
                    if not isinstance(inp, SinkOpTrait) and len(op_info.uses) == 0:
                        need_del.append(inp)
        newops = []
        for op in ops:
            info = op_to_id[op]
            if isinstance(op, SinkOpTrait) or len(info.uses) > 0:
                newops.append(op)
                info.idx = len(newops) - 1
        return newops

    def set_ops(self, ops: List[OpBase]) -> None:
        op_to_id: Dict[OpBase, OpInfo] = dict()
        for idx, op in enumerate(ops):
            if op in op_to_id:
                raise RuntimeError("Duplicated ops: " + str(op))
            for inp in op.inputs:
                if not isinstance(inp, OpBase):
                    raise RuntimeError(f"Bad input: {str(inp)} in {op.__class__.__name__}. op.inputs={op.inputs}")
                if inp not in op_to_id:
                    raise RuntimeError(
                        f"Bad op: {str(op)}, because the input has not been executed: {str(inp)}")
                op_to_id[inp].uses[op] = 1

            if op.get_parent() and op.get_parent() not in op_to_id:
                raise RuntimeError(
                    f"Bad op: {str(op)}, because the input has not been executed: {str(op.get_parent())}")
            op_to_id[op] = OpInfo(idx, OrderedDict())
        self.op_to_id = op_to_id
        for idx, op in enumerate(ops):
            op.verify(self)

        # garbage collect
        self.ops = Function._garbage_collect(ops, op_to_id)
        self.op_to_id = op_to_id

    def single_op_str(self, idx: int) -> str:
        op = self.ops[idx]
        args = ",".join([f"v{self.get_op_idx(inp)}" for inp in op.inputs])
        in_clause = ""
        if op.get_parent():
            in_clause = f" in v{self.get_op_idx(op.get_parent())}"
        return f"v{idx} = {op.__class__.__name__}@{op.attrs_str()}({args}){in_clause}"

    def __str__(self) -> str:
        lines = []
        if self.name:
            lines.append(f"name = {self.name}")
        for i in range(len(self.ops)):
            lines.append(self.single_op_str(i))
        return "\n".join(lines)

    def __init__(self, ops: List[OpBase], strict_window: bool = False, name: str = "") -> None:
        self.strict_window = strict_window
        self.name = name
        self.set_ops(ops)
