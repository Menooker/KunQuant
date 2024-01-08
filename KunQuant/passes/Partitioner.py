from KunQuant.Op import OpBase, Output, Input, Rank, GraphSourceTrait, ConstantOp
from KunQuant.Stage import Function, OpInfo
from KunQuant.ops import GenericPartition
from typing import List, Dict, Set, Tuple
import typing
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class _PartitionOpInfo:
    depender: Set[OpBase]
    pending_dep: Dict[OpBase, None]

@dataclass
class _Partition:
    ops: typing.OrderedDict[OpBase, None]
    depender: Set[OpBase]
    num_outputs = 0
    depending: typing.OrderedDict['_Partition', None] = None
    stage_op: GenericPartition = None
    impl_func: Function = None

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self) -> int:
        return id(self)

    def add(self, info: Dict[OpBase, _PartitionOpInfo], op: OpBase):
        self.ops[op] = None
        # input op is replicated in all partitions, referencing it will not cause cylic dependency
        if not isinstance(op, GraphSourceTrait):
            if info is not None:
                self.depender.update(info[op].depender)
            if isinstance(op, Output):
                self.num_outputs += 1

    def is_op_not_in_and_depending(self, op: OpBase):
        if op in self.ops:
            return False
        return op in self.depender

def _collect_op_info(f: Function)-> Dict[OpBase, _PartitionOpInfo]:
    ret: Dict[OpBase, _PartitionOpInfo] = dict()
    for op in reversed(f.ops):
        finfo = f.op_to_id[op]
        depender = set()
        for user in finfo.uses:
            depender.add(user)
            depender.update(f.op_to_id[user].uses)
        ret[op] = _PartitionOpInfo(depender, dict([(inp, None) for inp in op.inputs]))
    return ret

def _select_next(ready_ops: List[OpBase], info: Dict[OpBase, _PartitionOpInfo], partiton: _Partition) -> OpBase:
    '''
    Select the next op to put into the partition.
    1. If there is Rank Op, always select it first
    2. If the op is directly linked to the current partiton, it is more likely to be selected
    3. The larger index in read_ops, the more likely to be selected (Try Depth-first-search-like)
    4. The less depender, the more likely to be selected
    '''
    cur_best = 999999999999999999
    cur_best_op: OpBase = None
    for idx, op in enumerate(ready_ops):
        if isinstance(op, Rank):
            return op
        # correctness check, if the partition has another path to the op and the path itself is not
        # in the partition, the op cannot be in the partition. Otherwise there will be a loop in the
        # partition dependency graph
        connected_to_parti = False
        has_bad_input = False
        for inp in op.inputs:
            if inp in partiton.ops:
                connected_to_parti = True
            else:
                if partiton.is_op_not_in_and_depending(inp):
                    has_bad_input = True
        if has_bad_input:
            continue
        cur_cost = 0
        if connected_to_parti:
            cur_cost -= 10000
        cur_cost += (len(ready_ops) - idx) * 10
        cur_cost += len(info[op].depender)
        if cur_cost <= cur_best:
            cur_best = cur_cost
            cur_best_op = op
    return cur_best_op
        
def _partition(f: Function, partition_thres = 3) -> List[_Partition]:
    opinfo = _collect_op_info(f)
    partitions: List[_Partition] = []
    ready_ops = list(filter(lambda op: isinstance(op, GraphSourceTrait), f.ops))
    to_visit = dict([(op, None) for op in f.ops])
    while len(ready_ops):
        partition = _Partition(OrderedDict(), set())
        # print("============\nnew partition:", partition)
        selected = _select_next(ready_ops, opinfo, partition)
        while selected:
            # maintain the ready queue for topology sort
            for user in f.op_to_id[selected].uses:
                pendingset = opinfo[user].pending_dep
                del pendingset[selected]
                if len(pendingset) == 0:
                    ready_ops.append(user)
            del to_visit[selected]
            ready_ops.remove(selected)
            if isinstance(selected, GraphSourceTrait):
                # don't put input in partition yet
                pass
            elif isinstance(selected, Rank):
                single_partition = _Partition(OrderedDict(), set())
                single_partition.add(opinfo, selected)
                # if an output op is directly connected with Rank, merge it in the partition
                for user in f.op_to_id[selected].uses:
                    if isinstance(user, Output):
                        ready_ops.remove(user)
                        single_partition.add(opinfo, user)
                        del to_visit[user]
                partitions.append(single_partition)
            else:
                # add op to partition
                for inp in selected.inputs:
                    # only put the referenced inputs into partition
                    if isinstance(inp, GraphSourceTrait):
                        partition.add(opinfo, inp)
                partition.add(opinfo, selected)
                # print("@@@add ", selected)
            if partition.num_outputs > partition_thres:
                # too many outputs visited, make a new partition
                break
            selected = _select_next(ready_ops, opinfo, partition)
        if partition.ops.__len__():
            partitions.append(partition)
    assert(to_visit.__len__()==0)
    return partitions

def _search_output_use(op: OpBase, info: OpInfo) -> OpBase:
    if isinstance(op, Output):
        return op
    for use in info.uses:
        if isinstance(use, Output):
            return use
    return None

def _transform_partitions(partitions: List[_Partition], f: Function) -> Tuple[Function, List[Function]]:
    naming_table = set()
    for op in f.ops:
        if isinstance(op, Input) or isinstance(op, Output):
            name = op.attrs["name"]
            if name == "" or name in naming_table:
                raise RuntimeError("Duplicated or bad name of op: " + str(op))
            naming_table.add(name)
    def add_to_naming_table(v: str) -> str:
        nonlocal naming_table
        cur_name = v
        idx = 0
        while cur_name in naming_table:
            cur_name = f"{v}_{idx}"
            idx += 1
        naming_table.add(cur_name)
        return cur_name
    # now partitions are built. We should build the dependencies between partitions
    op_lookup_table: Dict[OpBase, _Partition] = dict()
    for p in partitions:
        for op in p.ops:
            if not isinstance(op, GraphSourceTrait):
                # input is shared by all ops
                assert(op not in op_lookup_table)
                op_lookup_table[op] = p
    for p in partitions:
        name_to_input = dict()
        depending : typing.OrderedDict[_Partition, None] = OrderedDict()
        # for each op in partition
        for op in list(p.ops):
            for idx, inp in enumerate(op.inputs):
                if inp not in p.ops:
                    # if the partition depends on an op of another partition
                    if inp.get_parent():
                        raise RuntimeError("Bad cross partition op: " + str(inp))
                    inp_info = f.op_to_id[inp]
                    if isinstance(inp, ConstantOp):
                        if op in inp_info.uses:
                            del inp_info.uses[op]
                        inop = ConstantOp(inp.attrs["value"])
                        p.add(None, inop)
                        op.inputs[idx] = inop
                        continue
                    if not isinstance(inp, Input):
                        outop = _search_output_use(inp, inp_info)
                        if not outop:
                            # add an output op to that partition
                            out_name = add_to_naming_table(inp.hash_hex())
                            outop = Output(inp, out_name)
                            inp_info.uses[outop] = 1
                            inp_partition = op_lookup_table[inp]
                            inp_partition.add(None, outop)
                            op_lookup_table[outop] = inp_partition
                        else:
                            out_name = outop.attrs["name"]
                            inp_partition = op_lookup_table[outop]
                        if inp_partition != p:
                            depending[inp_partition] = None
                    else:
                        out_name = inp.attrs["name"]
                    if op in inp_info.uses:
                        del inp_info.uses[op]
                    
                    inop = name_to_input.get(out_name, None)
                    if not inop:
                        inop = Input(out_name)
                        p.add(None, inop)
                        name_to_input[out_name] = inop
                    op.inputs[idx] = inop
        p.depending = depending
        p.stage_op = GenericPartition([], None)
    
    out_impl = []
    out_stages = []
    for p in partitions:
        ops = list(p.ops)
        ops = Function.topo_sort_ops(ops)
        func_name = []
        for op in ops:
            if isinstance(op, Output):
                func_name.append(op.attrs["name"])
        thename = "_".join(func_name)
        p.impl_func = Function(ops, True, thename)
        p.stage_op.attrs["name"] = thename
        out_stages.append(p.stage_op)
        out_impl.append(p.impl_func)
        for dep in p.depending:
            p.stage_op.inputs.append(dep.stage_op)
    out_stages = Function.topo_sort_ops(out_stages)
    return Function(out_stages), out_impl

def do_partition(f: Function, factor: int) -> Tuple[Function, List[Function]]:
    partitions = _partition(f, factor)
    return _transform_partitions(partitions, f)