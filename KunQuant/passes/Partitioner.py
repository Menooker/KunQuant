from KunQuant.Op import OpBase, Output, Input, CrossSectionalOp, GraphSourceTrait, ConstantOp, ReductionOp, BoolOpTrait
from KunQuant.ops.MiscOp import Accumulator, SetAccumulator, WindowedLinearRegressionConsumerTrait, WindowedLinearRegression, ReturnFirstValue
from KunQuant.Stage import Function, OpInfo
from KunQuant.ops import GenericPartition
from typing import List, Dict, Set, Tuple
from .Util import debug_mode
import typing
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class _PartitionOpInfo:
    depender: Set[OpBase]
    pending_dep: Dict[OpBase, None]

def _is_non_output_op(op: OpBase) -> bool:
    '''
    return true if the op cannot be at the edge of a partition as output
    '''
    if op.get_parent() is not None or isinstance(op, (WindowedLinearRegression, BoolOpTrait, Accumulator)):
        return True
    return False

def _is_fast_select_op(op: OpBase) -> bool:
    '''
    return true if the op should be selected ASAP
    '''
    if isinstance(op, (ReductionOp, WindowedLinearRegressionConsumerTrait, SetAccumulator, ReturnFirstValue)) or _is_non_output_op(op):
        return True
    else:
        # don't stop at boolean op
        for inpt in op.inputs:
            if isinstance(inpt, BoolOpTrait):
                return True
    return False

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
    
    def get_edge_ops(self, f: Function) -> Dict[OpBase, bool]:
        out = dict()
        for op in self.ops:
            if isinstance(op, GraphSourceTrait):
                continue
            is_in_loop = _is_non_output_op(op)
            for user in f.op_to_id[op].uses:
                if user not in self.ops:
                    out[user] = is_in_loop or out.get(user, False)
        return out

def _collect_op_info(f: Function)-> Dict[OpBase, _PartitionOpInfo]:
    ret: Dict[OpBase, _PartitionOpInfo] = dict()
    for op in reversed(f.ops):
        finfo = f.op_to_id[op]
        depender = set()
        for user in finfo.uses:
            depender.add(user)
            depender.update(ret[user].depender)
        ret[op] = _PartitionOpInfo(depender, dict([(inp, None) for inp in op.inputs]))
    return ret

def del_from_ready_op(ready_ops: List[Tuple[OpBase, int]], op: OpBase):
    for idx, tup in enumerate(ready_ops):
        if tup[0] == op:
            del ready_ops[idx]
            return
    raise RuntimeError("del_from_ready_op: OP not found")

def _is_bad_op_to_add(op: OpBase, partiton: _Partition):
    connected_to_parti = False
    has_bad_input = False
    for inp in op.inputs:
        if inp in partiton.ops:
            connected_to_parti = True
        else:
            if partiton.is_op_not_in_and_depending(inp):
                has_bad_input = True
    return has_bad_input, connected_to_parti

def _select_next(ready_ops: List[Tuple[OpBase, int]], info: Dict[OpBase, _PartitionOpInfo], partiton: _Partition, f: Function) -> OpBase:
    '''
    Select the next op to put into the partition.
    1. If there is CrossSectionalOp, always select it first
    2. If the op is directly linked to the current partiton, it is more likely to be selected
    3. The larger index in ready_ops, the more likely to be selected (Try Depth-first-search-like)
    4. The less depender, the more likely to be selected
    '''
    cur_best = (-1, -1, -1) # (is_loop, critical, score)
    cur_best_op: OpBase = None
    edge_ops = partiton.get_edge_ops(f)
    for op, idx in ready_ops:
        if isinstance(op, CrossSectionalOp):
            return op
        # correctness check, if the partition has another path to the op and the path itself is not
        # in the partition, the op cannot be in the partition. Otherwise there will be a loop in the
        # partition dependency graph
        has_bad_input, connected_to_parti = _is_bad_op_to_add(op, partiton)
        if has_bad_input:
            continue
        op_info = info[op]
        # check for critical op
        critical = 0
        cur_score = 0
        for edge_op, is_in_loop in edge_ops.items():
            if edge_op in op_info.depender:
                if is_in_loop:
                    critical = 3 if not isinstance(op, Output) else 2
                    break
                else:
                    cur_score += 50000
        # If an op in the partition is blocked, because this ready op has not been executed, add scores to cur_score
        # if critical = 3, it means besides the condition above, the blocked op in the partition is an op in loop
        # special case for isinstance(op, Output): critical should not be 3, but 2, to make other ops schedule first

        loop_score = 0
        # need to run the ops in the loop as soon as possible
        if _is_fast_select_op(op):
            loop_score = 1

        if connected_to_parti:
            cur_score += 100000
        cur_score += idx * 10
        cur_score -= len(op_info.depender)
        score_tuple = (loop_score, critical, cur_score)
        if score_tuple >= cur_best:
            cur_best = score_tuple
            cur_best_op = op
    return cur_best_op
        
def _partition(f: Function, partition_thres = 3) -> List[_Partition]:
    opinfo = _collect_op_info(f)
    partitions: List[_Partition] = []
    ready_ops = [(v,0) for v in filter(lambda op: isinstance(op, GraphSourceTrait), f.ops)]
    to_visit = dict([(op, None) for op in f.ops])
    num_batch = 0
    while len(ready_ops):
        partition = _Partition(OrderedDict(), set())
        # print("============\nnew partition:", partition)
        selected = _select_next(ready_ops, opinfo, partition, f)
        while selected:
            # remove the pending dependency. If an op is ready, put into ready queue
            def maintain_ready_queue(s_op: OpBase):
                nonlocal num_batch
                num_batch += 1
                # maintain the ready queue for topology sort
                for user in f.op_to_id[s_op].uses:
                    pendingset = opinfo[user].pending_dep
                    del pendingset[s_op]
                    if len(pendingset) == 0:
                        ready_ops.append((user, num_batch))
                del to_visit[s_op]
                del_from_ready_op(ready_ops, s_op)
            # end of maintain_ready_queue()
            maintain_ready_queue(selected)
            if isinstance(selected, GraphSourceTrait):
                # don't put input in partition yet
                pass
            elif isinstance(selected, CrossSectionalOp):
                single_partition = _Partition(OrderedDict(), set())
                single_partition.add(opinfo, selected)
                # if an output op is directly connected with CrossSectionalOp, merge it in the partition
                for user in f.op_to_id[selected].uses:
                    if isinstance(user, Output):
                        maintain_ready_queue(user)
                        single_partition.add(opinfo, user)
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
                # if an output is directly connected with the partition, add it
                direct_output = None
                for candidate, bat in ready_ops:
                    if isinstance(candidate, Output):
                        if _is_bad_op_to_add(candidate, partition)[0]:
                            continue
                        direct_output = candidate
                        break
                if direct_output is not None:
                    
                    selected = direct_output
                    continue
                # too many outputs visited, make a new partition
                break
            selected = _select_next(ready_ops, opinfo, partition, f)
        if partition.ops.__len__():
            partitions.append(partition)
    if to_visit.__len__() != 0:
        raise RuntimeError("Some ops are not visited") #"Some ops not visited: "+ "\n".join([str(eop) for eop in to_visit]))
    return partitions

def _search_output_use(op: OpBase, info: OpInfo) -> OpBase:
    if isinstance(op, Output):
        return op
    for use in info.uses:
        if isinstance(use, Output):
            return use
    return None

def _print_partition_info(stages: List[GenericPartition], impl: List[Function]):
    if debug_mode < 2:
        return
    name_to_impl = dict([(f.name, f) for f in impl])
    for s in stages:
        print(f'''===========================
partition: {s.attrs["name"]}
inputs: {[f.attrs["name"] for f in s.inputs]}
Impl:
{name_to_impl[s.attrs["name"]]}''')
    if debug_mode < 3:
        return
    # loop check
    for s in stages:
        stck = []
        def recurive(x: GenericPartition):
            if x in stck and len(stck) > 1:
                stck.append(x)
                raise RuntimeError("Loop in partitions: " + str([val.attrs["name"] for val in stck]))
            stck.append(x)
            for p in x.inputs:
                recurive(p)
            stck.pop()
        recurive(s)

    

def _transform_partitions(partitions: List[_Partition], f: Function) -> Tuple[Function, List[Function]]:
    naming_table = dict()
    for op in f.ops:
        if isinstance(op, Input) or isinstance(op, Output):
            name = op.attrs["name"]
            if name == "" or name in naming_table:
                old_op = naming_table.get(name, None)
                raise RuntimeError(f"Duplicated or bad name of op: {id(op)} {op}\nold op is: {id(old_op)} {old_op}")
            naming_table[name] = op
    def add_to_naming_table(v: str) -> str:
        nonlocal naming_table
        cur_name = v
        idx = 0
        while cur_name in naming_table:
            cur_name = f"{v}_{idx}"
            idx += 1
        naming_table[cur_name] = None
        return cur_name
    # now partitions are built. We should build the dependencies between partitions
    op_lookup_table: Dict[OpBase, _Partition] = dict()
    for p in partitions:
        for op in p.ops:
            if not isinstance(op, GraphSourceTrait):
                # input is shared by all ops
                assert(op not in op_lookup_table)
                op_lookup_table[op] = p
    hash_cache: Dict['OpBase', int] = dict()
    for p in partitions:
        name_to_input = dict()
        depending : typing.OrderedDict[_Partition, None] = OrderedDict()
        # for each op in partition
        for op in list(p.ops):
            for idx, inp in enumerate(op.inputs):
                if inp not in p.ops:
                    # if the partition depends on an op of another partition
                    if inp.get_parent():
                        raise RuntimeError("Bad cross partition op: " + str(inp) + "\ncur op=" + str(op))
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
                            out_name = add_to_naming_table(inp.hash_hex(hash_cache))
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
    _print_partition_info(out_stages, out_impl)
    out_stages = Function.topo_sort_ops(out_stages)
    return Function(out_stages), out_impl

def do_partition(f: Function, factor: int, options: dict = {}) -> Tuple[Function, List[Function]]:
    partitions = _partition(f, factor)
    return _transform_partitions(partitions, f)