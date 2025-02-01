from KunQuant.Op import *
from KunQuant.Stage import *
from KunQuant.ops import *
import KunQuant.passes
from KunQuant.passes import *

def optimize(f: Function):
    decompose(f)
    expr_fold(f)
    temp_window_elim(f)

def expect_output(f: Function, out: str):
    strf = str(f)
    if strf != out:
        print(f"expecting {out}\nbut got\n{strf}")
        raise RuntimeError()

def print_all(f, funcs):
    print(f)
    print("number funcs:", len(funcs))
    for fn in funcs:
        print(fn)

def check_partition(f: Function, expected_graph: str, expected_funcs: list):
    f, impl = do_partition(f, 2)
    #print_all(f, impl)
    expect_output(f, expected_graph)
    if len(impl) != len(expected_funcs):
        raise RuntimeError("Unmatched number of funcs")
    for implf, expt in zip(impl, expected_funcs):
        expect_output(implf, expt)

def test_partition1():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp2 = Input("b")
        v1 = Mul(inp1, inp2)
        v2 = AddConst(v1, 10)
        out = Output(v2, "out1")
    f = Function(builder.ops)
    optimize(f)
    exp1 = "v0 = GenericPartition@{name:out1}()"
    exp2 = ['''name = out1
v0 = Input@{name:a}()
v1 = Input@{name:b}()
v2 = Mul@(v0,v1)
v3 = AddConst@{value:10}(v2)
v4 = Output@{name:out1}(v3)''']
    check_partition(f, exp1, exp2)

def test_partition_cylic():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp2 = Input("b")
        v1 = Mul(inp1, inp2)
        v2 = Rank(v1)
        v3 = Mul(v2, v1)
        out1 = Output(AddConst(v2, 1), "out1")
        out2 = Output(v3, "out2")
    f = Function(builder.ops)
    optimize(f)
    exp1 = '''v0 = GenericPartition@{name:18588255c1407f1d3}()
v1 = GenericPartition@{name:14dad88c7b817c3dc}(v0)
v2 = GenericPartition@{name:out1_out2}(v1,v0)'''
    exp2 = [
'''name = 14dad88c7b817c3dc
v0 = Input@{name:18588255c1407f1d3}()
v1 = Rank@(v0)
v2 = Output@{name:14dad88c7b817c3dc}(v1)''',
'''name = 18588255c1407f1d3
v0 = Input@{name:a}()
v1 = Input@{name:b}()
v2 = Mul@(v0,v1)
v3 = Output@{name:18588255c1407f1d3}(v2)''',
'''name = out1_out2
v0 = Input@{name:14dad88c7b817c3dc}()
v1 = AddConst@{value:1}(v0)
v2 = Output@{name:out1}(v1)
v3 = Input@{name:18588255c1407f1d3}()
v4 = Mul@(v0,v3)
v5 = Output@{name:out2}(v4)''']
    check_partition(f, exp1, exp2)

def test_partition_rank_out():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp2 = Input("b")
        v1 = Mul(inp1, inp2)
        v2 = Rank(v1)
        v3 = Mul(v2, v1)
        out1 = Output(v2, "out1")
        out2 = Output(v3, "out2")
    f = Function(builder.ops)
    optimize(f)
    exp1 = '''v0 = GenericPartition@{name:18588255c1407f1d3}()
v1 = GenericPartition@{name:out1}(v0)
v2 = GenericPartition@{name:out2}(v1,v0)'''
    exp2 = [
'''name = out1
v0 = Input@{name:18588255c1407f1d3}()
v1 = Rank@(v0)
v2 = Output@{name:out1}(v1)''',
'''name = 18588255c1407f1d3
v0 = Input@{name:a}()
v1 = Input@{name:b}()
v2 = Mul@(v0,v1)
v3 = Output@{name:18588255c1407f1d3}(v2)''',
'''name = out2
v0 = Input@{name:out1}()
v1 = Input@{name:18588255c1407f1d3}()
v2 = Mul@(v0,v1)
v3 = Output@{name:out2}(v2)''']
    check_partition(f, exp1, exp2)

test_partition1()
test_partition_cylic()
test_partition_rank_out()