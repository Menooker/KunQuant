from KunQuant.Op import *
from KunQuant.Stage import *
from KunQuant.ops import *
import KunQuant.passes
from KunQuant.passes import *

def expect_output(f: Function, out: str):
    strf = str(f)
    if strf != out:
        raise RuntimeError(f"expecting {out}\nbut got\n{strf}")

def check_simple():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp2 = Input("b")
        v1 = Mul(inp1, inp2)
        v2 = AddConst(v1, 10)
        out = Output(v2)
    f = Function(builder.ops)
    expect_output(f, '''v0 = Input@{name:a}()
v1 = Input@{name:b}()
v2 = Mul@(v0,v1)
v3 = AddConst@{value:10}(v2)
v4 = Output@{name:}(v3)''')

def build_avg_and_stddev():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        v1 = WindowedAvg(inp1, 10)
        v2 = WindowedStddev(inp1, 10)
        out1 = Output(v1, "ou1")
        out2 = Output(v2, "ou2")
    return Function(builder.ops)

def check_decompose():
    f = build_avg_and_stddev()
    decompose(f)
    expect_output(f, '''v0 = Input@{name:a}()
v1 = WindowedTempOutput@{window:10}(v0)
v2 = ForeachBackWindow@{window:10}(v1)
v3 = ReduceAdd@(v2)
v4 = DivConst@{value:10}(v3)
v5 = WindowedTempOutput@{window:10}(v0)
v6 = ForeachBackWindow@{window:10}(v5)
v7 = ReduceAdd@(v6)
v8 = DivConst@{value:10}(v7)
v9 = WindowedTempOutput@{window:10}(v0)
v10 = ForeachBackWindow@{window:10}(v9)
v11 = Sub@(v10,v8) in v10
v12 = Mul@(v11,v11) in v10
v13 = ReduceAdd@(v12)
v14 = DivConst@{value:9}(v13)
v15 = Sqrt@(v14)
v16 = Output@{name:ou1}(v4)
v17 = Output@{name:ou2}(v15)''')

def check_fold():
    f = build_avg_and_stddev()
    decompose(f)
    expr_fold(f)
    expected = '''v0 = Input@{name:a}()
v1 = WindowedTempOutput@{window:10}(v0)
v2 = ForeachBackWindow@{window:10}(v1)
v3 = ReduceAdd@(v2)
v4 = DivConst@{value:10}(v3)
v5 = ForeachBackWindow@{window:10}(v1)
v6 = Sub@(v5,v4) in v5
v7 = Mul@(v6,v6) in v5
v8 = ReduceAdd@(v7)
v9 = DivConst@{value:9}(v8)
v10 = Sqrt@(v9)
v11 = Output@{name:ou1}(v4)
v12 = Output@{name:ou2}(v10)'''
    expect_output(f, expected)
    special_optimize(f)
    expected2 = '''v0 = Input@{name:a}()
v1 = WindowedTempOutput@{window:10}(v0)
v2 = FastWindowedSum@{window:10}(v1)
v3 = DivConst@{value:10}(v2)
v4 = ForeachBackWindow@{window:10}(v1)
v5 = Sub@(v4,v3) in v4
v6 = Mul@(v5,v5) in v4
v7 = ReduceAdd@(v6)
v8 = DivConst@{value:9}(v7)
v9 = Sqrt@(v8)
v10 = Output@{name:ou1}(v3)
v11 = Output@{name:ou2}(v9)'''
    expect_output(f, expected2)

def check_gc():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp2 = Input("b")
        v1 = Mul(inp1, inp2)
        unused1 = Mul(inp1, inp2)
        unused2 = Mul(unused1, unused1)
        v2 = AddConst(v1, 10)
        out = Output(v2)
    f = Function(builder.ops)
    expect_output(f, '''v0 = Input@{name:a}()
v1 = Input@{name:b}()
v2 = Mul@(v0,v1)
v3 = AddConst@{value:10}(v2)
v4 = Output@{name:}(v3)''')

def check_tempwindow_elim():
    # case 1, temp window on input
    builder = Builder()
    with builder:
        inp1 = Input("a")
        v1 = ReduceAdd(ForeachBackWindow(WindowedTempOutput(inp1, 10),10))
        out = Output(v1)
    f = Function(builder.ops)
    temp_window_elim(f)
    expect_output(f, '''v0 = Input@{name:a}()
v1 = ForeachBackWindow@{window:10}(v0)
v2 = ReduceAdd@(v1)
v3 = Output@{name:}(v2)''')
                  
    # case 2, temp window on output
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp1 = Mul(inp1, inp1)
        v1 = ReduceAdd(ForeachBackWindow(WindowedTempOutput(inp1, 10),10))
        out = Output(inp1)
        out2 = Output(v1)
    f = Function(builder.ops)
    temp_window_elim(f)
    expect_output(f, '''v0 = Input@{name:a}()
v1 = Mul@(v0,v0)
v2 = Output@{name:}(v1)
v3 = ForeachBackWindow@{window:10}(v2)
v4 = ReduceAdd@(v3)
v5 = Output@{name:}(v4)''')
                  
    # case 3, temp window on other tempout
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp1 = Mul(inp1, inp1)
        v1 = ReduceAdd(ForeachBackWindow(WindowedTempOutput(inp1, 10),10))
        v2 = ReduceAdd(ForeachBackWindow(WindowedTempOutput(inp1, 15),10))
        out = Output(v1)
        out2 = Output(v2)
    f = Function(builder.ops)
    temp_window_elim(f)
    expect_output(f, '''v0 = Input@{name:a}()
v1 = Mul@(v0,v0)
v2 = WindowedTempOutput@{window:15}(v1)
v3 = ForeachBackWindow@{window:10}(v2)
v4 = ReduceAdd@(v3)
v5 = ForeachBackWindow@{window:10}(v2)
v6 = ReduceAdd@(v5)
v7 = Output@{name:}(v4)
v8 = Output@{name:}(v6)''')        

def check_window():
    # case 1, temp window on input
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp1 = Mul(inp1, inp1)
        v1 = WindowedSum(inp1, 10)
        out = Output(v1)
    ok = False
    try:
        f = Function(builder.ops, True)
    except:
        ok = True
    assert(ok)

def check_opt_sum():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        v1 = WindowedSum(inp1, 10)
        v2 = AddConst(v1, 10)
        v3 = WindowedSum(v2, 10)
        out = Output(v2)
        out = Output(v3)
    f = Function(builder.ops)
    decompose(f)
    special_optimize(f)
    expect_output(f, '''v0 = Input@{name:a}()
v1 = WindowedTempOutput@{window:10}(v0)
v2 = FastWindowedSum@{window:10}(v1)
v3 = AddConst@{value:10}(v2)
v4 = WindowedTempOutput@{window:10}(v3)
v5 = FastWindowedSum@{window:10}(v4)
v6 = Output@{name:}(v3)
v7 = Output@{name:}(v5)''')

if __name__ == "__main__":
    check_window()
    check_simple()
    check_gc()
    check_decompose()
    check_fold()
    check_tempwindow_elim()
    check_opt_sum()