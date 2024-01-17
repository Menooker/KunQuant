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
v3 = IterValue@(v2,v1) in v2
v4 = ReduceAdd@(v3)
v5 = DivConst@{value:10}(v4)
v6 = WindowedTempOutput@{window:10}(v0)
v7 = ForeachBackWindow@{window:10}(v6)
v8 = IterValue@(v7,v6) in v7
v9 = ReduceAdd@(v8)
v10 = DivConst@{value:10}(v9)
v11 = WindowedTempOutput@{window:10}(v0)
v12 = ForeachBackWindow@{window:10}(v11)
v13 = IterValue@(v12,v11) in v12
v14 = Sub@(v13,v10) in v12
v15 = Mul@(v14,v14) in v12
v16 = ReduceAdd@(v15)
v17 = DivConst@{value:9}(v16)
v18 = Sqrt@(v17)
v19 = Output@{name:ou1}(v5)
v20 = Output@{name:ou2}(v18)''')

def check_fold():
    f = build_avg_and_stddev()
    decompose(f)
    expr_fold(f)
    expected = '''v0 = Input@{name:a}()
v1 = WindowedTempOutput@{window:10}(v0)
v2 = ForeachBackWindow@{window:10}(v1)
v3 = IterValue@(v2,v1) in v2
v4 = ReduceAdd@(v3)
v5 = DivConst@{value:10}(v4)
v6 = ForeachBackWindow@{window:10}(v1)
v7 = IterValue@(v6,v1) in v6
v8 = Sub@(v7,v5) in v6
v9 = Mul@(v8,v8) in v6
v10 = ReduceAdd@(v9)
v11 = DivConst@{value:9}(v10)
v12 = Sqrt@(v11)
v13 = Output@{name:ou1}(v5)
v14 = Output@{name:ou2}(v12)'''
    expect_output(f, expected)
    special_optimize(f)
    expected2 = '''v0 = Input@{name:a}()
v1 = WindowedTempOutput@{window:11}(v0)
v2 = FastWindowedSum@{window:10}(v1)
v3 = DivConst@{value:10}(v2)
v4 = ForeachBackWindow@{window:10}(v1)
v5 = IterValue@(v4,v1) in v4
v6 = Sub@(v5,v3) in v4
v7 = Mul@(v6,v6) in v4
v8 = ReduceAdd@(v7)
v9 = DivConst@{value:9}(v8)
v10 = Sqrt@(v9)
v11 = Output@{name:ou1}(v3)
v12 = Output@{name:ou2}(v10)'''
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
v1 = WindowedTempOutput@{window:11}(v0)
v2 = FastWindowedSum@{window:10}(v1)
v3 = AddConst@{value:10}(v2)
v4 = WindowedTempOutput@{window:11}(v3)
v5 = FastWindowedSum@{window:10}(v4)
v6 = Output@{name:}(v3)
v7 = Output@{name:}(v5)''')

# check print identity
def check_fold_window():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp2 = Input("b")
        v1 = WindowedCorrelation(inp1, 10, inp2)
        v2 = WindowedStddev(inp1, 10)
        out1 = Output(v1, "ou1")
        out2 = Output(v2, "ou2")
    f = Function(builder.ops)
    decompose(f)
    expr_fold(f)
    expect_output(f, '''v0 = Input@{name:a}()
v1 = Input@{name:b}()
v2 = WindowedTempOutput@{window:10}(v0)
v3 = ForeachBackWindow@{window:10}(v2)
v4 = IterValue@(v3,v2) in v3
v5 = ReduceAdd@(v4)
v6 = DivConst@{value:10}(v5)
v7 = WindowedTempOutput@{window:10}(v1)
v8 = ForeachBackWindow@{window:10}(v7)
v9 = IterValue@(v8,v7) in v8
v10 = ReduceAdd@(v9)
v11 = DivConst@{value:10}(v10)
v12 = ForeachBackWindow@{window:10}(v2,v7)
v13 = IterValue@(v12,v2) in v12
v14 = Sub@(v13,v6) in v12
v15 = IterValue@(v12,v7) in v12
v16 = Sub@(v15,v11) in v12
v17 = Mul@(v14,v14) in v12
v18 = Mul@(v16,v16) in v12
v19 = Mul@(v14,v16) in v12
v20 = ReduceAdd@(v19)
v21 = ReduceAdd@(v17)
v22 = Sqrt@(v21)
v23 = ReduceAdd@(v18)
v24 = Sqrt@(v23)
v25 = Mul@(v22,v24)
v26 = Div@(v20,v25)
v27 = DivConst@{value:9}(v21)
v28 = Sqrt@(v27)
v29 = Output@{name:ou1}(v26)
v30 = Output@{name:ou2}(v28)''')

def check_div_cmp():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp2 = Input("b")
        out1 = inp1/inp2 > 10
        out1 = Output(out1, "ou1")
        out2 = ConstantOp(1) < inp1/inp2
        out2 = Output(out2, "ou1")
    f = Function(builder.ops)
    special_optimize(f)
    expect_output(f, '''v0 = Input@{name:a}()
v1 = Input@{name:b}()
v2 = ConstantOp@{value:10}()
v3 = Mul@(v1,v2)
v4 = GreaterThan@(v0,v3)
v5 = Output@{name:ou1}(v4)
v6 = LessThan@(v1,v0)
v7 = Output@{name:ou1}(v6)''')

if __name__ == "__main__":
    check_window()
    check_simple()
    check_gc()
    check_decompose()
    check_fold()
    check_tempwindow_elim()
    check_opt_sum()
    check_fold_window()
    check_div_cmp()