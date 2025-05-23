from KunQuant.Op import *
from KunQuant.Stage import *
from KunQuant.ops import *
import KunQuant.passes
from KunQuant.passes import *

def expect_output(f: Function, out: str):
    strf = str(f)
    if strf != out:
        raise RuntimeError(f"expecting {out}\nbut got\n{strf}")

def expect_str_output(strf: str, out: str):
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
    # check that identity print works (out2 does not depend on input b)
    expect_str_output(out2.to_string(0, True), '''Output@{name:ou2}(
  Sqrt@(
    DivConst@{value:9}(
      ReduceAdd@(
        Mul@(
          Sub@(
            IterValue@(
              ForeachBackWindow@{window:10}(
                WindowedTempOutput@{window:10}(
                  input(a)
                )
              ),
              WindowedTempOutput@{window:10}(
                input(a)
              )
            ),
            DivConst@{value:10}(
              ReduceAdd@(
                IterValue@(
                  ForeachBackWindow@{window:10}(
                    WindowedTempOutput@{window:10}(
                      input(a)
                    )
                  ),
                  WindowedTempOutput@{window:10}(
                    input(a)
                  )
                )
              )
            )
          ),
          Sub@(
            IterValue@(
              ForeachBackWindow@{window:10}(
                WindowedTempOutput@{window:10}(
                  input(a)
                )
              ),
              WindowedTempOutput@{window:10}(
                input(a)
              )
            ),
            DivConst@{value:10}(
              ReduceAdd@(
                IterValue@(
                  ForeachBackWindow@{window:10}(
                    WindowedTempOutput@{window:10}(
                      input(a)
                    )
                  ),
                  WindowedTempOutput@{window:10}(
                    input(a)
                  )
                )
              )
            )
          )
        )
      )
    )
  )
)''')

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


def check_duplicate_rank_in():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        a = inp1 + 1
        r = Rank(a)
        s = Scale(a)
        out1 = Output(r, "ou1")
        out2 = Output(s, "ou1")
    f = Function(builder.ops)
    decompose_rank(f)
    expect_output(f, '''v0 = Input@{name:a}()
v1 = AddConst@{value:1}(v0)
v2 = Output@{name:1302190f66243c466}(v1)
v3 = Rank@(v2)
v4 = Scale@(v2)
v5 = Output@{name:ou1}(v3)
v6 = Output@{name:ou1}(v4)''')

def check_duplicate_rank_out():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        r = Rank(inp1)
        out1 = Output(r, "ou1")
        out2 = Output(r, "ou1")
    f = Function(builder.ops)
    move_dup_rank_output(f)
    expect_output(f, '''v0 = Input@{name:a}()
v1 = Rank@(v0)
v2 = Output@{name:ou1}(v1)
v3 = Output@{name:ou1}(v2)''')

# check reduction op dependency sorted before ForeachBackWindow
def check_toposort():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp2 = Input("b")
        loopa1 = ForeachBackWindow(inp1, 10)
        builder.loop = loopa1
        vara1 = IterValue(loopa1, inp1)
        builder.loop = None
        Output(ReduceRank(vara1,inp2))
    ops = builder.ops
    ops[1],ops[2] = ops[2],ops[1]
    expect_output(Function(Function.topo_sort_ops(ops)),'''v0 = Input@{name:a}()
v1 = Input@{name:b}()
v2 = ForeachBackWindow@{window:10}(v0)
v3 = IterValue@(v2,v0) in v2
v4 = ReduceRank@(v3,v1)
v5 = Output@{name:}(v4)''')

def check_mergeLoop():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        inp2 = Input("b")
        loopa1 = ForeachBackWindow(inp1, 10)
        builder.loop = loopa1
        vara1 = IterValue(loopa1, inp1)
        builder.loop = None
        reducea1 = ReduceAdd(vara1)

        loopa2 = ForeachBackWindow(inp1, 20)
        builder.loop = loopa2
        vara2 = IterValue(loopa2, inp1)
        builder.loop = None
        reducea2 = ReduceMax(vara2)

        loopa3 = ForeachBackWindow(inp1, 20)
        builder.loop = loopa3
        vara3 = IterValue(loopa3, inp1)
        builder.loop = None
        reducea3 = ReduceMin(vara3)

        loopa4 = ForeachBackWindow(inp1, 32)
        builder.loop = loopa4
        vara4 = IterValue(loopa4, inp1)
        builder.loop = None
        reducea4 = ReduceMin(vara4)

        loopb = ForeachBackWindow(inp2, 32)
        builder.loop = loopb
        varb = IterValue(loopb, inp2)
        builder.loop = None
        reduceb = ReduceMin(varb)

        out2 = Output(reducea1 + reducea2 + reducea3 + reducea4 + reduceb, "ou1")
    f = Function(builder.ops)
    merge_loops(f)
    expect_output(f, '''v0 = Input@{name:a}()
v1 = ForeachBackWindow@{window:32,segment_end:20}(v0)
v2 = IterValue@(v1,v0) in v1
v3 = ReduceMin@(v2)
v4 = ForeachBackWindow@{window:20,copy_prev_body:True,segment_end:10}(v0)
v5 = IterValue@(v4,v0) in v4
v6 = ReduceMax@(v5)
v7 = ReduceMin@(v5)
v8 = ForeachBackWindow@{window:10,copy_prev_body:True}(v0)
v9 = IterValue@(v8,v0) in v8
v10 = ReduceAdd@(v9)
v11 = Input@{name:b}()
v12 = ForeachBackWindow@{window:32}(v11)
v13 = IterValue@(v12,v11) in v12
v14 = ReduceMin@(v13)
v15 = Add@(v10,v6)
v16 = Add@(v15,v7)
v17 = Add@(v16,v3)
v18 = Add@(v17,v14)
v19 = Output@{name:ou1}(v18)''')


def check_pow():
    builder = Builder()
    with builder:
        inp1 = Input("a")
        Output(Pow(inp1, ConstantOp(5)), "out5")
        Output(Pow(inp1, ConstantOp(5.5)), "out55")
        Output(Pow(inp1, ConstantOp(0.5)), "out05")
        Output(Pow(inp1, ConstantOp(-5)), "outm5")
        Output(Pow(inp1, ConstantOp(-5.5)), "outm55")
        Output(Pow(inp1, ConstantOp(-0.5)), "outm05")
    f = Function(builder.ops)
    decompose(f)
    expect_output(f,'''v0 = Input@{name:a}()
v1 = Mul@(v0,v0)
v2 = Mul@(v1,v1)
v3 = Mul@(v0,v2)
v4 = Output@{name:out5}(v3)
v5 = Mul@(v0,v0)
v6 = Mul@(v5,v5)
v7 = Mul@(v0,v6)
v8 = Sqrt@(v0)
v9 = Mul@(v8,v7)
v10 = Output@{name:out55}(v9)
v11 = Sqrt@(v0)
v12 = Output@{name:out05}(v11)
v13 = Mul@(v0,v0)
v14 = Mul@(v13,v13)
v15 = Mul@(v0,v14)
v16 = ConstantOp@{value:1}()
v17 = Div@(v16,v15)
v18 = Output@{name:outm5}(v17)
v19 = Mul@(v0,v0)
v20 = Mul@(v19,v19)
v21 = Mul@(v0,v20)
v22 = Sqrt@(v0)
v23 = Mul@(v22,v21)
v24 = ConstantOp@{value:1}()
v25 = Div@(v24,v23)
v26 = Output@{name:outm55}(v25)
v27 = Sqrt@(v0)
v28 = ConstantOp@{value:1}()
v29 = Div@(v28,v27)
v30 = Output@{name:outm05}(v29)''')


if __name__ == "__main__":
    check_pow()
    check_window()
    check_simple()
    check_gc()
    check_decompose()
    check_fold()
    check_tempwindow_elim()
    check_opt_sum()
    check_fold_window()
    check_div_cmp()
    check_mergeLoop()
    check_toposort()
    check_duplicate_rank_out()
    check_duplicate_rank_in()