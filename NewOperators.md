# Adding Custom Operators

Currently, KunQuant includes dozens of operators, ranging from basic arithmetic (add, subtract, multiply, divide) to statistical functions and linear regression. If you need to use a custom operator, you’ll need to modify KunQuant and add your own implementation. This document introduces the basics of how to do so.

Refer to the documentation here before proceeding: [Operators.md](./Operators.md)

Here's the English translation of your document in **Markdown** format:

## Operators Overview

KunQuant is a lightweight compiler that converts user expressions into **computation graphs**, specifically **Directed Acyclic Graphs (DAGs)**. Each node in the graph is an **operator**, which accepts 0 or more inputs and produces 0 or 1 outputs.

For example, the `Add` operator takes 2 inputs and produces 1 output.

From an implementation perspective, operators can be classified into:

1. **Elementwise**: Output depends only on inputs from the *same stock at the same time*. Most simple ops (add, subtract, pow, log) are of this type.

2. **Rolling Windowed**: Output depends on a *sliding window* of past inputs, e.g. `WindowedAvg`, `WindowedStddev`.

3. **Fully Dependent**: Output depends on *all previous data points* for the same stock. Some ops are technically not "fully dependent" but are grouped here for implementation ease.

4. **Cross-Sectional**: Output at a time `T` depends on *all stocks' values* at `T`.

---

## Elementwise Operators

If the operator you want to implement can be composed from existing operators, you can inherit from `CompositiveOp`.
For example, the `Clip` operator limits values to ±eps:

```python
class Clip(CompositiveOp):
    '''
    Elementwisely clip the input value `v` with a given positive constant `eps`:
    Clip(v) = max(min(v, eps), -eps)
    The output will be between [-eps, +eps]
    '''
    def __init__(self, v: OpBase, eps: float) -> None:
        inputs = [v]
        super().__init__(inputs, [("value", eps)])

    def decompose(self) -> List[OpBase]:
        eps = self.attrs["value"]
        inp = self.inputs[0]
        b = Builder(self.get_parent())
        with b:
            v0 = Min(inp, ConstantOp(eps))
            out = Max(v0, ConstantOp(-eps))
        return b.ops
```

In KunQuant, compile-time constants (like `eps`) are stored in the operator’s `attrs`. Compositive operators just need to implement a `decompose()` method to return a list of operators that replace the original one. KunQuant processes this type of operator by calling its `decompose` method, which breaks the operator down into a sequence of other operators. The original operator's output is represented by the **last operator** in the list returned by `decompose()`.

---

### Native Elementwise Operator

If an operator **cannot** be expressed via existing ones, you need to implement it natively and write C++ code. For example, `Add`:

Python definition in `KunQuant/ops/ElewiseOp.py`:

```python
class Add(BinaryElementwiseOp):
    pass
```

This operator inherits from `BinaryElementwiseOp`, indicating that it is an elementwise operator with two inputs.
If you want to implement a single-input operator like `exp`, you can inherit from `UnaryElementwiseOp`.


The actual implementation is in C++:
[cpp/Kun/Ops.hpp](https://github.com/Menooker/KunQuant/blob/main/cpp/Kun/Ops.hpp).
This C++ header file implements a function named `Add` with the same name. Note that it must be written as a template.
The function takes SIMD vectors as input, representing **N packed floats or doubles**, where each value corresponds to **a different stock at the same time**.
The SIMD width `N` and the data type are determined by KunQuant’s `compileit` parameters.


```cpp
template <typename T1, typename T2>
inline auto Add(T1 a, T2 b) -> decltype(kun_simd::operator+(a, b)) {
    return kun_simd::operator+(a, b);
}
```

KunQuant wraps SIMD intrinsics (AVX, AVX512) under the `kun_simd` namespace:
[cpp/KunSIMD/cpu](https://github.com/Menooker/KunQuant/tree/main/cpp/KunSIMD/cpu)

---

## Rolling Windowed Operators

Example: `WindowedStddev`
[KunQuant/ops/CompOp.py](https://github.com/Menooker/KunQuant/blob/main/KunQuant/ops/CompOp.py). The code has been slightly simplified for easier understanding.

```python
class WindowedStddev(WindowedCompositiveOp):
    '''
    Unbiased standard deviation of a rolling look back window, including the current newest data.
    For indices < window-1, the output will be NaN
    similar to pandas.DataFrame.rolling(n).std()
    '''
    def decompose(self) -> List[OpBase]:
        window = self.attrs["window"]
        b = Builder(self.get_parent())
        with b:
            v0 = self.inputs[0]
            avg = WindowedAvg(v0, window)
            each = ForeachBackWindow(v0, window)
            b.set_loop(each)
            diff = Sub(IterValue(each, v0), avg)
            sqr = Mul(diff, diff)
            b.set_loop(self.get_parent())
            vsum = ReduceAdd(sqr)
            out = Sqrt(vsum/(window - 1))
        return b.ops
```

Similar to the `Clip` operator mentioned earlier, this operator inherits from the `WindowedCompositiveOp` class.
`WindowedCompositiveOp` itself inherits from `CompositiveOp`, and provides an `__init__` method that requires the user to specify the **rolling window size** when creating the operator, storing it in `self.attrs['window']`.

We use the following algorithm (pseudocode) to compute the standard deviation at a given time point:

```python
def stddev(data, time, window):
    avg = WindowedAvg(data, time, window)
    sum = 0
    for i in range(window):
        sum += (data[time - i] - avg) ** 2
    return sqrt(sum) / (window - 1)
```

Now let’s look at the `decompose` method. First, compute the average over the current sliding window:

```python
avg = WindowedAvg(v0, window)
```

Then iterate over the input using a sliding window:

```python
each = ForeachBackWindow(v0, window)
b.set_loop(each)
```

This corresponds to the `for` loop in the pseudocode above. The `set_loop` line means the operators generated afterward will be placed **inside** the loop.

Next, compute the squared difference between the current window value and the average:

```python
diff = Sub(IterValue(each, v0), avg)
sqr = Mul(diff, diff)
```

Here, `IterValue(each, v0)` corresponds to `data[time - i]` in the pseudocode — it fetches the value at the current position in the sliding window.
In contrast, using `v0` directly would access the value at the current time only.

Then we sum up the squared differences:

```python
b.set_loop(self.get_parent())
vsum = ReduceAdd(sqr)
```

The `ReduceAdd` operator aggregates all the `sqr` values generated inside the loop.
**Important**: All reduction operators (`Reduce*`) must be placed **outside** of the `Foreach` loop. The `set_loop(self.get_parent())` line ensures this.
The order of `set_loop` and `ReduceAdd` **must be reversed**.

Finally, compute the result:

```python
out = Sqrt(vsum / (window - 1))
```

This file also contains many other, more complex **rolling windowed operator implementations** — for example, `corr`, which requires **iterating over two inputs simultaneously**. These implementations can serve as useful references when creating your own custom operators.

In the `stddev` example above, we used KunQuant's built-in **reduction operator** `ReduceAdd`. The project also provides a variety of other reduction operators designed for sliding window computations, such as `ReduceMin`, `ReduceArgMin`, and more.


---

## Reduction Operators

You can define your own reduction operators. Example: `ReduceMin`
[KunQuant/ops/ReduceOp.py](https://github.com/Menooker/KunQuant/blob/main/KunQuant/ops/ReduceOp.py)

```python
class ReduceMin(ReductionOp):
    pass
```

C++ implementation:
[cpp/Kun/Ops.hpp](https://github.com/Menooker/KunQuant/blob/main/cpp/Kun/Ops.hpp)

```cpp
template <typename T, int stride>
struct ReduceMin {
    using simd_t = kun_simd::vec<T, stride>;
    simd_t v = std::numeric_limits<T>::infinity();
    void step(simd_t input, size_t index) {
        v = sc_select(sc_isnan(v, input), NAN, sc_min(v, input));
    }
    operator simd_t() { return v; }
};
```

The reason for using a `struct` here instead of a simple C++ function is that a reduction operator needs to **maintain an internal "state"**, and the reduction process revolves around updating this state.

For example, the state of `ReduceMin` is the **current minimum value within the window**. The template parameter `T` represents the data type (`float` or `double`), and `stride` represents the SIMD width — i.e., how many stocks are processed simultaneously at one time.

A reduction operator must implement two functions, one of which is `step()`.
The `step` function is called for each value in the sliding window, updating the internal state. The window values are passed **sequentially from oldest to newest** to the operator via `step()`.
The parameter `input` contains `N` stock values at the current time (packed as SIMD vector), and `index` is the position in the window (from 0 to `window-1`).

The reduction operator returns its result via the conversion operator `operator simd_t()`. For `ReduceMin`, this simply returns the computed minimum value `v`.

The minimum value state is initialized as:

```cpp
simd_t v = std::numeric_limits<T>::infinity();
```

Here, `std::numeric_limits<T>::infinity()` is a scalar floating-point infinity, which the `simd_t` constructor broadcasts to all lanes of the SIMD vector.

Inside the `step` function, the state `v` is updated as follows:

```cpp
v = sc_select(sc_isnan(v, input), NAN, sc_min(v, input));
```

* `sc_isnan(v, input)` checks if any element in the vectors `v` or `input` is NaN, returning a SIMD mask of booleans.
* `sc_select` acts like a ternary operator (`?:`) but lane-wise: for each lane, if the mask bit is true, it returns the second argument (`NAN`); otherwise, it returns the third (`sc_min(v, input)`).

Thus, this updates `v` to hold the minimum value, ignoring NaNs.

---

The `step` function is called for every value in the sliding window. The **state object is reset for each timestamp `T`**: when computing the factor value at time `T`, a new state is created, then values from `T - window + 1` to `T` are fed into `step()`. After completing the window, the reduction result is output for time `T`. Then the state is reset for the next time step `T + 1`.

The generated C++ pseudocode roughly looks like this:

```cpp
void compute(...) {
    for (int time = 0; time < end; ++time) {
        // other fused operators
        ...

        ReduceMin r;  // create reduction state

        for (int back = 0; back < window; ++back) {
            auto d = get_data(time - back);
            r.step(d, back);  // update state with each window value
        }

        auto other = op(r); // other fused operators
        ...
    }
}
```

---

## Fully Dependent Operators

After understanding the reduction operators discussed above, we can now explain **fully dependent operators**. For reduction operators, the **state object is reset at each time point** (see the pseudocode above).

However, some special reduction operators require the state to **persist across different time points** (i.e., the state object exists **outside** the `for (time)` loop). These are called **fully dependent operators**.

An example of such an operator is the **Exponential Moving Average (ExpMovingAvg)**, source at [here](https://github.com/Menooker/KunQuant/blob/generic_cpp_codegen/KunQuant/ops/MiscOp.py)


```python
class ExpMovingAvg(OpBase, GloablStatefulOpTrait):
    '''
    Exponential Moving Average (EMA)
    Similar to pd.DataFrame.ewm(span=window, adjust=False, ignore_na=True).mean()
    '''
    def __init__(self, v: OpBase, window: int) -> None:
        super().__init__([v], [("window", window)])

    def get_state_variable_name_prefix(self) -> str:
        return "ema_"
    
    def generate_step_code(self, idx: str, time_idx: str, inputs: List[str]) -> str:
        return f"auto v{idx} = ema_{idx}.step({inputs[0]}, {time_idx});"
```

The key point is to extend `GloablStatefulOpTrait` class. Other than `__init__` contructor to provide, `ExpMovingAvg` also provides `get_state_variable_name_prefix` to return the **state object variable** name prefix. The state object variable will defined outside of the loop for **time**. The function `generate_step_code` should return the C++ source code string which computes the operator output value for each timepoint.

The generated C++ pseudocode looks roughly like this:

```cpp
void compute(...) {
    ExpMovingAvg ema;  // state persists across time points
    for (int time = 0; time < end; ++time) {
        // other fused operators
        auto value = op(...);
        auto ema_result = ema.step(value, time);  // compute EMA update
        auto other = op(r);  // other fused operators
        ...
    }
}
```

---

## Cross-Sectional Operators

Take the cross-sectional ranking operator `Rank` as an example.
It inherits from the `CrossSectionalOp` class.

In C++, this operator is defined here:
[cpp/Kun/Rank.hpp](https://github.com/Menooker/KunQuant/blob/main/cpp/Kun/Rank.hpp)

Additionally, KunQuant provides an interface to develop cross-sectional operators **purely in Python** (though C++ code is still required).
You can write a Python class that inherits from `GenericCrossSectionalOp`, which will generate C++ code to implement the cross-sectional operator.

For example, `DiffWithWeightedSum` in:
[KunQuant/ops/MiscOp.py](https://github.com/Menooker/KunQuant/blob/main/KunQuant/ops/MiscOp.py)
is implemented this way.

Developers need to provide two functions, `generate_head` and `generate_body`, which return C++ source code strings.

KunQuant injects the returned code snippets into the following template (simplified here for clarity):

```cpp
void compute(...) {
    using T = float;  // or double
    size_t num_stocks = get_numstocks();
    // {generate_head()} code is inserted here
    for (int time = ...; time < ...; ++time) {
        T* input_0 = ...;   // operator's 0th input
        // T* input_1 = ...
        T* output_0 = ...;  // operator's 0th output
        // {generate_body()} code is inserted here
    }
}
```
