import os


def _safe_cast(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


debug_mode = _safe_cast(os.environ.get("KUN_DEBUG", "0"))
jit_debug_mode = _safe_cast(os.environ.get("KUN_DEBUG_JIT", ""))


def kun_pass(p):
    def inner(f, opt={}):
        if debug_mode > 0:
            print("Running pass", p, "num original ops:", len(f.ops))
        p(f, opt)
        if debug_mode > 1:
            print("After the pass: ", f)
        if debug_mode > 0:
            print("new num ops:", len(f.ops))
    return inner
