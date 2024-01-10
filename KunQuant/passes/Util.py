import os

debug_mode = os.environ.get("KUN_DEBUG", "0") != "0"

def kun_pass(p):
    def inner(f, opt = {}):
        if debug_mode:
            print("Running pass", p, "num original ops:", len(f.ops))
        p(f, opt)
        if debug_mode:
            print("After the pass: ", f)
            print("new num ops:", len(f.ops))
    return inner