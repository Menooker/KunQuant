import os
import subprocess
import tempfile
from typing import List, Tuple
import KunRunner
from KunQuant.Driver import compileit as driver_compileit
from KunQuant.Stage import Function
from KunQuant.passes import Util
import timeit

_cpp_root = os.path.join(os.path.dirname(__file__), "..", "..", "cpp")
_include_path = [_cpp_root]

def call_cpp_compiler(source: str, compiler: str, options: List[str], tempdir: str) -> str:
    inpath = os.path.join(tempdir, "1.cpp")
    with open(inpath, 'w') as f:
        f.write(source)
    outpath = os.path.join(tempdir, "1.so")
    if Util.jit_debug_mode:
        print("[KUN_JIT] temp jit files:", inpath, outpath)
    cmd = [compiler] + options + [inpath, "-o", outpath]
    subprocess.check_call(cmd, shell=False)
    return outpath

class _fake_temp:
    def __init__(self, dir) -> None:
        self.dir = dir

    def __enter__(self):
        return self.dir

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass

def compile_cpp_and_load(source: str, tempdir: str, compiler: str, options: List[str], keep_files: bool) -> KunRunner.Library:
    tempclass = _fake_temp if keep_files else tempfile.TemporaryDirectory 
    with tempclass(dir=tempdir) as tmpdirname:
        outdir = call_cpp_compiler(source, compiler, options, tmpdirname)
        lib = KunRunner.Library.load(outdir)
        return lib

def compileit(f: Function, module_name: str, compiler: str = "g++", tempdir: str = None, keep_files: bool = False, **kwargs) -> Tuple[KunRunner.Library, KunRunner.Module]:
    lib = None
    src = None
    if keep_files and not tempdir:
        raise RuntimeError("if keep_files=True, tempdir should not be empty")
    def kuncompile():
        nonlocal src
        src = driver_compileit(f, module_name, **kwargs)
    def dowork():
        nonlocal lib
        lib = compile_cpp_and_load(src, tempdir, compiler, ["-std=c++11", "-O2", "-shared", "-fPIC", "-march=native"] + [f"-I{v}" for v in _include_path], keep_files)
    if Util.jit_debug_mode:
        print("[KUN_JIT] Source generation takes ", timeit.timeit(kuncompile, number=1), "s")
        print("[KUN_JIT] C++ compiler takes ", timeit.timeit(dowork, number=1), "s")
    else:
        kuncompile()
        dowork()
    return lib, lib.getModule(module_name)
