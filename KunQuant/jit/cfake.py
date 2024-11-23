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

def compile_cpp_and_load(source: str, tempdir: str, compiler: str, options: List[str]) -> KunRunner.Library:
    with tempfile.TemporaryDirectory(dir=tempdir) as tmpdirname:
        outdir = call_cpp_compiler(source, compiler, options, tmpdirname)
        lib = KunRunner.Library.load(outdir)
        return lib

def compileit(f: Function, module_name: str, compiler: str = "g++", tempdir: str = None, **kwargs) -> Tuple[KunRunner.Library, KunRunner.Module]:
    timeit.timeit()
    lib = None
    src = None
    def kuncompile():
        nonlocal src
        src = driver_compileit(f, module_name, **kwargs)
    def dowork():
        nonlocal lib
        lib = compile_cpp_and_load(src, tempdir, compiler, ["-std=c++11", "-O2", "-shared", "-fPIC", "-march=native"] + [f"-I{v}" for v in _include_path])
    if Util.jit_debug_mode:
        print("[KUN_JIT] Source generation takes ", timeit.timeit(kuncompile, number=1), "s")
        print("[KUN_JIT] C++ compiler takes ", timeit.timeit(dowork, number=1), "s")
    else:
        kuncompile()
        dowork()
    return lib, lib.getModule(module_name)
