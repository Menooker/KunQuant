import os
import subprocess
import tempfile
from typing import List, Tuple
import KunRunner
from KunQuant.Driver import compileit as driver_compileit
from KunQuant.Stage import Function

def call_cpp_compiler(source: str, compiler: str, options: List[str], tempdir: str) -> str:
    inpath = os.path.join(tempdir, "1.cpp")
    with open(inpath, 'w') as f:
        f.write(source)
    outpath = os.path.join(tempdir, "1.so")
    cmd = [compiler] + options + [inpath, "-o", outpath]
    subprocess.check_call(cmd, shell=False)
    return outpath

def compile_cpp_and_load(source: str, tempdir: str, compiler: str, options: List[str]) -> KunRunner.Library:
    with tempfile.TemporaryDirectory(dir=compiler) as tmpdirname:
        outdir = call_cpp_compiler(source, tempdir, compiler, options, tmpdirname)
        lib = KunRunner.Library.load(libpath)
        return lib

def compileit(f: Function, module_name: str, tempdir: str = None, **kwargs) -> Tuple[KunRunner.Library, KunRunner.Module]:
    src = driver_compileit(f, module_name, **kwargs)
    lib = compile_cpp_and_load(src, tempdir, ["-O2", "-shared", "-fPIC"])
    return lib, lib.getModule(module_name)
