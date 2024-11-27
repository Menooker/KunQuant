import os
from platform import machine
import subprocess
import tempfile
from typing import List, Tuple, Union
from collections.abc import Callable
import KunQuant.runner.KunRunner as KunRunner
from KunQuant.Driver import compileit as driver_compileit
from KunQuant.Stage import Function
from KunQuant.passes import Util
import timeit
import dataclasses
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

_cpp_root = os.path.join(os.path.dirname(__file__), "..", "..", "cpp")
_include_path = [_cpp_root]


FunctionList = List[Tuple[str, str]]
CallableOnFunction = Callable[[Tuple[str, str]], List[str]]

def single_thread_compile(lst: FunctionList, func: CallableOnFunction) -> List[str]:
    return [func(f) for f in lst]

_pool: ThreadPoolExecutor = None
def multi_thread_compile(lst: FunctionList, func: CallableOnFunction) -> List[str]:
    global _pool
    if not _pool:
        _pool = ThreadPoolExecutor()
    fut = [_pool.submit(func, l) for l in lst]
    return [f.result() for f in fut]


@dataclass
class X64CPUFlags:
    avx512: bool = False
    avx512dq: bool = False
    avx512vl: bool = False

class NativeCPUFlags:
    pass

@dataclass
class CppCompilerConfig:
    opt_level: int = 3
    machine: Union[NativeCPUFlags, X64CPUFlags] = NativeCPUFlags()
    for_each: Callable[[FunctionList, CallableOnFunction], List[str]] = multi_thread_compile
    other_flags : Tuple[str] = ()
    compiler: str = "g++"
    obj_ext: str = "o"
    dll_ext: str = "so"

    def build_machine_flags(self) -> List[str]:
        if isinstance(self.machine, NativeCPUFlags):
            return ["-march=native"]
        else:
            ret = ["-favx2", "-mfma"]
            if self.machine.avx512:
                ret.append("-mavx512f")
            if self.machine.avx512dq:
                ret.append("-mavx512dq")
            if self.machine.avx512vl:
                ret.append("-mavx512vl")
            return ret

@dataclass
class KunCompilerConfig:
    partition_factor:int = 3
    dtype:str = "float"
    blocking_len: int = None
    input_layout:str = "STs"
    output_layout:str = "STs"
    allow_unaligned: Union[bool, None] = None
    options = dict()

def call_cpp_compiler(path: List[str], module_name: str, compiler: str, options: List[str], tempdir: str, outext: str) -> str:
    outpath = os.path.join(tempdir, f"{module_name}.{outext}")
    if Util.jit_debug_mode:
        print("[KUN_JIT] temp jit files:", path, outpath)
    cmd = [compiler] + options + path + ["-o", outpath]
    subprocess.check_call(cmd, shell=False)
    return outpath

def call_cpp_compiler_src(source: str, module_name: str, compiler: str, options: List[str], tempdir: str, outext: str) -> str:
    inpath = os.path.join(tempdir, f"{module_name}.cpp")
    with open(inpath, 'w') as f:
        f.write(source)
    return call_cpp_compiler([inpath], module_name, compiler, options, tempdir, outext)

class _fake_temp:
    def __init__(self, dir) -> None:
        self.dir = dir

    def __enter__(self):
        return self.dir

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass

def compileit(func: Tuple[str, Function, KunCompilerConfig], libname: str, compiler_config: CppCompilerConfig, tempdir: str = None, keep_files: bool = False) -> List[KunRunner.Library]:
    lib = None
    src: List[Tuple[str, str]] = []
    if keep_files and not tempdir:
        raise RuntimeError("if keep_files=True, tempdir should not be empty")
    def kuncompile():
        for name, f, cfg in func:
            src.append((name, driver_compileit(f, name, **dataclasses.asdict(cfg))))
    def dowork():
        nonlocal lib      
        tempclass = _fake_temp if keep_files else tempfile.TemporaryDirectory 
        with tempclass(dir=tempdir) as tmpdirname:
            def foreach_func(named_src):
                name, src = named_src
                return call_cpp_compiler_src(src, name, compiler_config.compiler, ["-std=c++11", f"-O{compiler_config.opt_level}", "-c", "-fPIC", "-fvisibility=hidden", "-fvisibility-inlines-hidden"] + compiler_config.build_machine_flags() + [f"-I{v}" for v in _include_path], tmpdirname, compiler_config.obj_ext)
  
            libs = compiler_config.for_each(src, foreach_func)
            finallib = call_cpp_compiler(libs, libname, compiler_config.compiler, ["-shared"], tmpdirname, compiler_config.dll_ext)
            lib = KunRunner.Library.load(finallib)
        
    if Util.jit_debug_mode:
        print("[KUN_JIT] Source generation takes ", timeit.timeit(kuncompile, number=1), "s")
        print("[KUN_JIT] C++ compiler takes ", timeit.timeit(dowork, number=1), "s")
    else:
        kuncompile()
        dowork()
    return lib
