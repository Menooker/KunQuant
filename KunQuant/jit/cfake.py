import os
import platform
import subprocess
import tempfile
from typing import List, Tuple, Union
import sys
if sys.version_info[1] < 9:
    from typing import Callable
else:
    from collections.abc import Callable
import KunQuant.runner.KunRunner as KunRunner
from KunQuant.Driver import compileit as driver_compileit
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Stage import Function
from KunQuant.passes import Util
from KunQuant.jit.env import get_compiler_env, get_msvc_compiler_dir
import timeit
import dataclasses
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import shutil

_cpp_root = os.path.join(os.path.dirname(__file__), "..", "..", "cpp")
_include_path = [_cpp_root]
_runtime_path = os.path.dirname(KunRunner.getRuntimePath())

_os_name = platform.system()
_win32 = _os_name == "Windows"

FunctionList = List[Tuple[str, str]]
CallableOnFunction = Callable[[Tuple[str, str]], List[str]]

def is_windows():
    return _win32

def get_runtime_path() -> str:
    return _runtime_path

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
    avx2: bool = True
    fma: bool = True
    avx512: bool = False
    avx512dq: bool = False
    avx512vl: bool = False

class NativeCPUFlags:
    pass

class MSVCCommandLineBuilder:
    @staticmethod
    def build_compile_options(cfg: 'CppCompilerConfig', srcpath: str, outpath: str) -> List[str]:
        cmd = [cfg.compiler, "/nologo", "/c", "/EHsc", f"/O{min(cfg.opt_level, 2)}", "/wd4251", "/wd4200", "/wd4305", srcpath] + [f"/I{v}" for v in _include_path]
        if isinstance(cfg.machine, NativeCPUFlags):
            # todo: should use native cpu flags
            cmd.append("/arch:AVX2")
        else:
            if cfg.machine.fma or cfg.machine.avx2:
                cmd.append("/arch:AVX2")
            else:
                cmd.append("/arch:AVX")
            if cfg.machine.avx512 or cfg.machine.avx512dq or cfg.machine.avx512vl:
                cmd.append("/arch:AVX512")
            cmd.append("/arch:AVX2")
        cmd.append(f"/Fo{outpath}")
        return cmd

    @staticmethod
    def build_link_options(cfg: 'CppCompilerConfig', paths: List[str], outpath: str) -> List[str]:
        cmd = [cfg.compiler, "/nologo", "/MP", "/LD", "KunRuntime.lib"]
        cmd += paths
        cmd.append(f"/Fe{outpath}")
        cmd += ["/link", f'/LIBPATH:"{_runtime_path}"']
        return cmd

class GCCCommandLineBuilder:
    @staticmethod
    def build_compile_options(cfg: 'CppCompilerConfig', srcpath: str, outpath: str) -> List[str]:
        cmd = [cfg.compiler, "-std=c++11", f"-O{cfg.opt_level}", "-c", "-fPIC", "-fvisibility=hidden", "-fvisibility-inlines-hidden"] + list(cfg.other_flags)
        if 'clang' in cfg.compiler:
            cmd += ["-Wno-unused-value"]
        if isinstance(cfg.machine, NativeCPUFlags):
            cmd += ["-march=native"]
        else:
            if cfg.machine.avx2:
                cmd.append("-mavx2")
            else:
                cmd.append("-mavx")
            if cfg.machine.fma:
                cmd.append("-mfma")
            if cfg.machine.avx512:
                cmd.append("-mavx512f")
            if cfg.machine.avx512dq:
                cmd.append("-mavx512dq")
            if cfg.machine.avx512vl:
                cmd.append("-mavx512vl")
        cmd += [f"-I{v}" for v in _include_path]
        cmd += [srcpath, "-o", outpath]
        return cmd

    @staticmethod
    def build_link_options(cfg: 'CppCompilerConfig', paths: List[str], outpath: str) -> List[str]:
        ret = [cfg.compiler] + paths + ["-l", "KunRuntime", "-shared", "-L", _runtime_path, "-o", outpath] + list(cfg.other_flags)
        if cfg.fast_linker_threads:
            ret.append("-fuse-ld=gold")
            ret.append("-Wl,--threads")
            ret.append(f"-Wl,--thread-count={cfg.fast_linker_threads}")
        return ret

_config = {
    "Windows": ("cl.exe", "obj", "dll", MSVCCommandLineBuilder),
    "Linux": ("g++", "o", "so", GCCCommandLineBuilder),
    "Darwin": ("clang++", "o", "dylib", GCCCommandLineBuilder)
}
@dataclass
class CppCompilerConfig:
    opt_level: int = 3
    machine: Union[NativeCPUFlags, X64CPUFlags] = NativeCPUFlags()
    for_each: Callable[[FunctionList, CallableOnFunction], List[str]] = multi_thread_compile
    other_flags : Tuple[str] = ()
    fast_linker_threads: int = 0
    compiler: str = _config[_os_name][0]
    obj_ext: str = _config[_os_name][1]
    dll_ext: str = _config[_os_name][2]
    builder = _config[_os_name][3]



def call_cpp_compiler(cmd: List[str], outpath: str) -> str:
    if Util.jit_debug_mode:
        print("[KUN_JIT] temp jit files:", outpath)
    if Util.jit_debug_mode:
        print("[KUN_JIT] cmd:", cmd)
    subprocess.check_call(cmd, shell=False, env=get_compiler_env(), stderr=subprocess.STDOUT,
            universal_newlines=True,
            creationflags=(subprocess.CREATE_NO_WINDOW if _win32 and not Util.jit_debug_mode else 0))
    return outpath

def call_cpp_compiler_src(source: str, module_name: str, compiler: CppCompilerConfig, tempdir: str) -> str:
    inpath = os.path.join(tempdir, f"{module_name}.cpp")
    outpath = os.path.join(tempdir, f"{module_name}.{compiler.obj_ext}")
    with open(inpath, 'w') as f:
        f.write(source)
    return call_cpp_compiler( compiler.builder.build_compile_options(compiler, inpath, outpath), outpath)

class _fake_temp:
    def __init__(self, dir: str, module_name: str, keep_files: bool) -> None:
        if not keep_files:
            self.dir = tempfile.mkdtemp(dir=dir)
        else:
            self.dir = os.path.join(dir, module_name)
            os.makedirs(self.dir, exist_ok=True)
        self.keep_files = keep_files or _win32

    def __enter__(self):
        return self.dir

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if not self.keep_files:
            shutil.rmtree(self.dir)

def compileit(func: List[Tuple[str, Function, KunCompilerConfig]], libname: str, compiler_config: CppCompilerConfig, tempdir: str = None, keep_files: bool = False, load: bool = True) -> Union[KunRunner.Library, str]:
    get_compiler_env() # trigger cache
    lib = None
    src: List[Tuple[str, str]] = []
    if keep_files and not tempdir:
        raise RuntimeError("if keep_files=True, tempdir should not be empty")
    def kuncompile():
        for name, f, cfg in func:
            for idx, src_str in enumerate(driver_compileit(f, name, **dataclasses.asdict(cfg))):
                src.append((f"{name}_{idx}", src_str))
    def dowork():
        nonlocal lib
        with _fake_temp(tempdir, libname, keep_files) as tmpdirname:
            def foreach_func(named_src):
                name, src = named_src
                return call_cpp_compiler_src(src, name, compiler_config, tmpdirname)
  
            libs = compiler_config.for_each(src, foreach_func)
            finallib = os.path.join(tmpdirname, f"{libname}.{compiler_config.dll_ext}")
            finallib = call_cpp_compiler(compiler_config.builder.build_link_options(compiler_config, libs, finallib), finallib)
            if load:
                lib = KunRunner.Library.load(finallib)
                if _win32 and not keep_files:
                    def cleanup():
                        if Util.jit_debug_mode:
                            print("Cleanup temp dir", tmpdirname)
                        shutil.rmtree(tmpdirname)
                    lib.setCleanup(cleanup)
            else:
                lib = finallib
        
    if Util.jit_debug_mode:
        print("[KUN_JIT] Source generation takes ", timeit.timeit(kuncompile, number=1), "s")
        print("[KUN_JIT] C++ compiler takes ", timeit.timeit(dowork, number=1), "s")
    else:
        kuncompile()
        dowork()
    return lib
