# most parts of this file was originated from cupy project at
# https://github.com/cupy/cupy/blob/main/cupy/cuda/compiler.py

import shutil
from typing import List, Optional, Tuple
import platform
import os
import sys
import copy
from KunQuant.passes.Util import jit_debug_mode

def _normalize_arch() -> str:
    arch = platform.machine().lower()
    if arch in ('x86_64', 'amd64', 'i386'):
        return 'x86_64'
    if arch in ('aarch64', 'arm64'):
        return 'aarch64'
    return arch

cpu_arch = _normalize_arch()
_win32 = sys.platform.startswith('win32')
def _get_extra_path_for_msvc():
    cl_exe = shutil.which('cl.exe')
    if cl_exe:
        # The compiler is already on PATH, no extra path needed.
        return None, None, None

    cl_exe_dir, include, lib = _get_cl_exe_dir()
    if cl_exe_dir:
        return cl_exe_dir, include, lib

    cl_exe_dir, include = _get_cl_exe_dir_fallback()
    if cl_exe_dir:
        return cl_exe_dir, include, lib

    return None, None


def _get_cl_exe_dir() -> Tuple[Optional[str], Optional[List[str]], Optional[List[str]]]:
    try:
        try:
            # setuptools.msvc is missing in setuptools v74.0.0.
            # setuptools.msvc requires explicit import in setuptools v74.1.0+.
            import setuptools.msvc
        except Exception:
            return None, None, None
        env_info = setuptools.msvc.EnvironmentInfo(platform.machine())
        vctools = env_info.VCTools
        for path in vctools:
            cl_exe = os.path.join(path, 'cl.exe')
            os_includes = env_info.UCRTIncludes + env_info.OSIncludes
            os_libs = env_info.UCRTLibraries + env_info.OSLibraries
            if os.path.exists(cl_exe):
                return path, os_includes, os_libs
        print(f'cl.exe could not be found in {vctools}')
    except Exception as e:
        print(
            f'Failed to find cl.exe with setuptools.msvc: {type(e)}: {e}')
    return None, None, None


def _get_cl_exe_dir_fallback() -> Tuple[Optional[str], Optional[List[str]], Optional[List[str]]]:
    # Discover cl.exe without relying on undocumented setuptools.msvc API.
    # As of now this code path exists only for setuptools 74.0.0 (see #8583).
    # N.B. This takes few seconds as this incurs cmd.exe (vcvarsall.bat)
    # invocation.
    try:
        from setuptools import Distribution
        from setuptools.command.build_ext import build_ext
        ext = build_ext(Distribution({'name': 'cupy_cl_exe_discover'}))
        ext.setup_shlib_compiler()
        ext.shlib_compiler.initialize()  # MSVCCompiler only
        return os.path.dirname(ext.shlib_compiler.cc), ext.shlib_compiler.include_dirs, ext.shlib_compiler.library_dirs
    except Exception as e:
        print(
            f'Failed to find cl.exe with setuptools: {type(e)}: {e}')
    return None, None

_env = None
_dir = None
_includes = None
_libs = None
def get_msvc_compiler_dir() -> Tuple[str, List[str], List[str]]:
    global _dir, _includes, _libs
    if _dir:
        return _dir, _includes
    _dir, _includes, _libs = _get_extra_path_for_msvc()
    return _dir, _includes, _libs

def get_compiler_env():
    global _env
    if _env:
        return _env
    env = os.environ
    if _win32:
        extra_path, includes, libs = get_msvc_compiler_dir()
        if extra_path is not None:
            path = extra_path + os.pathsep + os.environ.get('PATH', '')
            env = copy.deepcopy(env)
            env['PATH'] = path
            env['LIB'] =os.pathsep.join([os.path.abspath(os.path.join(extra_path, "..", "..", "..", "lib", 'x64'))] + libs)
            env['INCLUDE'] = os.pathsep.join([os.path.abspath(os.path.join(extra_path, "..", "..", "..", "include"))] + includes)
            if jit_debug_mode:
                print("Reset env", "PATH+=", extra_path, "INCLUDE=", env['INCLUDE'], "LIB=", env['LIB'])
    _env = env
    return env
