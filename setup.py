import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
import datetime
import platform
import shutil
import glob

# Stable-ABI / abi3 baseline.  nanobind's STABLE_ABI mode (see
# 3rdparty/nanobind/cmake/nanobind-config.cmake) enables itself only on
# CPython >= 3.12 with `Development.SABIModule` available; below that
# nanobind silently builds a per-Python-version `.cpython-3X-*.so`.
# Mirror the same threshold here so the wheel filename matches:
#   * py >= 3.12  →  KunRunner.abi3.so  →  tag wheel `cpYY-abi3-*`
#                    (single wheel covers 3.12, 3.13, 3.14, ...)
#   * py <  3.12  →  KunRunner.cpython-3X-*.so  →  per-version tag
_STABLE_ABI_MIN = (3, 12)
_HAS_STABLE_ABI = (sys.version_info >= _STABLE_ABI_MIN and
                    platform.python_implementation() == "CPython")


class CMakeBuildExtension(build_ext):
    def build_extension(self, ext):
        # Get the directory containing the extension
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        is_windows = platform.system() == "Windows"
        if self.build_temp.endswith("\\Release"):
            build_temp = os.path.abspath(os.path.join(self.build_temp, ".."))
        else:
            build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)
        release_or_debug = os.environ.get("KUN_BUILD_TYPE", "Release")
        # Run CMake.  Modern FindPython uses `Python_EXECUTABLE` (not the
        # legacy `PYTHON_EXECUTABLE` from FindPythonInterp/Libs).  We pass
        # both for backward compatibility with consumers that may still
        # query the lowercase name; CMake's FindPython respects the new one.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={release_or_debug}"
        ]
        if "KUN_SANITIZER" in os.environ and os.environ["KUN_SANITIZER"] != "0":
            cmake_args += [f"-DKUN_SANITIZER=ON"]
        else:
            cmake_args += [f"-DKUN_SANITIZER=OFF"]
        build_args = ["cmake", "--build", "."]
        devbuild = False
        if "KUN_BUILD_TESTS" in os.environ and os.environ["KUN_BUILD_TESTS"] != "0":
            devbuild = True
            build_args += ["--target", "TestingTargets"]
        DKUN_NO_AVX = "OFF"
        if "KUN_NO_AVX2" in os.environ and os.environ["KUN_NO_AVX2"] != "0":
            DKUN_NO_AVX = "ON"
        cmake_args += [f"-DKUN_NO_AVX2={DKUN_NO_AVX}"]
        if is_windows:
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={ext_dir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={ext_dir}",
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE={ext_dir}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG={ext_dir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG={ext_dir}",
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG={ext_dir}",
                "-A", "x64"]
            build_args += ["--config", release_or_debug]
        else:
            build_args += ["--", "-j"]

        if "PLAT" in os.environ:
            del os.environ["PLAT"]
        subprocess.check_call(["cmake", os.path.join(ext.sourcedir, "..")] + cmake_args, cwd=build_temp)
        subprocess.check_call(build_args, cwd=build_temp)
        if devbuild:
            print("Copy dll files")
            ext_table = {"Windows": ["*.dll", "*.lib"], "Linux": ["*.so"], "Darwin": ["*.dylib"]}
            for fn in ext_table[platform.system()]:
                for file in glob.glob(os.path.join(ext_dir, fn)):
                    print("copy from debug:", file)
                    shutil.copy(file, os.path.join(".", "KunQuant", "runner"))

class CMakeExtension(Extension):
    def __init__(self, name, path, sourcedir=""):
        super().__init__(name, sources=[], py_limited_api=_HAS_STABLE_ABI)
        self.sourcedir = os.path.abspath(sourcedir)
        self.path = path


# Tag the wheel `cp312-abi3-*` when we know nanobind will produce a
# stable-ABI .so (Python >= 3.12 on CPython).  Without this override
# setuptools defaults to `cp3XX-cp3XX-*` (per-version) — wrong for
# abi3 builds because pip would then refuse to install our 3.12 wheel
# on 3.13.  Below 3.12 we keep the default per-version tag.
try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:                                 # setuptools < 70
    from wheel.bdist_wheel import bdist_wheel       # type: ignore

class BdistWheelABI3(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        if _HAS_STABLE_ABI:
            # Tag as e.g. `cp312-abi3-manylinux_2_28_x86_64`.
            self.py_limited_api = "cp{}{}".format(*_STABLE_ABI_MIN)
            # The extension is platform-specific; don't let setuptools
            # mark the wheel as pure-python.
            self.root_is_pure = False


if os.environ.get("KUN_USE_GIT_VERSION", "0") != '0':
    git_ver = "." + datetime.datetime.now().strftime("%Y%m%d")
else:
    git_ver = ""

setup(
    name="KunQuant",
    version="0.1.10" + git_ver,
    description="A compiler, optimizer and executor for financial expressions and factors",
    long_description=open("Readme.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    author="Menooker",
    author_email="menooker@live.com",
    packages=["KunQuant", "KunQuant.jit", "KunQuant.ops", "KunQuant.passes", "KunQuant.predefined", "KunQuant.runner"],
    package_dir={"KunQuant": "KunQuant"},
    package_data={"KunQuant": ["../cpp/Kun/*.hpp", "../cpp/Kun/Ops/*.hpp", "../cpp/KunSIMD/*.hpp", "../cpp/KunSIMD/cpu/*.hpp",
                              "../cpp/KunSIMD/cpu/x86/*.hpp", "../cpp/KunSIMD/cpu/neon/*.hpp"]},
    include_package_data=True,
    ext_modules=[
        CMakeExtension("KunQuant.runner.KunRunner", "KunRunner", sourcedir="cpp"),
    ],
    cmdclass={
        "build_ext":   CMakeBuildExtension,
        "bdist_wheel": BdistWheelABI3,
    },
    python_requires=">=3.9",
    install_requires=[
        # Add Python dependencies here
        "numpy",
    ],
    zip_safe=False,
)
