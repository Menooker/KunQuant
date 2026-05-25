import datetime
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


_STABLE_ABI_MIN = (3, 12)
_HAS_STABLE_ABI = (
    sys.version_info >= _STABLE_ABI_MIN
    and platform.python_implementation() == "CPython"
)

_PKG_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_ROOT.parents[1]
_VERSION_BASE = "0.1.10"


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: Path):
        super().__init__(name, sources=[], py_limited_api=_HAS_STABLE_ABI)
        self.sourcedir = str(sourcedir)


class CMakeBuildExtension(build_ext):
    def build_extension(self, ext):
        ext_dir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)
        ext_dir.mkdir(parents=True, exist_ok=True)

        build_type = os.environ.get("KUN_BUILD_TYPE", "Release")
        python_exe = sys.executable
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DKUN_MLIR_PYTHON_PACKAGE_DIR={ext_dir}",
            "-DKUN_BUILD_CPU_RUNNER=OFF",
            "-DKUN_BUILD_MLIR=ON",
            f"-DPython_EXECUTABLE={python_exe}",
            f"-DPYTHON_EXECUTABLE={python_exe}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ]

        if os.environ.get("KUN_SANITIZER", "0") != "0":
            cmake_args.append("-DKUN_SANITIZER=ON")
        else:
            cmake_args.append("-DKUN_SANITIZER=OFF")

        if os.environ.get("KUN_NO_AVX2", "0") != "0":
            cmake_args.append("-DKUN_NO_AVX2=ON")
        else:
            cmake_args.append("-DKUN_NO_AVX2=OFF")

        for var in (
            "LLVM_DIR",
            "MLIR_DIR",
            "CUDAToolkit_ROOT",
            "CMAKE_CUDA_COMPILER",
            "LLVM_EXTERNAL_LIT",
        ):
            value = os.environ.get(var)
            if value:
                cmake_args.append(f"-D{var}={value}")

        generator = os.environ.get("CMAKE_GENERATOR")
        if not generator and shutil.which("ninja"):
            cmake_args.extend(["-G", "Ninja"])

        if "PLAT" in os.environ:
            del os.environ["PLAT"]

        subprocess.check_call(
            ["cmake", "-S", ext.sourcedir, "-B", str(build_temp)] + cmake_args
        )

        build_args = [
            "cmake",
            "--build",
            str(build_temp),
            "--target",
            "KunMLIR",
        ]
        if platform.system() == "Windows":
            build_args += ["--config", build_type]
        else:
            build_args += ["--parallel"]
        subprocess.check_call(build_args)


try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel  # type: ignore


class BdistWheelABI3(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        if _HAS_STABLE_ABI:
            self.py_limited_api = "cp{}{}".format(*_STABLE_ABI_MIN)
            self.root_is_pure = False


if os.environ.get("KUN_USE_GIT_VERSION", "0") != "0":
    git_ver = "." + datetime.datetime.now().strftime("%Y%m%d")
else:
    git_ver = ""

version = _VERSION_BASE + git_ver
package_dir = os.path.relpath(_REPO_ROOT / "KunQuantMLIR", _PKG_ROOT)


setup(
    name="KunQuant-MLIR",
    version=version,
    description="Optional MLIR/CUDA backend for KunQuant",
    long_description=(_REPO_ROOT / "Readme.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Menooker",
    author_email="menooker@live.com",
    packages=["KunQuantMLIR"],
    package_dir={"KunQuantMLIR": package_dir},
    package_data={"KunQuantMLIR": ["*.so", "*.pyd", "*.dll", "*.dylib"]},
    include_package_data=True,
    ext_modules=[
        CMakeExtension("KunQuantMLIR.KunMLIR", _REPO_ROOT),
    ],
    cmdclass={
        "build_ext": CMakeBuildExtension,
        "bdist_wheel": BdistWheelABI3,
    },
    python_requires=">=3.9",
    install_requires=[
        f"KunQuant=={version}",
        "numpy",
    ],
    zip_safe=False,
)
