import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
import datetime
import platform
import shutil
import glob



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
        # Run CMake
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={os.sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release"
        ]
        build_args = ["cmake", "--build", "."]
        devbuild = False
        if "KUN_BUILD_TESTS" in os.environ and os.environ["KUN_BUILD_TESTS"] != "":
            devbuild = True
            build_args += ["--target", "TestingTargets"]
        if is_windows:
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={ext_dir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={ext_dir}",
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE={ext_dir}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG={ext_dir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG={ext_dir}",
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG={ext_dir}",
                "-A", "x64"]
            build_args += ["--config", "Release"]
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
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.path = path

if os.environ.get("KUN_USE_GIT_VERSION", "0") != '0':
    git_ver = "." + datetime.datetime.now().strftime("%Y%m%d")
else:
    git_ver = ""

setup(
    name="KunQuant",
    version="0.1.1" + git_ver,
    description="A compiler, optimizer and executor for financial expressions and factors",
    long_description=open("Readme.md").read(),
    long_description_content_type="text/markdown",
    author="Menooker",
    author_email="menooker@live.com",
    packages=["KunQuant", "KunQuant.jit", "KunQuant.ops", "KunQuant.passes", "KunQuant.predefined", "KunQuant.runner"],
    package_dir={"KunQuant": "KunQuant"},
    package_data={"KunQuant": ["../cpp/Kun/*.hpp", "../cpp/Kun/Ops/*.hpp", "../cpp/KunSIMD/*.hpp", "../cpp/KunSIMD/cpu/*.hpp"]},
    include_package_data=True,
    ext_modules=[
        CMakeExtension("KunQuant.runner.KunRunner", "KunRunner", sourcedir="cpp"),
    ],
    cmdclass={
        "build_ext": CMakeBuildExtension,
    },
    install_requires=[
        # Add Python dependencies here
        "numpy",
    ],
    zip_safe=False,
)
