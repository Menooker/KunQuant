import os
import subprocess
import lit.formats

config.name = "KunQuant MLIR Tests"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir", ".py"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.obj_root

def prepend_env(name, entries):
    entries = [entry for entry in entries if entry]
    old = config.environment.get(name, "")
    if old:
        entries.append(old)
    config.environment[name] = os.pathsep.join(entries)

# Python GPU tests import the in-tree KunQuant package and load the freshly
# built KunQuant-MLIR extension module from KunQuantMLIR/.
prepend_env("PYTHONPATH", [config.project_source_dir])

# KunMLIR.abi3.so links against the downloaded LLVM/MLIR shared libraries.
# The CUDA toolkit path is also made explicit so both CuPy and the MLIR
# libdevice/ptxas discovery use the same installation as CMake.
config.environment["CUDA_PATH"] = config.cuda_toolkit_root
config.environment["CUDA_HOME"] = config.cuda_toolkit_root
prepend_env("PATH", [os.path.join(config.cuda_toolkit_root, "bin")])
prepend_env("LD_LIBRARY_PATH", [
    config.llvm_lib_dir,
    os.path.join(config.cuda_toolkit_root, "lib"),
    os.path.join(config.cuda_toolkit_root, "lib64"),
    os.path.join(config.cuda_toolkit_root, "lib64", "stubs"),
])

def detect_cuda_device():
    try:
        result = subprocess.run(
            [config.python_executable, "-c",
             "from KunQuant.jit.env import get_cuda_compute_capability; "
             "print(get_cuda_compute_capability())"],
            env=config.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=True)
        arch = result.stdout.strip()
        if arch:
            lit_config.note("CUDA device detected for Python tests: " + arch)
        return True
    except Exception as exc:
        lit_config.note("No CUDA device detected for Python tests: " + str(exc))
        return False

if detect_cuda_device():
    config.available_features.add("cuda-device")

# Tool substitutions
config.substitutions.append(("%kun-opt", config.kun_opt))
config.substitutions.append(("%python", config.python_executable))
config.substitutions.append(
    ("%FileCheck", os.path.join(config.llvm_tools_dir, "FileCheck"))
)

# Exclude non-test directories from discovery
config.excludes = [
    "CMakeLists.txt",
    "lit.cfg.py",
    "lit.site.cfg.py.in",
    "utils.py",
]
