import os
import lit.formats

config.name = "KunQuant MLIR Tests"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.obj_root

# Tool substitutions
config.substitutions.append(("%kun-opt", config.kun_opt))
config.substitutions.append(
    ("%FileCheck", os.path.join(config.llvm_tools_dir, "FileCheck"))
)

# Exclude non-test directories from discovery
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py.in"]
