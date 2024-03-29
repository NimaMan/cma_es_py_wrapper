import os 
from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir, get_include

import sys

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)


here = os.path.dirname(os.path.realpath(__file__))


ext_modules = [
    Pybind11Extension("es",
        ["src/wrapper.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        include_dirs=[get_include(), os.path.join(here, "lib", "eigen3") ],
        extra_complier_args=["-O2", "-fopenmp"]
        ),
]

setup(
    name="es",
    version=__version__,
    author="Nima",
    author_email="ndizbin14@ku.edu.tr",
    url="",
    description="A python wrapper around the libcmaes library using pybind11",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
