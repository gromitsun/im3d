from distutils.core import setup
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

pyx_files = ['pyx_src/shapes.pyx',]

setup(ext_modules = cythonize(pyx_files))