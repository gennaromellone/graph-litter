from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Function Optimization',
    ext_modules=cythonize("optimization.pyx"),
)
#python setup.py build_ext --inplace