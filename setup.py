import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='Segmentation',
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize("segmentation_pyx.pyx")
)
