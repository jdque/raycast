from setuptools import setup
# from Cython.Build import cythonize
# import numpy

# setup(
#   name = "Hello",
#   ext_modules = cythonize('geometry_c.pyx'),
#   include_dirs=[numpy.get_include()]
# )

setup(
    name = "raycast",
    version = "0.1",
    install_requires = [
        'numpy',
        'pysdl2',
        'pyopengl'
    ]
)