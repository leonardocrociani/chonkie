from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "chonkie.chunker.c_extensions.token_chunker",
        ["src/chonkie/chunker/c_extensions/token_chunker.pyx"],
    )
]

setup(
    ext_modules=cythonize(extensions),
)
