#!/usr/bin/env python
import io

from setuptools import find_packages, setup


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

description = ("Python tools for building 'Benchmarks and Challenges for "
               "Neuromorphic Engineering' paper.")
long_description = read('README.rst')

url = "https://github.com/tbekolay/fnme2015"
setup(
    name="fnme",
    version="0.1.0",
    author="Trevor Bekolay",
    author_email="tbekolay@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    scripts=[],
    url=url,
    description=description,
    long_description=long_description,
    install_requires=[
        "doit",
        "nengo",
        "numpy",
        "matplotlib",
    ],
)
