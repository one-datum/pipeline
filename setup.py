#!/usr/bin/env python

import codecs
import os

from setuptools import find_packages, setup

HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


setup(
    name="one_datum",
    author="Dan Foreman-Mackey",
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/dfm/one-datum",
    license="MIT",
    description=(
        "What can we infer about an orbit from the RV or astrometric jitter?"
    ),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "astropy",
        "kepler.py",
        "numpy",
        "scipy",
    ],
    extras_require={"test": "pytest"},
)
