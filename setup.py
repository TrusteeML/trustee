import os
import re

from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


VERSIONFILE = "myniftyapp/_version.py"
verstrline = read(VERSIONFILE)
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError(f"Unable to find version string in {VERSIONFILE}.")

setup(
    name="trustee",
    version=verstr,
    packages=find_packages(),
    author="Arthur Selle Jacobs",
    author_email="asjacobs@inf.ufrgs.br",
    description="This package implements the Trustee framework to extract decision tree explanation from black-box ML models.",
    long_description=read("README.md"),
    keywords="trustee xai explainable ai machine-learning ml",
    license="BSD",
    python_requires=">=3.8<3.9",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        ("Programming Language :: Python :: Implementation :: CPython"),
        ("Programming Language :: Python :: Implementation :: PyPy"),
    ],
)
