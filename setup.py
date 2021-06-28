import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="scikit-explain",
    version="0.1",
    packages=find_packages(),
    author="Arthur Selle Jacobs",
    author_email="asjacobs@inf.ufrgs.br",
    description="This package is a toolkit of eXplainable Artificial Integillence (XAI) methods to interpret sklearn models.",
    long_description=read('README.md'),
    keywords="skexplain xai explainable ai machine-learning ml sklearn",
    license="BSD",
    python_requires=">=3.6.1",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        ('Programming Language :: Python :: Implementation :: CPython'),
        ('Programming Language :: Python :: Implementation :: PyPy')
    ]
)
