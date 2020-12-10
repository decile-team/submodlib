from setuptools import find_packages, setup
import sys

try:
    from pybind11.setup_helpers import Pybind11Extension
except ImportError:
    from setuptools import Extension as Pybind11Extension


with open("README.md", "r") as fh:
    long_description = fh.read()

exec(open('submodlib/version.py').read())

ext_modules = [
    Pybind11Extension("submodlib_cpp",
        ["cpp/submod/wrapper.cpp","cpp/submod/FacilityLocation.cpp", "cpp/submod/wr_FacilityLocation.cpp", "cpp/submod/helper.cpp", "cpp/submod/wr_helper.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        ),
]


setup(
    name='submodlib6',
    #packages=find_packages(include=['submodlib']),
    packages=['submodlib', 'submodlib/functions'],
    #packages=find_packages('submodlib'),
    #package_dir={'':'submodlib'},
    #version='0.0.2',
    version=__version__,
    description='submodlib is an efficient and scalable library for submodular optimization which finds its application in summarization, data subset selection, hyper parameter tuning etc.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Vishal Kaushal',
    ext_modules=ext_modules,
    author_email='vishal.kaushal@gmail.com',
    url="https://github.com/vishkaush/submodlib",
    #url='http://pypi.python.org/pypi/submodlib/',
    #url="https://github.com/pypa/sampleproject",
    license='MIT',
    # install_requires=[
    #     "numpy >= 1.14.2",
    #     "scipy >= 1.0.0",
    #     "numba >= 0.43.0",
    #     "tqdm >= 4.24.0",
    #     "nose"
    # ],
    install_requires=[],
    setup_requires=['pybind11','pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
    #classifiers=[
    #    "Programming Language :: Python :: 3",
    #    "License :: OSI Approved :: MIT License",
    #    "Operating System :: OS Independent",
    #],
    zip_safe=False 
)