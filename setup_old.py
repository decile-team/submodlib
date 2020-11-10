from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

exec(open('submodlib/version.py').read())

setup(
    name='submodlib',
    #packages=find_packages(include=['submodlib']),
    packages=['submodlib', 'submodlib/functions'],
    #version='0.0.2',
    version=__version__,
    description='submodlib is an efficient and scalable library for submodular optimization which finds its application in summarization, data subset selection, hyper parameter tuning etc.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Vishal Kaushal',
    author_email='vishal.kaushal@gmail.com',
    url='http://pypi.python.org/pypi/submodlib/',
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
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
    #classifiers=[
    #    "Programming Language :: Python :: 3",
    #    "License :: OSI Approved :: MIT License",
    #    "Operating System :: OS Independent",
    #],
)
