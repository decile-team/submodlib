#!/bin/bash
set -e -x

# Define the Python versions you want to build for
PYTHON_VERSIONS=("cp38-cp38" "cp39-cp39" "cp310-cp310" "cp311-cp311" "cp312-cp312")

# Install necessary system packages
yum install -y gcc gcc-c++ make

# Compile wheels for each Python version
for PYVER in "${PYTHON_VERSIONS[@]}"; do
    PYBIN="/opt/python/${PYVER}/bin"
    "${PYBIN}/pip" install --upgrade pip
    "${PYBIN}/pip" install setuptools wheel
    "${PYBIN}/pip" wheel /io/ -w /io/dist/
done

# Bundle external shared libraries into the wheels
for whl in /io/dist/*.whl; do
    auditwheel repair "$whl" -w /io/dist/
done

