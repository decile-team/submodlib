#!/bin/bash
set -e -x

# List of Python versions to build for.
PYTHON_VERSIONS=("cp38-cp38" "cp39-cp39" "cp310-cp310" "cp311-cp311" "cp312-cp312" "cp313-cp313")

# Install build tools (if needed inside the container)
yum install -y gcc gcc-c++ make

# Build wheels for each specified Python version.
#for PYVER in "${PYTHON_VERSIONS[@]}"; do
#    PYBIN="/opt/python/${PYVER}/bin"
#    "${PYBIN}/pip" install --upgrade pip setuptools wheel pybind11
#    "${PYBIN}/pip" wheel /io/ -w /io/dist/
#done

# Repair the wheels using auditwheel.
for whl in /io/dist/*.whl; do
    auditwheel repair "$whl" -w /io/dist/
done

