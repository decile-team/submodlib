cmake_minimum_required(VERSION 2.8.12)
project(submodlib)

SET(SOURCE_DIR "src/submodlib_alt/cpp")
include_directories(${SOURCE_DIR})
SET(SOURCES "${SOURCE_DIR}/HelloWorld/HelloWorld.cpp")

find_package(pybind11 REQUIRED)
pybind11_add_module(submodlib_alt ${SOURCES} "${SOURCE_DIR}/HelloWorld/wrapper.cpp")