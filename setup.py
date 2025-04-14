from setuptools import setup, find_packages, Distribution
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Read the long description from the README file.
with open("README.md", "r") as fh:
    long_description = fh.read()

# Get the version from your version file.
exec(open('submodlib/version.py').read())

# Custom distribution to indicate the presence of binary extensions.
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

# Define your extension modules.
ext_modules = [
    Pybind11Extension(
        name="submodlib_cpp",
        sources=[
            "cpp/SetFunction.cpp",
            "cpp/utils/helper.cpp",
            "cpp/wrappers/wr_helper.cpp",
            "cpp/utils/sparse_utils.cpp",
            "cpp/wrappers/wr_sparse_utils.cpp",
            "cpp/optimizers/NaiveGreedyOptimizer.cpp",
            "cpp/optimizers/LazyGreedyOptimizer.cpp",
            "cpp/optimizers/StochasticGreedyOptimizer.cpp",
            "cpp/optimizers/LazierThanLazyGreedyOptimizer.cpp",
            "cpp/wrappers/wrapper.cpp",
            "cpp/submod/FacilityLocation.cpp", "cpp/wrappers/wr_FacilityLocation.cpp",
            "cpp/submod/FacilityLocation2.cpp", "cpp/wrappers/wr_FacilityLocation2.cpp",
            "cpp/submod/DisparitySum.cpp", "cpp/wrappers/wr_DisparitySum.cpp",
            "cpp/submod/FeatureBased.cpp", "cpp/wrappers/wr_FeatureBased.cpp",
            "cpp/submod/GraphCut.cpp", "cpp/wrappers/wr_GraphCut.cpp",
            "cpp/submod/SetCover.cpp", "cpp/wrappers/wr_SetCover.cpp",
            "cpp/submod/ProbabilisticSetCover.cpp", "cpp/wrappers/wr_ProbabilisticSetCover.cpp",
            "cpp/submod/DisparityMin.cpp", "cpp/wrappers/wr_DisparityMin.cpp",
            "cpp/submod/LogDeterminant.cpp", "cpp/wrappers/wr_LogDeterminant.cpp",
            "cpp/smi/FacilityLocationMutualInformation.cpp", "cpp/wrappers/wr_FacilityLocationMutualInformation.cpp",
            "cpp/smi/FacilityLocationVariantMutualInformation.cpp", "cpp/wrappers/wr_FacilityLocationVariantMutualInformation.cpp",
            "cpp/smi/ConcaveOverModular.cpp", "cpp/wrappers/wr_ConcaveOverModular.cpp",
            "cpp/smi/GraphCutMutualInformation.cpp", "cpp/wrappers/wr_GraphCutMutualInformation.cpp",
            "cpp/condgain/GraphCutConditionalGain.cpp", "cpp/wrappers/wr_GraphCutConditionalGain.cpp",
            "cpp/condgain/ConditionalGain.cpp",
            "cpp/condgain/FacilityLocationConditionalGain.cpp", "cpp/wrappers/wr_FacilityLocationConditionalGain.cpp",
            "cpp/condgain/LogDeterminantConditionalGain.cpp", "cpp/wrappers/wr_LogDeterminantConditionalGain.cpp",
            "cpp/condgain/ProbabilisticSetCoverConditionalGain.cpp", "cpp/wrappers/wr_ProbabilisticSetCoverConditionalGain.cpp",
            "cpp/smi/ProbabilisticSetCoverMutualInformation.cpp", "cpp/wrappers/wr_ProbabilisticSetCoverMutualInformation.cpp",
            "cpp/smi/MutualInformation.cpp",
            "cpp/smi/LogDeterminantMutualInformation.cpp", "cpp/wrappers/wr_LogDeterminantMutualInformation.cpp",
            "cpp/smi/SetCoverMutualInformation.cpp", "cpp/wrappers/wr_SetCoverMutualInformation.cpp",
            "cpp/condgain/SetCoverConditionalGain.cpp", "cpp/wrappers/wr_SetCoverConditionalGain.cpp",
            "cpp/cmi/FacilityLocationConditionalMutualInformation.cpp", "cpp/wrappers/wr_FacilityLocationConditionalMutualInformation.cpp",
            "cpp/cmi/LogDeterminantConditionalMutualInformation.cpp", "cpp/wrappers/wr_LogDeterminantConditionalMutualInformation.cpp",
            "cpp/cmi/SetCoverConditionalMutualInformation.cpp", "cpp/wrappers/wr_SetCoverConditionalMutualInformation.cpp",
            "cpp/cmi/ProbabilisticSetCoverConditionalMutualInformation.cpp", "cpp/wrappers/wr_ProbabilisticSetCoverConditionalMutualInformation.cpp",
            "cpp/Clustered.cpp", "cpp/wrappers/wr_Clustered.cpp"
        ],
        extra_compile_args=['-O3'],
        define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    version=__version__,
    description='submodlib is an efficient and scalable library for submodular optimization which finds its application in summarization, data subset selection, hyper parameter tuning etc.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Vishal Kaushal, Krishnateja Killamsetty, Suraj Kothawade, Rishabh Iyer',
    author_email='vishal.kaushal@gmail.com',
    url="https://github.com/vishkaush/submodlib",
    license='MIT',
    # Restrict package discovery to only your intended packages.
    packages=find_packages(include=["submodlib", "submodlib.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    distclass=BinaryDistribution,
    install_requires=["numpy>=1.24.4", "scipy", "scikit-learn", "numba", "matplotlib", "tqdm", "pandas", "joblib",],
    setup_requires=['pybind11'],
    tests_require=['pytest'],
    test_suite='tests',
    zip_safe=False,
)
