import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension, 

setup(
    ### Minimal Example Args
    name="stars-block-cuda",
    install_requires=["torch", "pybind11"],
    ### PyTorch C++/CUDA Examples
    ### NOTE: /Zc:__cplusplus is related to MSVC incorrectly setting __cplusplus macro. See CUTLASS CMAKE and 
    ### https://github.com/NVIDIA/cutlass/issues/1474
    ext_modules=[
        CUDAExtension(
            name="stars_fused_parallel_linear", sources=["TorchFusedParallelLinear.cu"]
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)