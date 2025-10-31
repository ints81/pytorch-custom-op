from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import torch

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# For building custom cuda kernel faster and some cuda functions such as '__nanosleep'
compute_capability = [str(elem) for elem in torch.cuda.get_device_capability()]

CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
NVCC_FLAGS = [f"-arch=sm_{''.join(compute_capability)}", "-std=c++17"]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

csrcs = [
    "csrc/custom_op.cu",
    "csrc/torch_bindings.cpp",
]
ext_modules = []

cuda_ext = CUDAExtension(
    name="pytorch_custom_op._C",
    sources=csrcs,
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(cuda_ext)

setup(
    name="pytorch_custom_op",
    version="0.0.1",
    description="Package that implements a custom C++/CUDA op for PyTorch.",
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension
    },
)