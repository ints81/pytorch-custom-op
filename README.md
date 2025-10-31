# How to make custom C++/CUDA operations in PyTorch

기존 C++에서 `PYBIND11_MODULE` 매크로를 통한 C++/CUDA operation을 만드는 방법에서 
PyTorch 2.0.0 버전 이후 `TORCH_LIBRARY` 매크로로 바뀐 이후에 어떻게 C++/CUDA operation을 만드는지와 
custom C++/CUDA operation이 `torch.compile`를 거쳐도 에러가 발생하지 않는 C++/CUDA operation을 만드는 방법을 기록한 Repository

## Contents
1. C++의 `TORCH_LIBRARY` 매크로를 활용한 C++/CUDA operation 만드는 방법
2. 만든 C++/CUDA operation이 `torch.compile`로 처리될 수 있게 만드는 방법

## TODO
- 아직 에러 발생. `ImportError: /workspace/pytorch_custom_op/pytorch_custom_op/_C.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZNK3c107SymBool14guard_or_falseEPKcl`란 에러가 발생 중
