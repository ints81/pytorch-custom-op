#include <Python.h>
#include <torch/library.h>

TORCH_LIBRARY(pytorch_custom_op, m) {
    m.def("myadd(Tensor a, Tensor b) -> Tensor");
}

// 이거 없으면 package 설치 이후 import할 때 에러 발생
// vLLM에서 Macro로 코드 정리한 거 활용하면 좋을 듯
PyMODINIT_FUNC PyInit__C(void) {
  static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_C",
    NULL,
    -1,
    NULL,
  };
  return PyModule_Create(&module);
}