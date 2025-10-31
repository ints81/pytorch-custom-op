#include <torch/library.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Operators.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_kernel(int numel, const float* a, const float* b, float* c) {
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gidx < numel) {
        c[gidx] = a[gidx] + b[gidx];
    }
}

at::Tensor add_cuda(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(b.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_CHECK(a.sizes() == b.sizes());

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor c = at::empty_like(a_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int numel = a_contig.numel();
    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* c_ptr = c.data_ptr<float>();
    add_kernel<<<(numel + 255) / 256, 256, 0, stream>>>(numel, a_ptr, b_ptr, c_ptr);

    return c;
}

// 반드시 m.impl의 첫 argument의 이름은 위에서 선언한 다른 function의 이름과 달라야 함
TORCH_LIBRARY_IMPL(pytorch_custom_op, CUDA, m) {
    m.impl("myadd", &add_cuda);
}