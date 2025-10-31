# 이거 없으면 register_fake에서 에러 발생
import pytorch_custom_op._C

import torch
from torch import Tensor


# torch.compile에서 이 operation을 확인할 때 각 argument의 타입이 필요함
def myadd(a: Tensor, b: Tensor) -> Tensor:
    return torch.ops.pytorch_custom_op.myadd(a, b)


# 이게 없으면 이 operation을 포함한 모델을 torch.compile을 할 때 에러 발생
@torch.library.register_fake("pytorch_custom_op::myadd")
def _(a, b):
    torch._check(a.device == torch.device("cuda"))
    torch._check(b.device == torch.device("cuda"))
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.shape == b.shape)

    return torch.empty_like(a)