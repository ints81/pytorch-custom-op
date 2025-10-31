import pytorch_custom_op
import torch


def main():
    a = torch.randn(3, 3, device="cuda")
    b = torch.randn(3, 3, device="cuda")
    c = pytorch_custom_op.custom_ops.myadd(a, b)

    print(f"a: {a}")
    print(f"b: {b}")
    print(f"c: {c}")


if __name__ == "__main__":
    main()