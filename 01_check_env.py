"""检查 PyTorch 是否可以执行一次最基本的张量计算。"""

import torch


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("PyTorch:", torch.__version__)
    print("使用设备:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    a = torch.randn(256, 256, device=device)
    b = torch.randn(256, 256, device=device)
    result = a @ b

    print("矩阵乘法成功，结果形状:", result.shape)


if __name__ == "__main__":
    main()
