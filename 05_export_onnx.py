"""把 Transformer 导出为 ONNX 文件，供 Netron 查看。"""

from pathlib import Path

import torch

from model import TinyTransformerLM, make_example_inputs


def main() -> None:
    # 导出模型结构不需要 GPU，使用 CPU 最简单。
    device = torch.device("cpu")
    model = TinyTransformerLM().to(device).eval()
    tokens, _ = make_example_inputs(device)
    output = Path(__file__).parent / "artifacts" / "tiny_transformer.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        tokens,
        output,
        input_names=["token_ids"],
        output_names=["logits"],
        opset_version=18,
        dynamo=False,
    )

    print("ONNX 模型已保存到:", output)
    print("可以把它拖入 https://netron.app/ 查看。")


if __name__ == "__main__":
    main()
