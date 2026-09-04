"""查看 TorchDynamo 捕获到的 FX 计算图。"""

from pathlib import Path

import torch


OUTPUT = Path(__file__).parent / "artifacts" / "fx_graph.py"


def show_fx_graph(graph_module, _example_inputs):
    """这是一个最小 backend：打印计算图，然后直接执行它。"""
    print("\nDynamo 捕获到的 Python 代码:")
    print(graph_module.code)
    
    print("\nDynamo 捕获到Graph:")
    print(graph_module.graph)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(graph_module.code, encoding="utf-8")
    return graph_module.forward


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def simple_function(x):
        return torch.relu(x.sin() * 2)

    observed_function = torch.compile(
        simple_function,
        backend=show_fx_graph,
        fullgraph=True,
    )

    x = torch.tensor([-1.0, 0.0, 1.0, 2.0], device=device)
    output = observed_function(x)

    print("输入:", x)
    print("输出:", output)
    print("FX Python 代码已保存到:", OUTPUT)


if __name__ == "__main__":
    main()
