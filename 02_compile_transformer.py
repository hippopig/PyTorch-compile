"""分别用 eager 和 torch.compile 运行一次 Transformer 训练步骤。"""

import torch
import torch.nn.functional as F

from model import TinyTransformerLM, make_example_inputs


def training_step(model, tokens, targets):
    """前向计算 loss，再执行反向传播。"""
    # 清空上一次反向传播留下的梯度。PyTorch 默认会把新梯度累加到旧梯度上。
    model.zero_grad()

    # 调用 model(...) 会自动进入模型的 forward 方法。
    # logits 的形状是 [批量大小, 序列长度, 词表大小]，这里即 [4, 32, 256]。
    logits = model(tokens)

    # cross_entropy 希望预测值是 [样本数, 类别数]、答案是 [样本数]，
    # 所以把“批量”和“序列”两个维度合并：每个 token 位置都是一个分类样本。
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    # 从 loss 沿计算图反向求导，并把梯度保存到各模型参数的 .grad 中。
    # 此处没有 optimizer.step()，因此只计算梯度，并不会更新模型参数。
    loss.backward()

    # detach() 返回一个脱离计算图的 loss，后续打印它时不再需要保留反向传播信息。
    return loss.detach()


def main() -> None:
    # 固定随机种子，让模型参数初始化和示例输入可以重复得到相同结果。
    torch.manual_seed(2026)
    if not torch.cuda.is_available():
        raise RuntimeError("这个实验需要 CUDA，因为目标是生成 Triton GPU kernel。")

    # device 记录张量和模型要放置的位置；"cuda" 表示使用 NVIDIA GPU。
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    # 关闭 Inductor 缓存并打开调试输出，确保每次都生成一份可查看的新代码。
    torch._inductor.config.force_disable_caches = True
    torch._inductor.config.trace.enabled = True

    # .to(device) 把参数移到 GPU；.train() 切换为训练模式。
    model = TinyTransformerLM().to(device).train()
    # tokens 是输入 token 编号，targets 是每个位置要预测的下一个 token 编号。
    tokens, targets = make_example_inputs(device)

    # eager（即时执行）是 PyTorch 的普通模式：Python 遇到一个算子就执行一个算子。
    eager_loss = training_step(model, tokens, targets)
    print("eager loss:  ", eager_loss.item())

    # torch.compile 返回经过编译包装的模型，原来的 model 对象和参数仍会被复用。
    # fullgraph=True 要求 Dynamo 把整个 forward 捕获为一张图；遇到 graph break 就报错。
    compiled_model = torch.compile(
        model,
        backend="inductor",
        fullgraph=True,
    )

    # 第一次调用 compiled_model 时才真正开始编译：Dynamo 捕获前向计算图，
    # AOTAutograd 准备反向图，Inductor 再生成并编译当前 GPU 使用的代码。
    compiled_loss = training_step(compiled_model, tokens, targets)

    # CUDA 算子通常异步提交；synchronize() 等待 GPU 工作完成后再继续。
    torch.cuda.synchronize()
    print("compile loss:", compiled_loss.item())
    # 浮点计算可能因编译后的算子顺序不同而有微小误差，因此用容差比较。
    print("结果接近:", torch.allclose(eager_loss, compiled_loss, atol=1e-5))
    print("生成的 kernel 代码位于: torch_compile_debug/.../output_code.py")


if __name__ == "__main__":
    # 只有直接运行本文件时才执行 main；被其他 Python 文件 import 时不会执行。
    main()
