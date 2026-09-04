"""演示五种 graph break，以及对应的简单修复。"""

import torch


# 1. Tensor 数值进入 Python，并决定 if 分支。
def data_dependent_branch(x):
    y = x.sin()
    if y.sum().item() > 0:
        return y.cos()
    return y.tanh()


def fixed_data_dependent_branch(x):
    y = x.sin()
    return torch.where(y.sum() > 0, y.cos(), y.tanh())


# 2. print、日志和 I/O 等 Python 副作用不属于纯 Tensor 计算图。
def python_side_effect(x):
    y = x.sin()
    print("编译区域内的 Tensor 均值:", y.mean())
    return y.cos()


def fixed_python_side_effect(x):
    # 需要打印时，应在编译函数返回之后、调用方的 Python 代码中完成。
    return x.sin().cos()


# 3. 程序员主动要求在这里断图。
def explicit_break(x):
    y = x.sin()
    torch._dynamo.graph_break()
    return y.cos()


def fixed_explicit_break(x):
    return x.sin().cos()


# 4. id 等动态 Python 对象操作无法表示为 Tensor 图节点。
def dynamic_object_operation(x):
    y = x.sin()
    if id(y) != 0:
        return y.cos()
    return y


def fixed_dynamic_object_operation(x):
    return x.sin().cos()


# 5. 被 disable 的函数不会进入计算图；不透明的第三方扩展也常需要这样隔离。
@torch.compiler.disable
def disabled_helper(x):
    return x.cos()


def call_disabled_helper(x):
    y = disabled_helper(x.sin())
    # disabled_helper 之后再做一次 Tensor 运算，让 Dynamo 恢复捕获第二张图。
    return y.clone()


def fixed_disabled_helper(x):
    # 与上面的计算保持一致，但所有 Tensor 运算都可以进入同一张图。
    return x.sin().cos().clone()


def run_case(name, broken_function, fixed_function, x):
    print(f"\n=== {name} ===")

    explanation = torch._dynamo.explain(broken_function)(x)
    print("捕获的图数量:", explanation.graph_count)
    print("graph break 数量:", explanation.graph_break_count)
    for reason in explanation.break_reasons:
        print("原因:", reason.reason.splitlines()[0])

    # fullgraph=True 不允许悄悄退回 eager；遇到 graph break 就会报错。
    try:
        torch.compile(
            broken_function,
            backend="eager",
            fullgraph=True,
        )(x)
    except Exception as error:
        print("严格模式确认发生断图:", type(error).__name__)

    torch._dynamo.reset()
    result = torch.compile(
        fixed_function,
        backend="eager",
        fullgraph=True,
    )(x)
    print("修复版本成功，结果均值:", result.mean().item())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.linspace(-0.25, 1.0, steps=128, device=device)

    cases = [
        ("数据相关的 Python if", data_dependent_branch, fixed_data_dependent_branch),
        ("编译区域内的 Python 副作用", python_side_effect, fixed_python_side_effect),
        ("显式 graph_break", explicit_break, fixed_explicit_break),
        ("动态 Python 对象操作", dynamic_object_operation, fixed_dynamic_object_operation),
        ("调用 disabled 函数", call_disabled_helper, fixed_disabled_helper),
    ]

    for case in cases:
        torch._dynamo.reset()
        run_case(*case, x)


if __name__ == "__main__":
    main()
