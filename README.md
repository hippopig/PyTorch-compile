# PyTorch 编译入门实验

这个项目用 5 个尽量短小的脚本介绍 PyTorch 2.x 编译链路。各脚本承担不同任务，
并不是每一个脚本都使用 Transformer，也不是每一个脚本都会生成 Triton kernel。

## 先看结论：每个脚本到底用了什么

| 脚本 | 计算对象 | Transformer | Dynamo / FX | AOTAutograd | Inductor | Triton GPU kernel | ONNX |
|---|---|---:|---:|---:|---:|---:|---:|
| `01_check_env.py` | 矩阵乘法 | 否 | 否 | 否 | 否 | 否 | 否 |
| `02_compile_transformer.py` | 小型语言模型的前向和反向 | 是 | 是 | 是 | 是 | 是 | 否 |
| `03_capture_fx.py` | `sin → 乘法 → relu` | 否 | 是 | 否 | 否，自定义观察 backend | 否 | 否 |
| `04_graph_breaks.py` | 几个简单 Tensor 函数 | 否 | 是 | 否 | 否，使用 eager backend | 否 | 否 |
| `05_export_onnx.py` | 小型语言模型的前向 | 是 | 否 | 否 | 否 | 否 | 是 |

设备选择也不完全相同：

- 02 必须使用 NVIDIA GPU，因为该实验的目标就是生成 Triton CUDA kernel；
- 05 固定在 CPU 上导出 ONNX，导出模型结构不需要 GPU；
- 01、03、04 优先使用 CUDA，没有 CUDA 时自动使用 CPU。

## 项目文件

```text
01_check_env.py             检查 PyTorch 和设备
02_compile_transformer.py   用 Inductor 编译 Transformer
03_capture_fx.py            观察 Dynamo 生成的 FX 图
04_graph_breaks.py          制造并修复 graph break
05_export_onnx.py           把 Transformer 导出为 ONNX
model.py                    TinyTransformerLM 模型定义
run_lab.sh                  按顺序运行 5 个实验
artifacts/                  FX 和 ONNX 等简单产物
torch_compile_debug/        02 生成的 Inductor IR 和 kernel 代码
```

## 环境与运行方法

本机使用名为 `cuda-dev` 的 Conda 环境。可以先进入环境：

```bash
conda activate cuda-dev
python 01_check_env.py
```

也可以不激活环境，直接运行：

```bash
/home/binly/miniconda3/bin/conda run --no-capture-output -n cuda-dev \
  python 01_check_env.py
```

一键运行全部实验：

```bash
bash run_lab.sh
```

02 会强制从头编译 Transformer，在当前机器上可能需要数分钟。第一次学习时建议
依次打开源码并单独运行，而不是直接运行全部脚本。

## 模型：TinyTransformerLM

02 和 05 共用 `model.py` 中的 `TinyTransformerLM`。默认模型包含：

- 词表大小：256；
- hidden size：128；
- 注意力头数：4；
- Transformer 层数：2；
- FFN hidden size：256；
- 默认输入形状：`[batch=4, sequence=32]`；
- 输出 logits 形状：`[4, 32, 256]`。

模型的数据流是：

```text
token IDs
   │
   ├─ token embedding
   └─ position embedding
            │
            ▼
       两者相加
            │
            ▼
2 层 TransformerEncoder + causal mask
            │
            ▼
       LayerNorm
            │
            ▼
   Linear → vocabulary logits
```

虽然代码使用的是 `nn.TransformerEncoder`，但它传入了 causal mask，当前位置不能
看到未来 token。因此在行为上它是一个 decoder-style causal language model；它不是
包含 encoder 和 decoder 两套结构的完整 `nn.Transformer`。

训练 target 由输入 token 向左滚动一位得到，用来演示 next-token prediction。这个
模型只服务于编译实验，没有优化器、数据集或完整训练循环。

## 01：确认 PyTorch 环境

```bash
python 01_check_env.py
```

脚本完成三件事：

1. 检测 CUDA 是否可用；
2. 在选定设备上创建两个 `256 × 256` Tensor；
3. 执行矩阵乘法并打印结果形状。

这一阶段只是确认 PyTorch 能执行计算，没有调用 `torch.compile`。矩阵乘法成功也不
代表 Triton 已经参与运行；CUDA 上的普通 eager 矩阵乘法通常由 PyTorch 调用已有的
高性能库完成。

## 02：真正进行后端编译并生成 Triton

```bash
python 02_compile_transformer.py
```

这是项目中唯一真正调用 Inductor 后端并生成 Triton GPU kernel 的脚本：

```python
compiled_model = torch.compile(
    model,
    backend="inductor",
    fullgraph=True,
)
```

这里的主要过程是：

```text
Python model.forward
        │
        ▼
TorchDynamo 读取 Python bytecode，捕获 FX 图并建立 guards
        │
        ▼
AOTAutograd 为编译区域准备前向图和反向图
        │
        ▼
TorchInductor 做 lowering、调度和算子融合
        │
        ├─ 融合的 pointwise/reduction 等 → Triton CUDA kernel
        └─ GEMM 等算子                  → cuBLAS 等外部 kernel
        │
        ▼
在 CUDA stream 上启动 kernel
```

### 编译边界

当前代码把 `model` 传给了 `torch.compile`，所以主要编译区域是
`TinyTransformerLM.forward`。`cross_entropy` 在 `compiled_model(tokens)` 返回后才
计算，因此 loss 本身不在这个 forward FX 图中；调用 `loss.backward()` 时，梯度仍会
进入编译模型对应的反向实现。

如果希望将 model、loss 和更多训练逻辑捕获为更大的区域，可以以后把整个
`training_step` 作为进阶实验进行编译。本项目当前保持较简单的模型编译边界。

### 为什么关闭缓存

脚本设置了：

```python
torch._inductor.config.force_disable_caches = True
torch._inductor.config.trace.enabled = True
```

正常项目应该利用编译缓存来减少后续启动时间。这里关闭缓存纯粹是为了教学：每次
运行都真正重新编译，并生成一份可以检查的新调试目录。这两个 `_inductor` 配置属于
内部调试接口，不建议直接复制到生产代码。

### 在哪里看真实 Triton kernel

每次运行会生成类似目录：

```text
torch_compile_debug/run_.../torchinductor/
├── model__0_forward_1.0/
│   ├── fx_graph_readable.py
│   ├── ir_pre_fusion.txt
│   ├── ir_post_fusion.txt
│   └── output_code.py
└── model__0_backward_3.1/
    ├── fx_graph_readable.py
    ├── ir_pre_fusion.txt
    ├── ir_post_fusion.txt
    └── output_code.py
```

搜索真正的 Triton 函数：

```bash
rg -n "@triton.jit" torch_compile_debug
```

搜索实际的 kernel launch：

```bash
rg -n "triton_.*\.run\(" torch_compile_debug
```

判断它确实是 GPU kernel，可以在 `output_code.py` 中寻找以下三类证据：

```python
@triton.jit                       # Triton kernel 定义
device_str='cuda'                 # 编译目标是 CUDA
triton_some_kernel.run(..., stream=stream0)  # kernel launch
```

当前 RTX 5060 实际运行生成了 13 个前向和 17 个反向 `@triton.jit` 定义。数量不是
固定规范；PyTorch/Triton 版本、GPU、输入 shape 和融合决策改变后，数量都可能变化。

并非所有 Transformer 运算都会变成 Triton：

- LayerNorm、GELU、embedding 周边及张量变换等操作经常被融合成 Triton；
- `mm`、`addmm` 等矩阵乘法可能显示为 `extern_kernels.mm/addmm`，通常交给 cuBLAS；
- attention 也可能选择 PyTorch 已有的高效 CUDA 实现。

脚本比较 eager loss 和 compiled loss，只用于检查两条路径的数值是否接近，不是严谨
的性能基准。第一次调用含编译时间，不能直接拿它和 eager 执行时间比较。

## 03：只观察 Dynamo/FX，不做 Inductor 编译

```bash
python 03_capture_fx.py
```

03 故意不使用 Transformer，而使用一个容易读懂的小函数：

```python
return torch.relu(x.sin() * 2)
```

它把 `show_fx_graph` 作为自定义 backend 传给 `torch.compile`。Dynamo 会把 Python
函数转换成 FX `GraphModule`，然后调用这个 backend。backend 打印并保存
`graph_module.code`，最后直接返回 `graph_module.forward` 执行。

因此 03 的边界是：

```text
Python → Dynamo → FX GraphModule → 直接执行 FX forward
```

它不会进入 AOTAutograd 或 Inductor，也不会生成 Triton。输出文件是：

```text
artifacts/fx_graph.py
```

这个实验回答的是“Dynamo 捕获了什么”，不是“GPU kernel 长什么样”。

## 04：学习 graph break

```bash
python 04_graph_breaks.py
```

04 不使用 Transformer，只用简单 Tensor 函数演示五种断图原因：

| 原因 | 问题写法 | 修复思路 |
|---|---|---|
| 数据相关 Python 控制流 | `if tensor.item() > 0` | 用 `torch.where` 保持 Tensor 语义 |
| Python 副作用 | 在编译函数内 `print(tensor)` | 把打印、日志和 I/O 移到编译函数外 |
| 主动断图 | `torch._dynamo.graph_break()` | 不需要边界时删除它 |
| 动态 Python 对象操作 | 对中间 Tensor 调用 `id()` | 删除与 Tensor 计算无关的对象操作 |
| 无法追踪或禁止追踪的代码 | `@torch.compiler.disable` 包裹的函数 | 改写为可追踪代码；第三方扩展可注册 custom operator |

脚本先使用 `torch._dynamo.explain` 显示捕获图数量和原因，再用：

```python
torch.compile(function, backend="eager", fullgraph=True)
```

这里的 `backend="eager"` 容易被误解：Dynamo 仍然会捕获 FX 图，只是 backend 不会
继续调用 Inductor，而是直接执行捕获到的图。这样可以更快地单独排查 Dynamo 问题，
所以 04 不会生成 Triton。

`fullgraph=True` 要求编译区域形成单张 FX 图，遇到 graph break 就报错；默认的
`fullgraph=False` 则允许结束当前图、在 Python 中执行不支持部分，然后继续捕获。
因此“函数得到了正确结果”并不能证明它没有发生 graph break。

`torch.where` 也不是所有 Python `if` 的机械替代品：它通常会构造两个候选结果，
所以两个分支都必须可以安全计算。依赖 shape 等 Tensor 元数据的 Python 分支也不一定
断图，Dynamo 有时可以通过 guard 对它进行特化。

## 05：导出 Transformer ONNX

```bash
python 05_export_onnx.py
```

05 再次使用 `TinyTransformerLM`，但它做的是模型格式导出：

```text
PyTorch Transformer → ONNX graph → tiny_transformer.onnx
```

它不调用 `torch.compile`，所以没有 Dynamo、AOTAutograd、Inductor 或 Triton。ONNX
描述的是可移植的模型计算图，可供 Netron 查看或交给 ONNX Runtime 等工具；它不是
Dynamo FX 图，也不是 GPU kernel 源代码。

当前脚本使用 CPU 和固定示例输入，并显式设置 `dynamo=False` 使用 legacy exporter，
以避免增加 `onnxscript` 依赖。因为没有声明动态 shape，导出的 batch 和 sequence
维度固定为示例值 `4 × 32`。生成文件是：

```text
artifacts/tiny_transformer.onnx
```

可以把文件拖入 <https://netron.app/>，查看 embedding、LayerNorm、Linear，以及
attention 被展开后对应的低层计算。

## 三种“图”不要混淆

| 图或代码 | 在哪里产生 | 用途 |
|---|---|---|
| Dynamo FX 图 | 03，或 02 的编译前端 | 表达 PyTorch 运算和数据依赖 |
| ONNX 图 | 05 | 跨框架保存和部署模型结构 |
| Triton kernel | 02 的 Inductor GPU 后端 | 真正在 GPU 上执行融合计算 |

它们处于不同抽象层级，不能因为都能“可视化”或都叫 graph 就认为它们等价。

## 推荐学习顺序

1. 运行 01，确认 PyTorch 能看到 GPU；
2. 阅读 `model.py`，理解输入和输出 shape；
3. 运行 03，用简单函数理解 Dynamo 和 FX；
4. 运行 04，理解为什么一段 Python 可能无法形成完整 FX 图；
5. 运行 02，沿着 FX、Inductor IR、Triton kernel 逐层查看；
6. 运行 05，在 Netron 中对照 Transformer 的高层模型结构。

## 常见问题

### 为什么 02 很慢？

它明确关闭了缓存，而且同时产生编译区域的前向和反向实现。运行结束后的热执行才会
复用这次进程中的编译结果。若不需要每次观察新代码，可以注释掉
`force_disable_caches` 和 `trace.enabled` 两行。

### 为什么生成代码里看不到一个名为 Transformer 的 kernel？

Transformer 是高层模块。编译器会将它分解为 embedding、矩阵乘法、归一化、激活、
张量变换等低层运算，再融合其中适合融合的部分。因此生成文件中会看到
`triton_poi_*`、`triton_per_*`、`triton_red_*` 和 `extern_kernels.mm` 等名字。

### 为什么 04 的 disabled 案例可能显示 `graph_break_count=0`？

被禁用函数可能完全在捕获图之外执行，此时图之间的切分计数不一定直观，但
`break_reasons` 仍会记录跳过原因。判断断图时不要只看一个数字。

### 为什么 05 会出现 tracer 或 legacy exporter 警告？

当前脚本为了减少依赖，选择了 legacy ONNX exporter；同时模型中包含基于输入 shape
的 Python 检查。对于这个固定 shape 的教学导出，警告不妨碍文件生成，但它提醒你该
ONNX 图不会自动泛化到任意输入 shape。

## 官方延伸阅读

- [torch.compile 教程](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [torch.fx 文档](https://docs.pytorch.org/docs/stable/fx.html)
- [使用 fullgraph=True 定位 graph break](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html)
- [使用 fullgraph=False 与 eager backend 排查断图](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_false.html)
- [torch.compile 编译缓存](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
- [在 torch.compile 中使用自定义 Triton kernel](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)
- [PyTorch ONNX 文档](https://docs.pytorch.org/docs/stable/onnx.html)
