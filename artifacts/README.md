# 产物目录

简化版实验会生成两个主要文件：

- `fx_graph.py`：03 捕获到的 FX Python 代码；
- `tiny_transformer.onnx`：05 导出的 ONNX 模型。

02 生成的 Inductor IR 和 Triton 代码位于项目根目录的 `torch_compile_debug/`。
