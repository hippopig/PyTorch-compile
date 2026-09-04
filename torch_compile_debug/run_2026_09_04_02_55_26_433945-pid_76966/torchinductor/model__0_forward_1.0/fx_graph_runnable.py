
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_binly/tmpsefmm0hg'
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_binly/tmpsefmm0hg/triton'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.11.0+cu130
# torch cuda version: 13.0
# torch git version: 70d99e998b4955e0049d13a98d77ae1b14db1f45


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2025 NVIDIA Corporation 
# Built on Tue_Dec_16_07:23:41_PM_PST_2025 
# Cuda compilation tools, release 13.1, V13.1.115 
# Build cuda_13.1.r13.1/compiler.37061995_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 5060 Laptop GPU : 1 

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30):
        iota = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        embedding = torch.ops.aten.embedding.default(primals_2, primals_1);  primals_2 = None
        embedding_1 = torch.ops.aten.embedding.default(primals_3, iota);  primals_3 = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(add, getitem_1)
        mul = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_4);  mul = None
        add_2 = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None
        permute = torch.ops.aten.permute.default(add_2, [1, 0, 2]);  add_2 = None
        permute_1 = torch.ops.aten.permute.default(primals_7, [1, 0])
        clone = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view = torch.ops.aten.view.default(clone, [128, 128]);  clone = None
        mm = torch.ops.aten.mm.default(view, permute_1);  permute_1 = None
        view_1 = torch.ops.aten.view.default(mm, [32, 4, 384]);  mm = None
        add_3 = torch.ops.aten.add.Tensor(view_1, primals_6);  view_1 = primals_6 = None
        view_2 = torch.ops.aten.view.default(add_3, [32, 4, 3, 128]);  add_3 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(view_2, 0);  view_2 = None
        permute_2 = torch.ops.aten.permute.default(unsqueeze_2, [3, 1, 2, 0, 4]);  unsqueeze_2 = None
        squeeze = torch.ops.aten.squeeze.dim(permute_2, -2);  permute_2 = None
        clone_1 = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select = torch.ops.aten.select.int(clone_1, 0, 0)
        select_1 = torch.ops.aten.select.int(clone_1, 0, 1)
        select_2 = torch.ops.aten.select.int(clone_1, 0, 2);  clone_1 = None
        view_3 = torch.ops.aten.view.default(select, [32, 16, 32]);  select = None
        permute_3 = torch.ops.aten.permute.default(view_3, [1, 0, 2]);  view_3 = None
        view_4 = torch.ops.aten.view.default(select_1, [32, 16, 32]);  select_1 = None
        permute_4 = torch.ops.aten.permute.default(view_4, [1, 0, 2]);  view_4 = None
        view_5 = torch.ops.aten.view.default(select_2, [32, 16, 32]);  select_2 = None
        permute_5 = torch.ops.aten.permute.default(view_5, [1, 0, 2]);  view_5 = None
        view_6 = torch.ops.aten.view.default(permute_3, [4, 4, 32, 32]);  permute_3 = None
        view_7 = torch.ops.aten.view.default(permute_4, [4, 4, 32, 32]);  permute_4 = None
        view_8 = torch.ops.aten.view.default(permute_5, [4, 4, 32, 32]);  permute_5 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_6, view_7, view_8, None, True, 0.0, True)
        getitem_2 = _scaled_dot_product_efficient_attention[0]
        getitem_3 = _scaled_dot_product_efficient_attention[1]
        getitem_4 = _scaled_dot_product_efficient_attention[2]
        getitem_5 = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
        permute_6 = torch.ops.aten.permute.default(getitem_2, [2, 0, 1, 3])
        clone_2 = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_9 = torch.ops.aten.view.default(clone_2, [128, 128]);  clone_2 = None
        permute_7 = torch.ops.aten.permute.default(primals_8, [1, 0])
        addmm = torch.ops.aten.addmm.default(primals_9, view_9, permute_7);  primals_9 = permute_7 = None
        view_10 = torch.ops.aten.view.default(addmm, [32, 4, 128]);  addmm = None
        permute_8 = torch.ops.aten.permute.default(view_10, [1, 0, 2]);  view_10 = None
        add_4 = torch.ops.aten.add.Tensor(add, permute_8);  add = permute_8 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_1[0]
        getitem_7 = var_mean_1[1];  var_mean_1 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_4, getitem_7);  getitem_7 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_10)
        add_6 = torch.ops.aten.add.Tensor(mul_3, primals_11);  mul_3 = primals_11 = None
        view_11 = torch.ops.aten.view.default(add_6, [128, 128]);  add_6 = None
        permute_9 = torch.ops.aten.permute.default(primals_12, [1, 0])
        addmm_1 = torch.ops.aten.addmm.default(primals_13, view_11, permute_9);  primals_13 = permute_9 = None
        view_12 = torch.ops.aten.view.default(addmm_1, [4, 32, 256])
        mul_4 = torch.ops.aten.mul.Tensor(view_12, 0.5)
        mul_5 = torch.ops.aten.mul.Tensor(view_12, 0.7071067811865476);  view_12 = None
        erf = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_7 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
        view_13 = torch.ops.aten.view.default(mul_6, [128, 256]);  mul_6 = None
        permute_10 = torch.ops.aten.permute.default(primals_14, [1, 0])
        addmm_2 = torch.ops.aten.addmm.default(primals_15, view_13, permute_10);  primals_15 = permute_10 = None
        view_14 = torch.ops.aten.view.default(addmm_2, [4, 32, 128]);  addmm_2 = None
        add_8 = torch.ops.aten.add.Tensor(add_4, view_14);  add_4 = view_14 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_2[0]
        getitem_9 = var_mean_2[1];  var_mean_2 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_8, getitem_9);  getitem_9 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, primals_16)
        add_10 = torch.ops.aten.add.Tensor(mul_8, primals_17);  mul_8 = primals_17 = None
        permute_11 = torch.ops.aten.permute.default(add_10, [1, 0, 2]);  add_10 = None
        permute_12 = torch.ops.aten.permute.default(primals_19, [1, 0])
        clone_6 = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_15 = torch.ops.aten.view.default(clone_6, [128, 128]);  clone_6 = None
        mm_1 = torch.ops.aten.mm.default(view_15, permute_12);  permute_12 = None
        view_16 = torch.ops.aten.view.default(mm_1, [32, 4, 384]);  mm_1 = None
        add_11 = torch.ops.aten.add.Tensor(view_16, primals_18);  view_16 = primals_18 = None
        view_17 = torch.ops.aten.view.default(add_11, [32, 4, 3, 128]);  add_11 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(view_17, 0);  view_17 = None
        permute_13 = torch.ops.aten.permute.default(unsqueeze_3, [3, 1, 2, 0, 4]);  unsqueeze_3 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(permute_13, -2);  permute_13 = None
        clone_7 = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        select_3 = torch.ops.aten.select.int(clone_7, 0, 0)
        select_4 = torch.ops.aten.select.int(clone_7, 0, 1)
        select_5 = torch.ops.aten.select.int(clone_7, 0, 2);  clone_7 = None
        view_18 = torch.ops.aten.view.default(select_3, [32, 16, 32]);  select_3 = None
        permute_14 = torch.ops.aten.permute.default(view_18, [1, 0, 2]);  view_18 = None
        view_19 = torch.ops.aten.view.default(select_4, [32, 16, 32]);  select_4 = None
        permute_15 = torch.ops.aten.permute.default(view_19, [1, 0, 2]);  view_19 = None
        view_20 = torch.ops.aten.view.default(select_5, [32, 16, 32]);  select_5 = None
        permute_16 = torch.ops.aten.permute.default(view_20, [1, 0, 2]);  view_20 = None
        view_21 = torch.ops.aten.view.default(permute_14, [4, 4, 32, 32]);  permute_14 = None
        view_22 = torch.ops.aten.view.default(permute_15, [4, 4, 32, 32]);  permute_15 = None
        view_23 = torch.ops.aten.view.default(permute_16, [4, 4, 32, 32]);  permute_16 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_21, view_22, view_23, None, True, 0.0, True)
        getitem_10 = _scaled_dot_product_efficient_attention_1[0]
        getitem_11 = _scaled_dot_product_efficient_attention_1[1]
        getitem_12 = _scaled_dot_product_efficient_attention_1[2]
        getitem_13 = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
        permute_17 = torch.ops.aten.permute.default(getitem_10, [2, 0, 1, 3])
        clone_8 = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        view_24 = torch.ops.aten.view.default(clone_8, [128, 128]);  clone_8 = None
        permute_18 = torch.ops.aten.permute.default(primals_20, [1, 0])
        addmm_3 = torch.ops.aten.addmm.default(primals_21, view_24, permute_18);  primals_21 = permute_18 = None
        view_25 = torch.ops.aten.view.default(addmm_3, [32, 4, 128]);  addmm_3 = None
        permute_19 = torch.ops.aten.permute.default(view_25, [1, 0, 2]);  view_25 = None
        add_12 = torch.ops.aten.add.Tensor(add_8, permute_19);  add_8 = permute_19 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_3[0]
        getitem_15 = var_mean_3[1];  var_mean_3 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_12, getitem_15);  getitem_15 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, primals_22)
        add_14 = torch.ops.aten.add.Tensor(mul_10, primals_23);  mul_10 = primals_23 = None
        view_26 = torch.ops.aten.view.default(add_14, [128, 128]);  add_14 = None
        permute_20 = torch.ops.aten.permute.default(primals_24, [1, 0])
        addmm_4 = torch.ops.aten.addmm.default(primals_25, view_26, permute_20);  primals_25 = permute_20 = None
        view_27 = torch.ops.aten.view.default(addmm_4, [4, 32, 256])
        mul_11 = torch.ops.aten.mul.Tensor(view_27, 0.5)
        mul_12 = torch.ops.aten.mul.Tensor(view_27, 0.7071067811865476);  view_27 = None
        erf_1 = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_15 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_11, add_15);  mul_11 = add_15 = None
        view_28 = torch.ops.aten.view.default(mul_13, [128, 256]);  mul_13 = None
        permute_21 = torch.ops.aten.permute.default(primals_26, [1, 0])
        addmm_5 = torch.ops.aten.addmm.default(primals_27, view_28, permute_21);  primals_27 = permute_21 = None
        view_29 = torch.ops.aten.view.default(addmm_5, [4, 32, 128]);  addmm_5 = None
        add_16 = torch.ops.aten.add.Tensor(add_12, view_29);  add_12 = view_29 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_4[0]
        getitem_17 = var_mean_4[1];  var_mean_4 = None
        add_17 = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_16, getitem_17);  add_16 = getitem_17 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, primals_28)
        add_18 = torch.ops.aten.add.Tensor(mul_15, primals_29);  mul_15 = primals_29 = None
        permute_22 = torch.ops.aten.permute.default(primals_30, [1, 0])
        view_30 = torch.ops.aten.view.default(add_18, [128, 128]);  add_18 = None
        mm_2 = torch.ops.aten.mm.default(view_30, permute_22);  permute_22 = None
        view_31 = torch.ops.aten.view.default(mm_2, [4, 32, 256]);  mm_2 = None
        div = torch.ops.aten.div.Tensor(rsqrt_4, 128);  rsqrt_4 = None
        div_1 = torch.ops.aten.div.Tensor(rsqrt_3, 128);  rsqrt_3 = None
        div_2 = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
        div_3 = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
        return (view_31, primals_1, primals_4, primals_7, primals_8, primals_10, primals_12, primals_14, primals_16, primals_19, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, iota, embedding, embedding_1, getitem_1, rsqrt, view, view_6, view_7, view_8, getitem_2, getitem_3, getitem_4, getitem_5, view_9, mul_2, view_11, addmm_1, view_13, mul_7, view_15, view_21, view_22, view_23, getitem_10, getitem_11, getitem_12, getitem_13, view_24, mul_9, view_26, addmm_4, view_28, mul_14, view_30, div, div_1, div_2, div_3)

def load_args(reader):
    buf0 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (4, 32), dtype=torch.int64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf1, (256, 128), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128, 128), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128,), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf4, (128,), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf5, (384,), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf6, (384, 128), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf7, (128, 128), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf8, (128,), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf9, (128,), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf10, (128,), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (256, 128), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf12, (256,), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (128, 256), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf14, (128,), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf15, (128,), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf16, (128,), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf17, (384,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf18, (384, 128), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf19, (128, 128), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128,), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf21, (128,), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf22, (128,), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (256, 128), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf24, (256,), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf25, (128, 256), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf26, (128,), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf27, (128,), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf28, (128,), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf29, (256, 128), is_leaf=True)  # primals_30
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)