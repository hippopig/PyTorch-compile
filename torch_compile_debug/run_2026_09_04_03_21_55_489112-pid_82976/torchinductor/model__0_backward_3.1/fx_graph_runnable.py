
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_binly/tmpwqa2mwzg'
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_binly/tmpwqa2mwzg/triton'

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

torch._inductor.config.deterministic = False
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.test_configs.runtime_triton_dtype_assert = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = False
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



    def forward(self, primals_1, primals_4, primals_7, primals_8, primals_10, primals_12, primals_14, primals_16, primals_19, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, iota, embedding, embedding_1, getitem_1, rsqrt, view, view_6, view_7, view_8, getitem_2, getitem_3, getitem_4, getitem_5, view_9, mul_2, view_11, addmm_1, view_13, mul_7, view_15, view_21, view_22, view_23, getitem_10, getitem_11, getitem_12, getitem_13, view_24, mul_9, view_26, addmm_4, view_28, mul_14, view_30, div, div_1, div_2, div_3, tangents_1):
        view_32 = torch.ops.aten.view.default(tangents_1, [128, 256]);  tangents_1 = None
        permute_23 = torch.ops.aten.permute.default(view_32, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_23, view_30);  permute_23 = view_30 = None
        permute_22 = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        permute_25 = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_4 = torch.ops.aten.mm.default(view_32, permute_25);  view_32 = permute_25 = None
        view_33 = torch.ops.aten.view.default(mm_4, [4, 32, 128]);  mm_4 = None
        mul_17 = torch.ops.aten.mul.Tensor(view_33, primals_28);  primals_28 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, 128)
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_17, [2], True)
        mul_19 = torch.ops.aten.mul.Tensor(mul_17, mul_14);  mul_17 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_19, [2], True);  mul_19 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_14, sum_2);  sum_2 = None
        sub_7 = torch.ops.aten.sub.Tensor(mul_18, sum_1);  mul_18 = sum_1 = None
        sub_8 = torch.ops.aten.sub.Tensor(sub_7, mul_20);  sub_7 = mul_20 = None
        mul_21 = torch.ops.aten.mul.Tensor(div, sub_8);  div = sub_8 = None
        mul_22 = torch.ops.aten.mul.Tensor(view_33, mul_14);  mul_14 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_22, [0, 1]);  mul_22 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(view_33, [0, 1]);  view_33 = None
        view_34 = torch.ops.aten.view.default(mul_21, [128, 128])
        permute_21 = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        permute_27 = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_5 = torch.ops.aten.mm.default(view_34, permute_27);  permute_27 = None
        permute_28 = torch.ops.aten.permute.default(view_34, [1, 0])
        mm_6 = torch.ops.aten.mm.default(permute_28, view_28);  permute_28 = view_28 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(view_34, [0], True);  view_34 = None
        view_35 = torch.ops.aten.view.default(sum_5, [128]);  sum_5 = None
        view_36 = torch.ops.aten.view.default(mm_5, [4, 32, 256]);  mm_5 = None
        view_27 = torch.ops.aten.view.default(addmm_4, [4, 32, 256]);  addmm_4 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_27, 0.7071067811865476)
        erf_1 = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_15 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_24 = torch.ops.aten.mul.Tensor(add_15, 0.5);  add_15 = None
        mul_25 = torch.ops.aten.mul.Tensor(view_27, view_27)
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, -0.5);  mul_25 = None
        exp = torch.ops.aten.exp.default(mul_26);  mul_26 = None
        mul_27 = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_28 = torch.ops.aten.mul.Tensor(view_27, mul_27);  view_27 = mul_27 = None
        add_20 = torch.ops.aten.add.Tensor(mul_24, mul_28);  mul_24 = mul_28 = None
        mul_29 = torch.ops.aten.mul.Tensor(view_36, add_20);  view_36 = add_20 = None
        view_37 = torch.ops.aten.view.default(mul_29, [128, 256]);  mul_29 = None
        permute_20 = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_31 = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_7 = torch.ops.aten.mm.default(view_37, permute_31);  permute_31 = None
        permute_32 = torch.ops.aten.permute.default(view_37, [1, 0])
        mm_8 = torch.ops.aten.mm.default(permute_32, view_26);  permute_32 = view_26 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(view_37, [0], True);  view_37 = None
        view_38 = torch.ops.aten.view.default(sum_6, [256]);  sum_6 = None
        view_39 = torch.ops.aten.view.default(mm_7, [4, 32, 128]);  mm_7 = None
        mul_31 = torch.ops.aten.mul.Tensor(view_39, primals_22);  primals_22 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, 128)
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_31, [2], True)
        mul_33 = torch.ops.aten.mul.Tensor(mul_31, mul_9);  mul_31 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_33, [2], True);  mul_33 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_9, sum_8);  sum_8 = None
        sub_10 = torch.ops.aten.sub.Tensor(mul_32, sum_7);  mul_32 = sum_7 = None
        sub_11 = torch.ops.aten.sub.Tensor(sub_10, mul_34);  sub_10 = mul_34 = None
        mul_35 = torch.ops.aten.mul.Tensor(div_1, sub_11);  div_1 = sub_11 = None
        mul_36 = torch.ops.aten.mul.Tensor(view_39, mul_9);  mul_9 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_36, [0, 1]);  mul_36 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(view_39, [0, 1]);  view_39 = None
        add_21 = torch.ops.aten.add.Tensor(mul_21, mul_35);  mul_21 = mul_35 = None
        permute_35 = torch.ops.aten.permute.default(add_21, [1, 0, 2])
        clone_12 = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_40 = torch.ops.aten.view.default(clone_12, [128, 128]);  clone_12 = None
        permute_18 = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_36 = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        mm_9 = torch.ops.aten.mm.default(view_40, permute_36);  permute_36 = None
        permute_37 = torch.ops.aten.permute.default(view_40, [1, 0])
        mm_10 = torch.ops.aten.mm.default(permute_37, view_24);  permute_37 = view_24 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
        view_41 = torch.ops.aten.view.default(sum_11, [128]);  sum_11 = None
        view_42 = torch.ops.aten.view.default(mm_9, [32, 4, 4, 32]);  mm_9 = None
        permute_40 = torch.ops.aten.permute.default(view_42, [1, 2, 0, 3]);  view_42 = None
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_40, view_21, view_22, view_23, None, getitem_10, getitem_11, getitem_12, getitem_13, 0.0, [True, True, True, False], True);  permute_40 = view_21 = view_22 = view_23 = getitem_10 = getitem_11 = getitem_12 = getitem_13 = None
        getitem_18 = _scaled_dot_product_efficient_attention_backward[0]
        getitem_19 = _scaled_dot_product_efficient_attention_backward[1]
        getitem_20 = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None
        clone_13 = torch.ops.aten.clone.default(getitem_20, memory_format = torch.contiguous_format);  getitem_20 = None
        view_43 = torch.ops.aten.view.default(clone_13, [16, 32, 32]);  clone_13 = None
        clone_14 = torch.ops.aten.clone.default(getitem_19, memory_format = torch.contiguous_format);  getitem_19 = None
        view_44 = torch.ops.aten.view.default(clone_14, [16, 32, 32]);  clone_14 = None
        clone_15 = torch.ops.aten.clone.default(getitem_18, memory_format = torch.contiguous_format);  getitem_18 = None
        view_45 = torch.ops.aten.view.default(clone_15, [16, 32, 32]);  clone_15 = None
        permute_41 = torch.ops.aten.permute.default(view_43, [1, 0, 2]);  view_43 = None
        clone_16 = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        view_46 = torch.ops.aten.view.default(clone_16, [32, 4, 128]);  clone_16 = None
        permute_42 = torch.ops.aten.permute.default(view_44, [1, 0, 2]);  view_44 = None
        clone_17 = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
        view_47 = torch.ops.aten.view.default(clone_17, [32, 4, 128]);  clone_17 = None
        permute_43 = torch.ops.aten.permute.default(view_45, [1, 0, 2]);  view_45 = None
        clone_18 = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        view_48 = torch.ops.aten.view.default(clone_18, [32, 4, 128]);  clone_18 = None
        full_default = torch.ops.aten.full.default([3, 32, 4, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter = torch.ops.aten.select_scatter.default(full_default, view_46, 0, 2);  view_46 = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(full_default, view_47, 0, 1);  view_47 = None
        add_22 = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        select_scatter_2 = torch.ops.aten.select_scatter.default(full_default, view_48, 0, 0);  view_48 = None
        add_23 = torch.ops.aten.add.Tensor(add_22, select_scatter_2);  add_22 = select_scatter_2 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(add_23, 3);  add_23 = None
        permute_44 = torch.ops.aten.permute.default(unsqueeze_4, [3, 1, 2, 0, 4]);  unsqueeze_4 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(permute_44, 0);  permute_44 = None
        clone_19 = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        view_49 = torch.ops.aten.view.default(clone_19, [32, 4, 384]);  clone_19 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(view_49, [0, 1], True)
        view_50 = torch.ops.aten.view.default(sum_12, [384]);  sum_12 = None
        view_51 = torch.ops.aten.view.default(view_49, [128, 384]);  view_49 = None
        permute_45 = torch.ops.aten.permute.default(view_51, [1, 0])
        mm_11 = torch.ops.aten.mm.default(permute_45, view_15);  permute_45 = view_15 = None
        permute_12 = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_47 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_12 = torch.ops.aten.mm.default(view_51, permute_47);  view_51 = permute_47 = None
        view_52 = torch.ops.aten.view.default(mm_12, [32, 4, 128]);  mm_12 = None
        permute_49 = torch.ops.aten.permute.default(view_52, [1, 0, 2]);  view_52 = None
        mul_38 = torch.ops.aten.mul.Tensor(permute_49, primals_16);  primals_16 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, 128)
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_38, [2], True)
        mul_40 = torch.ops.aten.mul.Tensor(mul_38, mul_7);  mul_38 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(mul_40, [2], True);  mul_40 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_7, sum_14);  sum_14 = None
        sub_13 = torch.ops.aten.sub.Tensor(mul_39, sum_13);  mul_39 = sum_13 = None
        sub_14 = torch.ops.aten.sub.Tensor(sub_13, mul_41);  sub_13 = mul_41 = None
        mul_42 = torch.ops.aten.mul.Tensor(div_2, sub_14);  div_2 = sub_14 = None
        mul_43 = torch.ops.aten.mul.Tensor(permute_49, mul_7);  mul_7 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_43, [0, 1]);  mul_43 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(permute_49, [0, 1]);  permute_49 = None
        add_24 = torch.ops.aten.add.Tensor(add_21, mul_42);  add_21 = mul_42 = None
        view_53 = torch.ops.aten.view.default(add_24, [128, 128])
        permute_10 = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_50 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_13 = torch.ops.aten.mm.default(view_53, permute_50);  permute_50 = None
        permute_51 = torch.ops.aten.permute.default(view_53, [1, 0])
        mm_14 = torch.ops.aten.mm.default(permute_51, view_13);  permute_51 = view_13 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(view_53, [0], True);  view_53 = None
        view_54 = torch.ops.aten.view.default(sum_17, [128]);  sum_17 = None
        view_55 = torch.ops.aten.view.default(mm_13, [4, 32, 256]);  mm_13 = None
        view_12 = torch.ops.aten.view.default(addmm_1, [4, 32, 256]);  addmm_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(view_12, 0.7071067811865476)
        erf = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_7 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_45 = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
        mul_46 = torch.ops.aten.mul.Tensor(view_12, view_12)
        mul_47 = torch.ops.aten.mul.Tensor(mul_46, -0.5);  mul_46 = None
        exp_1 = torch.ops.aten.exp.default(mul_47);  mul_47 = None
        mul_48 = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_49 = torch.ops.aten.mul.Tensor(view_12, mul_48);  view_12 = mul_48 = None
        add_26 = torch.ops.aten.add.Tensor(mul_45, mul_49);  mul_45 = mul_49 = None
        mul_50 = torch.ops.aten.mul.Tensor(view_55, add_26);  view_55 = add_26 = None
        view_56 = torch.ops.aten.view.default(mul_50, [128, 256]);  mul_50 = None
        permute_9 = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_54 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_15 = torch.ops.aten.mm.default(view_56, permute_54);  permute_54 = None
        permute_55 = torch.ops.aten.permute.default(view_56, [1, 0])
        mm_16 = torch.ops.aten.mm.default(permute_55, view_11);  permute_55 = view_11 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(view_56, [0], True);  view_56 = None
        view_57 = torch.ops.aten.view.default(sum_18, [256]);  sum_18 = None
        view_58 = torch.ops.aten.view.default(mm_15, [4, 32, 128]);  mm_15 = None
        mul_52 = torch.ops.aten.mul.Tensor(view_58, primals_10);  primals_10 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, 128)
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_52, [2], True)
        mul_54 = torch.ops.aten.mul.Tensor(mul_52, mul_2);  mul_52 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(mul_54, [2], True);  mul_54 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_2, sum_20);  sum_20 = None
        sub_16 = torch.ops.aten.sub.Tensor(mul_53, sum_19);  mul_53 = sum_19 = None
        sub_17 = torch.ops.aten.sub.Tensor(sub_16, mul_55);  sub_16 = mul_55 = None
        mul_56 = torch.ops.aten.mul.Tensor(div_3, sub_17);  div_3 = sub_17 = None
        mul_57 = torch.ops.aten.mul.Tensor(view_58, mul_2);  mul_2 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(mul_57, [0, 1]);  mul_57 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(view_58, [0, 1]);  view_58 = None
        add_27 = torch.ops.aten.add.Tensor(add_24, mul_56);  add_24 = mul_56 = None
        permute_58 = torch.ops.aten.permute.default(add_27, [1, 0, 2])
        clone_20 = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
        view_59 = torch.ops.aten.view.default(clone_20, [128, 128]);  clone_20 = None
        permute_7 = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_59 = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_17 = torch.ops.aten.mm.default(view_59, permute_59);  permute_59 = None
        permute_60 = torch.ops.aten.permute.default(view_59, [1, 0])
        mm_18 = torch.ops.aten.mm.default(permute_60, view_9);  permute_60 = view_9 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(view_59, [0], True);  view_59 = None
        view_60 = torch.ops.aten.view.default(sum_23, [128]);  sum_23 = None
        view_61 = torch.ops.aten.view.default(mm_17, [32, 4, 4, 32]);  mm_17 = None
        permute_63 = torch.ops.aten.permute.default(view_61, [1, 2, 0, 3]);  view_61 = None
        _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_63, view_6, view_7, view_8, None, getitem_2, getitem_3, getitem_4, getitem_5, 0.0, [True, True, True, False], True);  permute_63 = view_6 = view_7 = view_8 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = None
        getitem_22 = _scaled_dot_product_efficient_attention_backward_1[0]
        getitem_23 = _scaled_dot_product_efficient_attention_backward_1[1]
        getitem_24 = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None
        clone_21 = torch.ops.aten.clone.default(getitem_24, memory_format = torch.contiguous_format);  getitem_24 = None
        view_62 = torch.ops.aten.view.default(clone_21, [16, 32, 32]);  clone_21 = None
        clone_22 = torch.ops.aten.clone.default(getitem_23, memory_format = torch.contiguous_format);  getitem_23 = None
        view_63 = torch.ops.aten.view.default(clone_22, [16, 32, 32]);  clone_22 = None
        clone_23 = torch.ops.aten.clone.default(getitem_22, memory_format = torch.contiguous_format);  getitem_22 = None
        view_64 = torch.ops.aten.view.default(clone_23, [16, 32, 32]);  clone_23 = None
        permute_64 = torch.ops.aten.permute.default(view_62, [1, 0, 2]);  view_62 = None
        clone_24 = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        view_65 = torch.ops.aten.view.default(clone_24, [32, 4, 128]);  clone_24 = None
        permute_65 = torch.ops.aten.permute.default(view_63, [1, 0, 2]);  view_63 = None
        clone_25 = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        view_66 = torch.ops.aten.view.default(clone_25, [32, 4, 128]);  clone_25 = None
        permute_66 = torch.ops.aten.permute.default(view_64, [1, 0, 2]);  view_64 = None
        clone_26 = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
        view_67 = torch.ops.aten.view.default(clone_26, [32, 4, 128]);  clone_26 = None
        select_scatter_3 = torch.ops.aten.select_scatter.default(full_default, view_65, 0, 2);  view_65 = None
        select_scatter_4 = torch.ops.aten.select_scatter.default(full_default, view_66, 0, 1);  view_66 = None
        add_28 = torch.ops.aten.add.Tensor(select_scatter_3, select_scatter_4);  select_scatter_3 = select_scatter_4 = None
        select_scatter_5 = torch.ops.aten.select_scatter.default(full_default, view_67, 0, 0);  full_default = view_67 = None
        add_29 = torch.ops.aten.add.Tensor(add_28, select_scatter_5);  add_28 = select_scatter_5 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(add_29, 3);  add_29 = None
        permute_67 = torch.ops.aten.permute.default(unsqueeze_5, [3, 1, 2, 0, 4]);  unsqueeze_5 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(permute_67, 0);  permute_67 = None
        clone_27 = torch.ops.aten.clone.default(squeeze_3, memory_format = torch.contiguous_format);  squeeze_3 = None
        view_68 = torch.ops.aten.view.default(clone_27, [32, 4, 384]);  clone_27 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(view_68, [0, 1], True)
        view_69 = torch.ops.aten.view.default(sum_24, [384]);  sum_24 = None
        view_70 = torch.ops.aten.view.default(view_68, [128, 384]);  view_68 = None
        permute_68 = torch.ops.aten.permute.default(view_70, [1, 0])
        mm_19 = torch.ops.aten.mm.default(permute_68, view);  permute_68 = view = None
        permute_1 = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_70 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_20 = torch.ops.aten.mm.default(view_70, permute_70);  view_70 = permute_70 = None
        view_71 = torch.ops.aten.view.default(mm_20, [32, 4, 128]);  mm_20 = None
        permute_72 = torch.ops.aten.permute.default(view_71, [1, 0, 2]);  view_71 = None
        mul_59 = torch.ops.aten.mul.Tensor(permute_72, primals_4);  primals_4 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, 128)
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_59, [2], True)
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_59, mul);  mul_59 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(mul_61, [2], True);  mul_61 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul, sum_26);  sum_26 = None
        sub_19 = torch.ops.aten.sub.Tensor(mul_60, sum_25);  mul_60 = sum_25 = None
        sub_20 = torch.ops.aten.sub.Tensor(sub_19, mul_62);  sub_19 = mul_62 = None
        div_4 = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        mul_63 = torch.ops.aten.mul.Tensor(div_4, sub_20);  div_4 = sub_20 = None
        mul_64 = torch.ops.aten.mul.Tensor(permute_72, mul);  mul = None
        sum_27 = torch.ops.aten.sum.dim_IntList(mul_64, [0, 1]);  mul_64 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(permute_72, [0, 1]);  permute_72 = None
        add_30 = torch.ops.aten.add.Tensor(add_27, mul_63);  add_27 = mul_63 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(add_30, [0], True)
        view_72 = torch.ops.aten.view.default(sum_29, [32, 128]);  sum_29 = None
        eq = torch.ops.aten.eq.Scalar(iota, -1)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        full_default_6 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(unsqueeze_6, full_default_6, view_72);  unsqueeze_6 = view_72 = None
        full_default_7 = torch.ops.aten.full.default([128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put = torch.ops.aten.index_put.default(full_default_7, [iota], where_1, True);  full_default_7 = iota = where_1 = None
        eq_1 = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_2 = torch.ops.aten.where.self(unsqueeze_7, full_default_6, add_30);  unsqueeze_7 = full_default_6 = add_30 = None
        full_default_9 = torch.ops.aten.full.default([256, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1 = torch.ops.aten.index_put.default(full_default_9, [primals_1], where_2, True);  full_default_9 = primals_1 = where_2 = None
        return (None, index_put_1, index_put, sum_27, sum_28, view_69, mm_19, mm_18, view_60, sum_21, sum_22, mm_16, view_57, mm_14, view_54, sum_15, sum_16, view_50, mm_11, mm_10, view_41, sum_9, sum_10, mm_8, view_38, mm_6, view_35, sum_3, sum_4, mm_3)

def load_args(reader):
    buf0 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (4, 32), dtype=torch.int64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128,), is_leaf=True)  # primals_4
    buf2 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf2, (384, 128), is_leaf=True)  # primals_7
    buf3 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128, 128), is_leaf=True)  # primals_8
    buf4 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf4, (128,), is_leaf=True)  # primals_10
    buf5 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf5, (256, 128), is_leaf=True)  # primals_12
    buf6 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (128, 256), is_leaf=True)  # primals_14
    buf7 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf7, (128,), is_leaf=True)  # primals_16
    buf8 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf8, (384, 128), is_leaf=True)  # primals_19
    buf9 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf9, (128, 128), is_leaf=True)  # primals_20
    buf10 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf10, (128,), is_leaf=True)  # primals_22
    buf11 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (256, 128), is_leaf=True)  # primals_24
    buf12 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (128, 256), is_leaf=True)  # primals_26
    buf13 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf13, (128,), is_leaf=True)  # primals_28
    buf14 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf14, (256, 128), is_leaf=True)  # primals_30
    buf15 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf15, (32,), dtype=torch.int64, is_leaf=True)  # iota
    buf16 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf16, (4, 32, 128), is_leaf=True)  # embedding
    buf17 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf17, (32, 128), is_leaf=True)  # embedding_1
    buf18 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf18, (4, 32, 1), is_leaf=True)  # getitem_1
    buf19 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf19, (4, 32, 1), is_leaf=True)  # rsqrt
    buf20 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128, 128), is_leaf=True)  # view
    buf21 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf21, (4, 4, 32, 32), (128, 32, 512, 1), is_leaf=True)  # view_6
    reader.tensor(buf21, (4, 4, 32, 32), (128, 32, 512, 1), storage_offset=16384, is_leaf=True)  # view_7
    reader.tensor(buf21, (4, 4, 32, 32), (128, 32, 512, 1), storage_offset=32768, is_leaf=True)  # view_8
    buf22 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf22, (4, 4, 32, 32), (4096, 32, 128, 1), is_leaf=True)  # getitem_2
    buf23 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf23, (4, 4, 32), is_leaf=True)  # getitem_3
    buf24 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf24, (), dtype=torch.int64, is_leaf=True)  # getitem_4
    buf25 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf25, (), dtype=torch.int64, is_leaf=True)  # getitem_5
    buf26 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf26, (128, 128), is_leaf=True)  # view_9
    buf27 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf27, (4, 32, 128), is_leaf=True)  # mul_2
    buf28 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf28, (128, 128), is_leaf=True)  # view_11
    buf29 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf29, (128, 256), is_leaf=True)  # addmm_1
    buf30 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf30, (128, 256), is_leaf=True)  # view_13
    buf31 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf31, (4, 32, 128), is_leaf=True)  # mul_7
    buf32 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf32, (128, 128), is_leaf=True)  # view_15
    buf33 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf33, (4, 4, 32, 32), (128, 32, 512, 1), is_leaf=True)  # view_21
    reader.tensor(buf33, (4, 4, 32, 32), (128, 32, 512, 1), storage_offset=16384, is_leaf=True)  # view_22
    reader.tensor(buf33, (4, 4, 32, 32), (128, 32, 512, 1), storage_offset=32768, is_leaf=True)  # view_23
    buf34 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf34, (4, 4, 32, 32), (4096, 32, 128, 1), is_leaf=True)  # getitem_10
    buf35 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf35, (4, 4, 32), is_leaf=True)  # getitem_11
    buf36 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf36, (), dtype=torch.int64, is_leaf=True)  # getitem_12
    buf37 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf37, (), dtype=torch.int64, is_leaf=True)  # getitem_13
    buf38 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf38, (128, 128), is_leaf=True)  # view_24
    buf39 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf39, (4, 32, 128), is_leaf=True)  # mul_9
    buf40 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf40, (128, 128), is_leaf=True)  # view_26
    buf41 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf41, (128, 256), is_leaf=True)  # addmm_4
    buf42 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf42, (128, 256), is_leaf=True)  # view_28
    buf43 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf43, (4, 32, 128), is_leaf=True)  # mul_14
    buf44 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf44, (128, 128), is_leaf=True)  # view_30
    buf45 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf45, (4, 32, 1), is_leaf=True)  # div
    buf46 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf46, (4, 32, 1), is_leaf=True)  # div_1
    buf47 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf47, (4, 32, 1), is_leaf=True)  # div_2
    buf48 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf48, (4, 32, 1), is_leaf=True)  # div_3
    buf49 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf49, (4, 32, 256), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)