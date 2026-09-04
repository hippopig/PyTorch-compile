# AOT ID: ['0_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/dy/cdy6jtvjazfc3fc36uzdtwy4cu5e7rf7lxorfhi5kz6vzs6kl35g.py
# Topologically Sorted Source Nodes: [view_33, mul_17, mul_18, sum_1, mul_19, sum_2, mul_20, sub_7, sub_8, mul_21], Original ATen: [aten.view, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_17 => mul_17
#   mul_18 => mul_18
#   mul_19 => mul_19
#   mul_20 => mul_20
#   mul_21 => mul_21
#   sub_7 => sub_7
#   sub_8 => sub_8
#   sum_1 => sum_1
#   sum_2 => sum_2
#   view_33 => view_33
# Graph fragment:
#   %mm_4 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %primals_28 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_28]
#   %mul_14 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=mul_14]
#   %div : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0" = PlaceHolder[target=div]
#   %sum_1 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=sum_1]
#   %sum_2 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=sum_2]
#   %view_33 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [4, 32, 128]), kwargs = {})
#   %mul_17 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_33, %primals_28), kwargs = {})
#   %mul_18 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, 128), kwargs = {})
#   %sum_1 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_17, [2], True), kwargs = {})
#   %mul_19 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %mul_14), kwargs = {})
#   %sum_2 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_19, [2], True), kwargs = {})
#   %mul_20 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %sum_2), kwargs = {})
#   %sub_7 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_18, %sum_1), kwargs = {})
#   %sub_8 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_7, %mul_20), kwargs = {})
#   %mul_21 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sub_8), kwargs = {})
#   return %sum_1,%sum_2,%mul_21
triton_per_fused_native_layer_norm_backward_view_0 = async_compile.triton('triton_per_fused_native_layer_norm_backward_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 2, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 512, 'r0_': 262656}}
)
@triton.jit
def triton_per_fused_native_layer_norm_backward_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp14 = tl.full([1, 1], 128.0, tl.float32)
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(out_ptr2 + (r0_1 + 128*x0), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/z4/cz4nzdnvsra2jd3tjomao4ld5cx7zbrfpsvt4pcr3fq2qb7j5pzl.py
# Topologically Sorted Source Nodes: [view_33, mul_22, sum_3, sum_4], Original ATen: [aten.view, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_22 => mul_22
#   sum_3 => sum_3
#   sum_4 => sum_4
#   view_33 => view_33
# Graph fragment:
#   %mm_4 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %mul_14 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=mul_14]
#   %view_33 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [4, 32, 128]), kwargs = {})
#   %mul_22 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_33, %mul_14), kwargs = {})
#   %sum_3 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_22, [0, 1]), kwargs = {})
#   %sum_4 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_33, [0, 1]), kwargs = {})
#   return %sum_3,%sum_4
triton_red_fused_native_layer_norm_backward_view_1 = async_compile.triton('triton_red_fused_native_layer_norm_backward_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 133120, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_layer_norm_backward_view_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/fo/cfoe75efp73mxnrjzr2du7yccqyzznf6ao7uw65rsbdbqsmacheo.py
# Topologically Sorted Source Nodes: [view_34, sum_5], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_5 => sum_5
#   view_34 => view_34
# Graph fragment:
#   %mul_21 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=mul_21]
#   %view_34 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_21, [128, 128]), kwargs = {})
#   %sum_5 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_34, [0], True), kwargs = {})
#   return %sum_5
triton_red_fused_sum_view_2 = async_compile.triton('triton_red_fused_sum_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 66560, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_view_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/oj/cojznnaopfpmbrydyhiwcb2kauxni6avsljfj5v75vh3y7k7yx3p.py
# Topologically Sorted Source Nodes: [view_36, linear_2, gelu_1, mul_24, mul_25, mul_26, exp, mul_27, mul_28, add_20, mul_29], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
# Source node to ATen node mapping:
#   add_20 => add_20
#   exp => exp
#   gelu_1 => add_15, erf_1, mul_12
#   linear_2 => view_27
#   mul_24 => mul_24
#   mul_25 => mul_25
#   mul_26 => mul_26
#   mul_27 => mul_27
#   mul_28 => mul_28
#   mul_29 => mul_29
#   view_36 => view_36
# Graph fragment:
#   %mm_5 : Tensor "f32[128, 256][256, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %addmm_4 : Tensor "f32[128, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_4]
#   %view_36 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [4, 32, 256]), kwargs = {})
#   %view_27 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_4, [4, 32, 256]), kwargs = {})
#   %mul_12 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_27, 0.7071067811865476), kwargs = {})
#   %erf_1 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_12,), kwargs = {})
#   %add_15 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_1, 1), kwargs = {})
#   %mul_24 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, 0.5), kwargs = {})
#   %mul_25 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_27, %view_27), kwargs = {})
#   %mul_26 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, -0.5), kwargs = {})
#   %exp : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_26,), kwargs = {})
#   %mul_27 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, 0.3989422804014327), kwargs = {})
#   %mul_28 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_27, %mul_27), kwargs = {})
#   %add_20 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %mul_28), kwargs = {})
#   %mul_29 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_36, %add_20), kwargs = {})
#   return %mul_29
triton_poi_fused_gelu_gelu_backward_view_3 = async_compile.triton('triton_poi_fused_gelu_gelu_backward_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_gelu_backward_view_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.erf(tmp3)
    tmp5 = tl.full([1], 1.0, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 0.5, tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = tl.full([1], -0.5, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = libdevice.exp(tmp11)
    tmp13 = tl.full([1], 0.3989422804014327, tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/kj/ckj2xdu3hnqlmekr42p6j6m5zab3s746p3fkvpyo2cs2x5n7klf5.py
# Topologically Sorted Source Nodes: [view_36, linear_2, gelu_1, mul_24, mul_25, mul_26, exp, mul_27, mul_28, add_20, mul_29, view_37, sum_6], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten.sum]
# Source node to ATen node mapping:
#   add_20 => add_20
#   exp => exp
#   gelu_1 => add_15, erf_1, mul_12
#   linear_2 => view_27
#   mul_24 => mul_24
#   mul_25 => mul_25
#   mul_26 => mul_26
#   mul_27 => mul_27
#   mul_28 => mul_28
#   mul_29 => mul_29
#   sum_6 => sum_6
#   view_36 => view_36
#   view_37 => view_37
# Graph fragment:
#   %mul_29 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0" = PlaceHolder[target=mul_29]
#   %view_36 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [4, 32, 256]), kwargs = {})
#   %view_27 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_4, [4, 32, 256]), kwargs = {})
#   %mul_12 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_27, 0.7071067811865476), kwargs = {})
#   %erf_1 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_12,), kwargs = {})
#   %add_15 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_1, 1), kwargs = {})
#   %mul_24 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, 0.5), kwargs = {})
#   %mul_25 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_27, %view_27), kwargs = {})
#   %mul_26 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, -0.5), kwargs = {})
#   %exp : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_26,), kwargs = {})
#   %mul_27 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, 0.3989422804014327), kwargs = {})
#   %mul_28 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_27, %mul_27), kwargs = {})
#   %add_20 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %mul_28), kwargs = {})
#   %mul_29 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_36, %add_20), kwargs = {})
#   %view_37 : Tensor "f32[128, 256][256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_29, [128, 256]), kwargs = {})
#   %sum_6 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_37, [0], True), kwargs = {})
#   return %sum_6
triton_red_fused_gelu_gelu_backward_sum_view_4 = async_compile.triton('triton_red_fused_gelu_gelu_backward_sum_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_gelu_backward_sum_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 133120, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_gelu_gelu_backward_sum_view_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/gw/cgwibzccifdy2hny7uzlegemuagysy4xdapn4rwvgex3upa4cixv.py
# Topologically Sorted Source Nodes: [view_39, mul_31, mul_32, sum_7, mul_33, sum_8, mul_34, sub_10, sub_11, mul_35, add_21], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
# Source node to ATen node mapping:
#   add_21 => add_21
#   mul_31 => mul_31
#   mul_32 => mul_32
#   mul_33 => mul_33
#   mul_34 => mul_34
#   mul_35 => mul_35
#   sub_10 => sub_10
#   sub_11 => sub_11
#   sum_7 => sum_7
#   sum_8 => sum_8
#   view_39 => view_39
# Graph fragment:
#   %mm_7 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=mm_7]
#   %primals_22 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_22]
#   %mul_9 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=mul_9]
#   %mul_21 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=mul_21]
#   %div_1 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0" = PlaceHolder[target=div_1]
#   %sum_7 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=sum_7]
#   %sum_8 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=sum_8]
#   %view_39 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [4, 32, 128]), kwargs = {})
#   %mul_31 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_39, %primals_22), kwargs = {})
#   %mul_32 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, 128), kwargs = {})
#   %sum_7 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_31, [2], True), kwargs = {})
#   %mul_33 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %mul_9), kwargs = {})
#   %sum_8 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_33, [2], True), kwargs = {})
#   %mul_34 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %sum_8), kwargs = {})
#   %sub_10 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_32, %sum_7), kwargs = {})
#   %sub_11 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_10, %mul_34), kwargs = {})
#   %mul_35 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sub_11), kwargs = {})
#   %add_21 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %mul_35), kwargs = {})
#   return %sum_7,%sum_8,%add_21
triton_per_fused_add_native_layer_norm_backward_view_5 = async_compile.triton('triton_per_fused_add_native_layer_norm_backward_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 2, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 512, 'r0_': 328192}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_backward_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp15 = tl.full([1, 1], 128.0, tl.float32)
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/u5/cu5qjo6vyniy3of6r2cnvhnlbpkwogliv22nufdflupvposgxpko.py
# Topologically Sorted Source Nodes: [permute_35, clone_12, view_40, sum_11], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.sum]
# Source node to ATen node mapping:
#   clone_12 => clone_12
#   permute_35 => permute_35
#   sum_11 => sum_11
#   view_40 => view_40
# Graph fragment:
#   %add_21 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=add_21]
#   %view_40 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=view_40]
#   %permute_35 : Tensor "f32[32, 4, 128][128, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_21, [1, 0, 2]), kwargs = {})
#   %clone_12 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_35,), kwargs = {memory_format: torch.contiguous_format})
#   %view_40 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_12, [128, 128]), kwargs = {})
#   %sum_11 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_40, [0], True), kwargs = {})
#   return %view_40,%sum_11
triton_red_fused__unsafe_view_clone_sum_transpose_6 = async_compile.triton('triton_red_fused__unsafe_view_clone_sum_transpose_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_clone_sum_transpose_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 197632, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_clone_sum_transpose_6(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*(r0_1 // 4) + 4096*((r0_1 % 4))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
        tl.store(out_ptr0 + (x0 + 128*r0_1), tmp0, r0_mask & xmask)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/wl/cwlt2tzbbtzj3kscyjhjlgwfn3tyuppm2oidlp5mnwpryd6srbt6.py
# Topologically Sorted Source Nodes: [clone_13, view_43, clone_14, view_44, clone_15, view_45, permute_41, clone_16, view_46, permute_42, clone_17, view_47, permute_43, clone_18, view_48, full_default, add_22, add_23, unsqueeze_4, permute_44, squeeze_2, clone_19, view_49, sum_12], Original ATen: [aten.clone, aten._unsafe_view, aten.transpose, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
# Source node to ATen node mapping:
#   add_22 => add_22
#   add_23 => add_23
#   clone_13 => clone_13
#   clone_14 => clone_14
#   clone_15 => clone_15
#   clone_16 => clone_16
#   clone_17 => clone_17
#   clone_18 => clone_18
#   clone_19 => clone_19
#   full_default => full_default
#   permute_41 => permute_41
#   permute_42 => permute_42
#   permute_43 => permute_43
#   permute_44 => permute_44
#   squeeze_2 => squeeze_2
#   sum_12 => sum_12
#   unsqueeze_4 => unsqueeze_4
#   view_43 => view_43
#   view_44 => view_44
#   view_45 => view_45
#   view_46 => view_46
#   view_47 => view_47
#   view_48 => view_48
#   view_49 => view_49
# Graph fragment:
#   %getitem_20 : Tensor "f32[4, 4, 32, 32][4096, 32, 128, 1]cuda:0" = PlaceHolder[target=getitem_20]
#   %getitem_19 : Tensor "f32[4, 4, 32, 32][4096, 32, 128, 1]cuda:0" = PlaceHolder[target=getitem_19]
#   %getitem_18 : Tensor "f32[4, 4, 32, 32][4096, 32, 128, 1]cuda:0" = PlaceHolder[target=getitem_18]
#   %clone_13 : Tensor "f32[4, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_20,), kwargs = {memory_format: torch.contiguous_format})
#   %view_43 : Tensor "f32[16, 32, 32][1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_13, [16, 32, 32]), kwargs = {})
#   %clone_14 : Tensor "f32[4, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_19,), kwargs = {memory_format: torch.contiguous_format})
#   %view_44 : Tensor "f32[16, 32, 32][1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_14, [16, 32, 32]), kwargs = {})
#   %clone_15 : Tensor "f32[4, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_18,), kwargs = {memory_format: torch.contiguous_format})
#   %view_45 : Tensor "f32[16, 32, 32][1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_15, [16, 32, 32]), kwargs = {})
#   %permute_41 : Tensor "f32[32, 16, 32][32, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_43, [1, 0, 2]), kwargs = {})
#   %clone_16 : Tensor "f32[32, 16, 32][512, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_41,), kwargs = {memory_format: torch.contiguous_format})
#   %view_46 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_16, [32, 4, 128]), kwargs = {})
#   %permute_42 : Tensor "f32[32, 16, 32][32, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_44, [1, 0, 2]), kwargs = {})
#   %clone_17 : Tensor "f32[32, 16, 32][512, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_42,), kwargs = {memory_format: torch.contiguous_format})
#   %view_47 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_17, [32, 4, 128]), kwargs = {})
#   %permute_43 : Tensor "f32[32, 16, 32][32, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_45, [1, 0, 2]), kwargs = {})
#   %clone_18 : Tensor "f32[32, 16, 32][512, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_43,), kwargs = {memory_format: torch.contiguous_format})
#   %view_48 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_18, [32, 4, 128]), kwargs = {})
#   %full_default : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([3, 32, 4, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_46, 0, 2), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_47, 0, 1), kwargs = {})
#   %add_22 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_48, 0, 0), kwargs = {})
#   %add_23 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %select_scatter_default_2), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[3, 32, 4, 1, 128][16384, 512, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_23, 3), kwargs = {})
#   %permute_44 : Tensor "f32[1, 32, 4, 3, 128][128, 512, 128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_4, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_2 : Tensor "f32[32, 4, 3, 128][512, 128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_44, 0), kwargs = {})
#   %clone_19 : Tensor "f32[32, 4, 3, 128][1536, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_49 : Tensor "f32[32, 4, 384][1536, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_19, [32, 4, 384]), kwargs = {})
#   %sum_12 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_49, [0, 1], True), kwargs = {})
#   return %sum_12
triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_7 = async_compile.triton('triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 199680, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 384
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = (r0_index % 32)
        r0_2 = r0_index // 32
        tmp3 = tl.load(in_ptr0 + (128*r0_1 + 128*((128*r0_2 + ((x0 % 128))) // 512) + 4096*r0_2 + 4096*(((x0 % 128)) // 128) + ((x0 % 128))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr1 + (128*r0_1 + 128*((128*r0_2 + ((x0 % 128))) // 512) + 4096*r0_2 + 4096*(((x0 % 128)) // 128) + ((x0 % 128))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (128*r0_1 + 128*((128*r0_2 + ((x0 % 128))) // 512) + 4096*r0_2 + 4096*(((x0 % 128)) // 128) + ((x0 % 128))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0 // 128
        tmp1 = tl.full([1, 1], 2, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = tl.full([1, 1], 0.0, tl.float32)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.full([1, 1], 1, tl.int32)
        tmp7 = tmp0 == tmp6
        tmp9 = tl.where(tmp7, tmp8, tmp4)
        tmp10 = tmp5 + tmp9
        tmp11 = tl.full([1, 1], 0, tl.int32)
        tmp12 = tmp0 == tmp11
        tmp14 = tl.where(tmp12, tmp13, tmp4)
        tmp15 = tmp10 + tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/pk/cpkfywdhszqn5ge43heufiglzsp7lytvua4nq6vak7rmgbquo5ll.py
# Topologically Sorted Source Nodes: [clone_13, view_43, clone_14, view_44, clone_15, view_45, permute_41, clone_16, view_46, permute_42, clone_17, view_47, permute_43, clone_18, view_48, full_default, add_22, add_23, unsqueeze_4, permute_44, squeeze_2, clone_19], Original ATen: [aten.clone, aten._unsafe_view, aten.transpose, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze]
# Source node to ATen node mapping:
#   add_22 => add_22
#   add_23 => add_23
#   clone_13 => clone_13
#   clone_14 => clone_14
#   clone_15 => clone_15
#   clone_16 => clone_16
#   clone_17 => clone_17
#   clone_18 => clone_18
#   clone_19 => clone_19
#   full_default => full_default
#   permute_41 => permute_41
#   permute_42 => permute_42
#   permute_43 => permute_43
#   permute_44 => permute_44
#   squeeze_2 => squeeze_2
#   unsqueeze_4 => unsqueeze_4
#   view_43 => view_43
#   view_44 => view_44
#   view_45 => view_45
#   view_46 => view_46
#   view_47 => view_47
#   view_48 => view_48
# Graph fragment:
#   %getitem_20 : Tensor "f32[4, 4, 32, 32][4096, 32, 128, 1]cuda:0" = PlaceHolder[target=getitem_20]
#   %getitem_19 : Tensor "f32[4, 4, 32, 32][4096, 32, 128, 1]cuda:0" = PlaceHolder[target=getitem_19]
#   %getitem_18 : Tensor "f32[4, 4, 32, 32][4096, 32, 128, 1]cuda:0" = PlaceHolder[target=getitem_18]
#   %clone_13 : Tensor "f32[4, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_20,), kwargs = {memory_format: torch.contiguous_format})
#   %view_43 : Tensor "f32[16, 32, 32][1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_13, [16, 32, 32]), kwargs = {})
#   %clone_14 : Tensor "f32[4, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_19,), kwargs = {memory_format: torch.contiguous_format})
#   %view_44 : Tensor "f32[16, 32, 32][1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_14, [16, 32, 32]), kwargs = {})
#   %clone_15 : Tensor "f32[4, 4, 32, 32][4096, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%getitem_18,), kwargs = {memory_format: torch.contiguous_format})
#   %view_45 : Tensor "f32[16, 32, 32][1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_15, [16, 32, 32]), kwargs = {})
#   %permute_41 : Tensor "f32[32, 16, 32][32, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_43, [1, 0, 2]), kwargs = {})
#   %clone_16 : Tensor "f32[32, 16, 32][512, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_41,), kwargs = {memory_format: torch.contiguous_format})
#   %view_46 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_16, [32, 4, 128]), kwargs = {})
#   %permute_42 : Tensor "f32[32, 16, 32][32, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_44, [1, 0, 2]), kwargs = {})
#   %clone_17 : Tensor "f32[32, 16, 32][512, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_42,), kwargs = {memory_format: torch.contiguous_format})
#   %view_47 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_17, [32, 4, 128]), kwargs = {})
#   %permute_43 : Tensor "f32[32, 16, 32][32, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_45, [1, 0, 2]), kwargs = {})
#   %clone_18 : Tensor "f32[32, 16, 32][512, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_43,), kwargs = {memory_format: torch.contiguous_format})
#   %view_48 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_18, [32, 4, 128]), kwargs = {})
#   %full_default : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([3, 32, 4, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_46, 0, 2), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_47, 0, 1), kwargs = {})
#   %add_22 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_48, 0, 0), kwargs = {})
#   %add_23 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %select_scatter_default_2), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[3, 32, 4, 1, 128][16384, 512, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_23, 3), kwargs = {})
#   %permute_44 : Tensor "f32[1, 32, 4, 3, 128][128, 512, 128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_4, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_2 : Tensor "f32[32, 4, 3, 128][512, 128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_44, 0), kwargs = {})
#   %clone_19 : Tensor "f32[32, 4, 3, 128][1536, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_2,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_19
triton_poi_fused__unsafe_view_add_clone_select_backward_squeeze_transpose_unsqueeze_8 = async_compile.triton('triton_poi_fused__unsafe_view_add_clone_select_backward_squeeze_transpose_unsqueeze_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_select_backward_squeeze_transpose_unsqueeze_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_select_backward_squeeze_transpose_unsqueeze_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 128) % 3)
    x0 = (xindex % 128)
    x2 = ((xindex // 384) % 32)
    x3 = xindex // 12288
    x4 = (xindex % 384)
    tmp3 = tl.load(in_ptr0 + (x0 + 128*x2 + 128*((x0 + 128*x3) // 512) + 4096*x3), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 128*x2 + 128*((x0 + 128*x3) // 512) + 4096*x3), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0 + 128*x2 + 128*((x0 + 128*x3) // 512) + 4096*x3), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 0.0, tl.float32)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = tmp0 == tmp6
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp5 + tmp9
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp14 = tl.where(tmp12, tmp13, tmp4)
    tmp15 = tmp10 + tmp14
    tl.store(out_ptr0 + (x4 + 384*x3 + 1536*x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/wv/cwvfnltmyvmkwy5jleleasgekhtfuffqab3ccghqldnrvg2daic4.py
# Topologically Sorted Source Nodes: [view_52, permute_49, mul_38, sum_13], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_38 => mul_38
#   permute_49 => permute_49
#   sum_13 => sum_13
#   view_52 => view_52
# Graph fragment:
#   %mm_12 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=mm_12]
#   %primals_16 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_16]
#   %view_52 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_12, [32, 4, 128]), kwargs = {})
#   %permute_49 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_52, [1, 0, 2]), kwargs = {})
#   %mul_38 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_49, %primals_16), kwargs = {})
#   %sum_13 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_38, [2], True), kwargs = {})
#   return %sum_13
triton_per_fused_native_layer_norm_backward_transpose_view_9 = async_compile.triton('triton_per_fused_native_layer_norm_backward_transpose_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_transpose_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024, 'r0_': 66048}}
)
@triton.jit
def triton_per_fused_native_layer_norm_backward_transpose_view_9(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/ht/cht3aav4nla4japuw3pd4orb4kuf4icg3gngevv4rfk6goclmqnx.py
# Topologically Sorted Source Nodes: [view_52, permute_49, mul_38, mul_39, mul_40, sum_14, mul_41, sub_13, sub_14, mul_42, add_24], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.add]
# Source node to ATen node mapping:
#   add_24 => add_24
#   mul_38 => mul_38
#   mul_39 => mul_39
#   mul_40 => mul_40
#   mul_41 => mul_41
#   mul_42 => mul_42
#   permute_49 => permute_49
#   sub_13 => sub_13
#   sub_14 => sub_14
#   sum_14 => sum_14
#   view_52 => view_52
# Graph fragment:
#   %mm_12 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=mm_12]
#   %primals_16 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_16]
#   %mul_7 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=mul_7]
#   %add_21 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=add_21]
#   %div_2 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0" = PlaceHolder[target=div_2]
#   %sum_13 : Tensor "f32[4, 32, 1][1, 4, 128]cuda:0" = PlaceHolder[target=sum_13]
#   %sum_14 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=sum_14]
#   %view_52 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_12, [32, 4, 128]), kwargs = {})
#   %permute_49 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_52, [1, 0, 2]), kwargs = {})
#   %mul_38 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_49, %primals_16), kwargs = {})
#   %mul_39 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, 128), kwargs = {})
#   %mul_40 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, %mul_7), kwargs = {})
#   %sum_14 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_40, [2], True), kwargs = {})
#   %mul_41 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %sum_14), kwargs = {})
#   %sub_13 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_39, %sum_13), kwargs = {})
#   %sub_14 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_13, %mul_41), kwargs = {})
#   %mul_42 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sub_14), kwargs = {})
#   %add_24 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %mul_42), kwargs = {})
#   return %sum_14,%add_24
triton_red_fused_add_native_layer_norm_backward_transpose_view_10 = async_compile.triton('triton_red_fused_add_native_layer_norm_backward_transpose_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_transpose_view_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 512, 'r0_': 328192}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_backward_transpose_view_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    x3 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x1 + 512*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp9 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1 + 4*x0), xmask, eviction_policy='evict_last')
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp8 = tl.load(in_out_ptr0 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr0 + (r0_2 + 128*x1 + 512*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tmp10 * tmp11
        tmp13 = tl.full([1, 1], 128.0, tl.float32)
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 - tmp15
        tmp18 = tmp17 * tmp6
        tmp19 = tmp16 - tmp18
        tmp20 = tmp9 * tmp19
        tmp21 = tmp8 + tmp20
        tl.store(in_out_ptr0 + (r0_2 + 128*x3), tmp21, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/ll/cllj2yz4tathh3uke6yxth7tmuvm2gydckmuotyp6hjkpa6pjipw.py
# Topologically Sorted Source Nodes: [view_52, permute_49, mul_43, sum_15], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_43 => mul_43
#   permute_49 => permute_49
#   sum_15 => sum_15
#   view_52 => view_52
# Graph fragment:
#   %mm_12 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=mm_12]
#   %mul_7 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=mul_7]
#   %view_52 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_12, [32, 4, 128]), kwargs = {})
#   %permute_49 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_52, [1, 0, 2]), kwargs = {})
#   %mul_43 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_49, %mul_7), kwargs = {})
#   %sum_15 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_43, [0, 1]), kwargs = {})
#   return %sum_15
triton_red_fused_native_layer_norm_backward_transpose_view_11 = async_compile.triton('triton_red_fused_native_layer_norm_backward_transpose_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_transpose_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 132096, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_layer_norm_backward_transpose_view_11(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = (r0_index % 32)
        r0_2 = r0_index // 32
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 512*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/i7/ci7tta56blbxknc7wp6temkk3q2i4wdtkbhzkskh5naq3ernrdrc.py
# Topologically Sorted Source Nodes: [full_default_9], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   full_default_9 => full_default_9
# Graph fragment:
#   %full_default_9 : Tensor "f32[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([256, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %index_put_1
triton_poi_fused_embedding_dense_backward_12 = async_compile.triton('triton_poi_fused_embedding_dense_backward_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_12(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/vc/cvcufx3bbbbv4cbl5w23azji5ftueunidvkoo7nvtj2sxhjr4bo4.py
# Topologically Sorted Source Nodes: [view_71, permute_72, mul_59, mul_60, hidden, layer_norm, mul_61, sum_26, mul_62, sub_19, sub_20, div_4, mul_63, add_30, full_default_6, eq_1, unsqueeze_7, where_2, full_default_9, index_put_1], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_layer_norm, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   add_30 => add_30
#   div_4 => div_4
#   eq_1 => eq_1
#   full_default_6 => full_default_6
#   full_default_9 => full_default_9
#   hidden => add
#   index_put_1 => index_put_1
#   layer_norm => mul, sub_1
#   mul_59 => mul_59
#   mul_60 => mul_60
#   mul_61 => mul_61
#   mul_62 => mul_62
#   mul_63 => mul_63
#   permute_72 => permute_72
#   sub_19 => sub_19
#   sub_20 => sub_20
#   sum_26 => sum_26
#   unsqueeze_7 => unsqueeze_7
#   view_71 => view_71
#   where_2 => where_2
# Graph fragment:
#   %mm_20 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=mm_20]
#   %primals_4 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_4]
#   %embedding : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=embedding]
#   %embedding_1 : Tensor "f32[32, 128][128, 1]cuda:0" = PlaceHolder[target=embedding_1]
#   %getitem_1 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %rsqrt : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_27 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=add_27]
#   %sum_25 : Tensor "f32[4, 32, 1][1, 4, 128]cuda:0" = PlaceHolder[target=sum_25]
#   %sum_26 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=sum_26]
#   %primals_1 : Tensor "i64[4, 32][32, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %add_30 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=add_30]
#   %index_put_1 : Tensor "f32[256, 128][128, 1]cuda:0" = PlaceHolder[target=index_put_1]
#   %view_71 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_20, [32, 4, 128]), kwargs = {})
#   %permute_72 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_71, [1, 0, 2]), kwargs = {})
#   %mul_59 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_72, %primals_4), kwargs = {})
#   %mul_60 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_59, 128), kwargs = {})
#   %add : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %sub_1 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %mul : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_61 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_59, %mul), kwargs = {})
#   %sum_26 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_61, [2], True), kwargs = {})
#   %mul_62 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %sum_26), kwargs = {})
#   %sub_19 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_60, %sum_25), kwargs = {})
#   %sub_20 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_19, %mul_62), kwargs = {})
#   %div_4 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 128), kwargs = {})
#   %mul_63 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sub_20), kwargs = {})
#   %add_30 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %mul_63), kwargs = {})
#   %full_default_6 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq_1 : Tensor "b8[4, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%primals_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "b8[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq_1, -1), kwargs = {})
#   %where_2 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_7, %full_default_6, %add_30), kwargs = {})
#   %full_default_9 : Tensor "f32[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([256, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_1 : Tensor "f32[256, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_9, [%primals_1], %where_2, True), kwargs = {})
#   return %sum_26,%add_30,%buf68
triton_red_fused_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_transpose_view_13 = async_compile.triton('triton_red_fused_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_transpose_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*i64', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_transpose_view_13', 'mutated_arg_names': ['in_out_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': True, 'num_load': 13, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2048, 'r0_': 344576}}
)
@triton.jit
def triton_red_fused_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_transpose_view_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    x3 = xindex
    tmp6 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x1 + 512*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 128*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 - tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tmp2 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp22 = tl.load(in_ptr6 + (x1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp14 = tl.load(in_out_ptr0 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr0 + (r0_2 + 128*x1 + 512*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr2 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.load(in_ptr3 + (r0_2 + 128*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.full([1, 1], 0.0078125, tl.float32)
        tmp16 = tmp8 * tmp15
        tmp19 = tmp17 * tmp18
        tmp20 = tl.full([1, 1], 128.0, tl.float32)
        tmp21 = tmp19 * tmp20
        tmp23 = tmp21 - tmp22
        tmp26 = tmp24 + tmp25
        tmp27 = tmp26 - tmp6
        tmp28 = tmp27 * tmp8
        tmp29 = tmp28 * tmp12
        tmp30 = tmp23 - tmp29
        tmp31 = tmp16 * tmp30
        tmp32 = tmp14 + tmp31
        tmp34 = tl.full([1, 1], 256, tl.int32)
        tmp35 = tmp33 + tmp34
        tmp36 = tmp33 < 0
        tmp37 = tl.where(tmp36, tmp35, tmp33)
        tl.device_assert(((0 <= tmp37) & (tmp37 < 256)) | ~(xmask), "index out of bounds: 0 <= tmp37 < 256")
        tmp39 = tl.full([1, 1], -1, tl.int64)
        tmp40 = tmp33 == tmp39
        tmp41 = tl.full([1, 1], 0.0, tl.float32)
        tmp42 = tl.where(tmp40, tmp41, tmp32)
        tl.store(in_out_ptr0 + (r0_2 + 128*x3), tmp32, r0_mask & xmask)
        tl.atomic_add(out_ptr1 + (tl.broadcast_to(r0_2 + 128*tmp37, [XBLOCK, R0_BLOCK])), tmp42, r0_mask & xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/py/cpyvztrx6cgdsv55yngu26aahplwknayyo2a46co6tlnjub6mbvz.py
# Topologically Sorted Source Nodes: [view_71, permute_72, hidden, layer_norm, mul_64, sum_27], Original ATen: [aten.view, aten.transpose, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   hidden => add
#   layer_norm => mul, sub_1
#   mul_64 => mul_64
#   permute_72 => permute_72
#   sum_27 => sum_27
#   view_71 => view_71
# Graph fragment:
#   %mm_20 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=mm_20]
#   %embedding : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=embedding]
#   %embedding_1 : Tensor "f32[32, 128][128, 1]cuda:0" = PlaceHolder[target=embedding_1]
#   %getitem_1 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %rsqrt : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %view_71 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_20, [32, 4, 128]), kwargs = {})
#   %permute_72 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_71, [1, 0, 2]), kwargs = {})
#   %add : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %sub_1 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %mul : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_64 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_72, %mul), kwargs = {})
#   %sum_27 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_64, [0, 1]), kwargs = {})
#   return %sum_27
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_14 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 148480, 'r0_': 1024}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = (r0_index % 32)
        r0_2 = r0_index // 32
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 512*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr4 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = tmp3 - tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/t2/ct22fz56rkeg5yfmyh5fp2i6ywokuidvglbgm5ppxqtvchqw3vw7.py
# Topologically Sorted Source Nodes: [full_default_7], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   full_default_7 => full_default_7
# Graph fragment:
#   %full_default_7 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %index_put
triton_poi_fused_embedding_dense_backward_15 = async_compile.triton('triton_poi_fused_embedding_dense_backward_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_15(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmp81ib749v/yp/cyp5xmjb2ezo3puzg77arm5bvx4r4cqdqrixgsqstbv3xcmuqug2.py
# Topologically Sorted Source Nodes: [sum_29, view_72, eq, unsqueeze_6, full_default_6, where_1, full_default_7, index_put], Original ATen: [aten.sum, aten.view, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   eq => eq
#   full_default_6 => full_default_6
#   full_default_7 => full_default_7
#   index_put => index_put
#   sum_29 => sum_29
#   unsqueeze_6 => unsqueeze_6
#   view_72 => view_72
#   where_1 => where_1
# Graph fragment:
#   %iota : Tensor "i64[32][1]cuda:0" = PlaceHolder[target=iota]
#   %add_30 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=add_30]
#   %index_put : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=index_put]
#   %sum_29 : Tensor "f32[1, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_30, [0], True), kwargs = {})
#   %view_72 : Tensor "f32[32, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_29, [32, 128]), kwargs = {})
#   %eq : Tensor "b8[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%iota, -1), kwargs = {})
#   %unsqueeze_6 : Tensor "b8[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, -1), kwargs = {})
#   %full_default_6 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "f32[32, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_6, %full_default_6, %view_72), kwargs = {})
#   %full_default_7 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_7, [%iota], %where_1, True), kwargs = {})
#   return %buf66
triton_poi_fused_embedding_dense_backward_sum_view_16 = async_compile.triton('triton_poi_fused_embedding_dense_backward_sum_view_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_sum_view_16', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': True, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 65536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_sum_view_16(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 128
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x2), None)
    tmp9 = tl.load(in_ptr1 + (4096 + x2), None)
    tmp11 = tl.load(in_ptr1 + (8192 + x2), None)
    tmp13 = tl.load(in_ptr1 + (12288 + x2), None)
    tmp1 = tl.full([XBLOCK], 128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 128), "index out of bounds: 0 <= tmp4 < 128")
    tmp6 = tl.full([1], -1, tl.int64)
    tmp7 = tmp0 == tmp6
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 0.0, tl.float32)
    tmp16 = tl.where(tmp7, tmp15, tmp14)
    tl.atomic_add(out_ptr0 + (x0 + 128*tmp4), tmp16, None, sem='relaxed')
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_4, primals_7, primals_8, primals_10, primals_12, primals_14, primals_16, primals_19, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, iota, embedding, embedding_1, getitem_1, rsqrt, view, view_6, view_7, view_8, getitem_2, getitem_3, getitem_4, getitem_5, view_9, mul_2, view_11, addmm_1, view_13, mul_7, view_15, view_21, view_22, view_23, getitem_10, getitem_11, getitem_12, getitem_13, view_24, mul_9, view_26, addmm_4, view_28, mul_14, view_30, div, div_1, div_2, div_3, tangents_1 = args
        args.clear()
        assert_size_stride(primals_1, (4, 32), (32, 1))
        assert_size_stride(primals_4, (128, ), (1, ))
        assert_size_stride(primals_7, (384, 128), (128, 1))
        assert_size_stride(primals_8, (128, 128), (128, 1))
        assert_size_stride(primals_10, (128, ), (1, ))
        assert_size_stride(primals_12, (256, 128), (128, 1))
        assert_size_stride(primals_14, (128, 256), (256, 1))
        assert_size_stride(primals_16, (128, ), (1, ))
        assert_size_stride(primals_19, (384, 128), (128, 1))
        assert_size_stride(primals_20, (128, 128), (128, 1))
        assert_size_stride(primals_22, (128, ), (1, ))
        assert_size_stride(primals_24, (256, 128), (128, 1))
        assert_size_stride(primals_26, (128, 256), (256, 1))
        assert_size_stride(primals_28, (128, ), (1, ))
        assert_size_stride(primals_30, (256, 128), (128, 1))
        assert_size_stride(iota, (32, ), (1, ))
        assert_size_stride(embedding, (4, 32, 128), (4096, 128, 1))
        assert_size_stride(embedding_1, (32, 128), (128, 1))
        assert_size_stride(getitem_1, (4, 32, 1), (32, 1, 1))
        assert_size_stride(rsqrt, (4, 32, 1), (32, 1, 1))
        assert_size_stride(view, (128, 128), (128, 1))
        assert_size_stride(view_6, (4, 4, 32, 32), (128, 32, 512, 1))
        assert_size_stride(view_7, (4, 4, 32, 32), (128, 32, 512, 1))
        assert_size_stride(view_8, (4, 4, 32, 32), (128, 32, 512, 1))
        assert_size_stride(getitem_2, (4, 4, 32, 32), (4096, 32, 128, 1))
        assert_size_stride(getitem_3, (4, 4, 32), (128, 32, 1))
        assert_size_stride(getitem_4, (), ())
        assert_size_stride(getitem_5, (), ())
        assert_size_stride(view_9, (128, 128), (128, 1))
        assert_size_stride(mul_2, (4, 32, 128), (4096, 128, 1))
        assert_size_stride(view_11, (128, 128), (128, 1))
        assert_size_stride(addmm_1, (128, 256), (256, 1))
        assert_size_stride(view_13, (128, 256), (256, 1))
        assert_size_stride(mul_7, (4, 32, 128), (4096, 128, 1))
        assert_size_stride(view_15, (128, 128), (128, 1))
        assert_size_stride(view_21, (4, 4, 32, 32), (128, 32, 512, 1))
        assert_size_stride(view_22, (4, 4, 32, 32), (128, 32, 512, 1))
        assert_size_stride(view_23, (4, 4, 32, 32), (128, 32, 512, 1))
        assert_size_stride(getitem_10, (4, 4, 32, 32), (4096, 32, 128, 1))
        assert_size_stride(getitem_11, (4, 4, 32), (128, 32, 1))
        assert_size_stride(getitem_12, (), ())
        assert_size_stride(getitem_13, (), ())
        assert_size_stride(view_24, (128, 128), (128, 1))
        assert_size_stride(mul_9, (4, 32, 128), (4096, 128, 1))
        assert_size_stride(view_26, (128, 128), (128, 1))
        assert_size_stride(addmm_4, (128, 256), (256, 1))
        assert_size_stride(view_28, (128, 256), (256, 1))
        assert_size_stride(mul_14, (4, 32, 128), (4096, 128, 1))
        assert_size_stride(view_30, (128, 128), (128, 1))
        assert_size_stride(div, (4, 32, 1), (32, 1, 1))
        assert_size_stride(div_1, (4, 32, 1), (32, 1, 1))
        assert_size_stride(div_2, (4, 32, 1), (32, 1, 1))
        assert_size_stride(div_3, (4, 32, 1), (32, 1, 1))
        assert_size_stride(tangents_1, (4, 32, 256), (8192, 256, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((256, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_32, permute_23, mm_3], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(tangents_1, (256, 128), (1, 256), 0), view_30, out=buf0)
            del view_30
            buf1 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_32, linear_4, permute_25, mm_4], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(tangents_1, (128, 256), (256, 1), 0), primals_30, out=buf1)
            del primals_30
            del tangents_1
            buf4 = empty_strided_cuda((4, 32, 128), (4096, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_33, mul_17, mul_18, sum_1, mul_19, sum_2, mul_20, sub_7, sub_8, mul_21], Original ATen: [aten.view, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_view_0.run(buf1, primals_28, mul_14, div, buf4, 128, 128, stream=stream0)
            del div
            del primals_28
            buf5 = empty_strided_cuda((128, ), (1, ), torch.float32)
            buf6 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_33, mul_22, sum_3, sum_4], Original ATen: [aten.view, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_layer_norm_backward_view_1.run(buf1, mul_14, buf5, buf6, 128, 128, stream=stream0)
            del buf1
            del mul_14
            buf7 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_34, x_6, permute_27, mm_5], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf4, (128, 128), (128, 1), 0), primals_26, out=buf7)
            del primals_26
            buf8 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_34, permute_28, mm_6], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf4, (128, 128), (1, 128), 0), view_28, out=buf8)
            del view_28
            buf9 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_34, sum_5], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_2.run(buf4, buf9, 128, 128, stream=stream0)
            buf10 = reinterpret_tensor(buf7, (4, 32, 256), (8192, 256, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [view_36, linear_2, gelu_1, mul_24, mul_25, mul_26, exp, mul_27, mul_28, add_20, mul_29], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_gelu_backward_view_3.run(buf10, addmm_4, 32768, stream=stream0)
            del addmm_4
            buf11 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_36, linear_2, gelu_1, mul_24, mul_25, mul_26, exp, mul_27, mul_28, add_20, mul_29, view_37, permute_31, mm_7], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf10, (128, 256), (256, 1), 0), primals_24, out=buf11)
            del primals_24
            buf12 = empty_strided_cuda((256, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_36, linear_2, gelu_1, mul_24, mul_25, mul_26, exp, mul_27, mul_28, add_20, mul_29, view_37, permute_32, mm_8], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf10, (256, 128), (1, 256), 0), view_26, out=buf12)
            del view_26
            buf13 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_36, linear_2, gelu_1, mul_24, mul_25, mul_26, exp, mul_27, mul_28, add_20, mul_29, view_37, sum_6], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_sum_view_4.run(buf10, buf13, 256, 128, stream=stream0)
            del buf10
            buf18 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [view_39, mul_31, mul_32, sum_7, mul_33, sum_8, mul_34, sub_10, sub_11, mul_35, add_21], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_backward_view_5.run(buf18, buf11, primals_22, mul_9, div_1, 128, 128, stream=stream0)
            del div_1
            del primals_22
            buf16 = empty_strided_cuda((128, ), (1, ), torch.float32)
            buf17 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_39, mul_36, sum_9, sum_10], Original ATen: [aten.view, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_layer_norm_backward_view_1.run(buf11, mul_9, buf16, buf17, 128, 128, stream=stream0)
            del mul_9
            buf19 = buf11; del buf11  # reuse
            buf22 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_35, clone_12, view_40, sum_11], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_sum_transpose_6.run(buf18, buf19, buf22, 128, 128, stream=stream0)
            buf20 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1, permute_36, mm_9], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf19, primals_20, out=buf20)
            del primals_20
            buf21 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_37, mm_10], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf19, (128, 128), (1, 128), 0), view_24, out=buf21)
            del buf19
            del view_24
            # Topologically Sorted Source Nodes: [view_42, permute_40, _scaled_dot_product_efficient_attention_backward], Original ATen: [aten.view, aten.permute, aten._scaled_dot_product_efficient_attention_backward]
            buf23 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf20, (4, 4, 32, 32), (128, 32, 512, 1), 0), view_21, view_22, view_23, None, getitem_10, getitem_11, getitem_12, getitem_13, 0.0, [True, True, True, False], True)
            del buf20
            del getitem_10
            del getitem_11
            del getitem_12
            del getitem_13
            del view_21
            del view_22
            del view_23
            buf24 = buf23[0]
            assert_size_stride(buf24, (4, 4, 32, 32), (4096, 32, 128, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf24, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf25 = buf23[1]
            assert_size_stride(buf25, (4, 4, 32, 32), (4096, 32, 128, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf25, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf26 = buf23[2]
            assert_size_stride(buf26, (4, 4, 32, 32), (4096, 32, 128, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf26, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf23
            buf27 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [clone_13, view_43, clone_14, view_44, clone_15, view_45, permute_41, clone_16, view_46, permute_42, clone_17, view_47, permute_43, clone_18, view_48, full_default, add_22, add_23, unsqueeze_4, permute_44, squeeze_2, clone_19, view_49, sum_12], Original ATen: [aten.clone, aten._unsafe_view, aten.transpose, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_7.run(buf26, buf25, buf24, buf27, 384, 128, stream=stream0)
            buf28 = empty_strided_cuda((32, 4, 3, 128), (1536, 384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [clone_13, view_43, clone_14, view_44, clone_15, view_45, permute_41, clone_16, view_46, permute_42, clone_17, view_47, permute_43, clone_18, view_48, full_default, add_22, add_23, unsqueeze_4, permute_44, squeeze_2, clone_19], Original ATen: [aten.clone, aten._unsafe_view, aten.transpose, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_select_backward_squeeze_transpose_unsqueeze_8.run(buf26, buf25, buf24, buf28, 49152, stream=stream0)
            buf29 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [clone_13, view_43, clone_14, view_44, clone_15, view_45, permute_41, clone_16, view_46, permute_42, clone_17, view_47, permute_43, clone_18, view_48, full_default, add_22, add_23, unsqueeze_4, permute_44, squeeze_2, clone_19, view_49, view_51, permute_45, mm_11], Original ATen: [aten.clone, aten._unsafe_view, aten.transpose, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf28, (384, 128), (1, 384), 0), view_15, out=buf29)
            del view_15
            buf30 = reinterpret_tensor(buf26, (128, 128), (128, 1), 0); del buf26  # reuse
            # Topologically Sorted Source Nodes: [clone_13, view_43, clone_14, view_44, clone_15, view_45, permute_41, clone_16, view_46, permute_42, clone_17, view_47, permute_43, clone_18, view_48, full_default, add_22, add_23, unsqueeze_4, permute_44, squeeze_2, clone_19, view_49, view_51, multi_head_attention_forward_1, permute_47, mm_12], Original ATen: [aten.clone, aten._unsafe_view, aten.transpose, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf28, (128, 384), (384, 1), 0), primals_19, out=buf30)
            del buf28
            del primals_19
            buf31 = empty_strided_cuda((4, 32, 1), (1, 4, 128), torch.float32)
            # Topologically Sorted Source Nodes: [view_52, permute_49, mul_38, sum_13], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_transpose_view_9.run(buf30, primals_16, buf31, 128, 128, stream=stream0)
            buf35 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [view_52, permute_49, mul_38, mul_39, mul_40, sum_14, mul_41, sub_13, sub_14, mul_42, add_24], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.add]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_backward_transpose_view_10.run(buf35, buf30, primals_16, mul_7, div_2, buf31, 128, 128, stream=stream0)
            del div_2
            del primals_16
            buf33 = reinterpret_tensor(buf31, (128, ), (1, ), 0); del buf31  # reuse
            # Topologically Sorted Source Nodes: [view_52, permute_49, mul_43, sum_15], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_layer_norm_backward_transpose_view_11.run(buf30, mul_7, buf33, 128, 128, stream=stream0)
            del mul_7
            buf34 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_52, permute_49, sum_16], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_2.run(buf30, buf34, 128, 128, stream=stream0)
            buf36 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_53, x_2, permute_50, mm_13], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf35, (128, 128), (128, 1), 0), primals_14, out=buf36)
            del primals_14
            buf37 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_53, permute_51, mm_14], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf35, (128, 128), (1, 128), 0), view_13, out=buf37)
            del view_13
            buf38 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_53, sum_17], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_2.run(buf35, buf38, 128, 128, stream=stream0)
            buf39 = reinterpret_tensor(buf36, (4, 32, 256), (8192, 256, 1), 0); del buf36  # reuse
            # Topologically Sorted Source Nodes: [view_55, linear, gelu, mul_45, mul_46, mul_47, exp_1, mul_48, mul_49, add_26, mul_50], Original ATen: [aten.view, aten.gelu, aten.gelu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_gelu_backward_view_3.run(buf39, addmm_1, 32768, stream=stream0)
            del addmm_1
            buf40 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [view_55, linear, gelu, mul_45, mul_46, mul_47, exp_1, mul_48, mul_49, add_26, mul_50, view_56, permute_54, mm_15], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf39, (128, 256), (256, 1), 0), primals_12, out=buf40)
            del primals_12
            buf41 = empty_strided_cuda((256, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_55, linear, gelu, mul_45, mul_46, mul_47, exp_1, mul_48, mul_49, add_26, mul_50, view_56, permute_55, mm_16], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf39, (256, 128), (1, 256), 0), view_11, out=buf41)
            del view_11
            buf42 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_55, linear, gelu, mul_45, mul_46, mul_47, exp_1, mul_48, mul_49, add_26, mul_50, view_56, sum_18], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_sum_view_4.run(buf39, buf42, 256, 128, stream=stream0)
            buf47 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [view_58, mul_52, mul_53, sum_19, mul_54, sum_20, mul_55, sub_16, sub_17, mul_56, add_27], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_backward_view_5.run(buf47, buf40, primals_10, mul_2, div_3, 128, 128, stream=stream0)
            del div_3
            del primals_10
            buf45 = empty_strided_cuda((128, ), (1, ), torch.float32)
            buf46 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_58, mul_57, sum_21, sum_22], Original ATen: [aten.view, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_layer_norm_backward_view_1.run(buf40, mul_2, buf45, buf46, 128, 128, stream=stream0)
            del mul_2
            buf48 = buf40; del buf40  # reuse
            buf51 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_58, clone_20, view_59, sum_23], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_sum_transpose_6.run(buf47, buf48, buf51, 128, 128, stream=stream0)
            buf49 = reinterpret_tensor(buf25, (128, 128), (128, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, permute_59, mm_17], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf48, primals_8, out=buf49)
            del primals_8
            buf50 = reinterpret_tensor(buf24, (128, 128), (128, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [permute_60, mm_18], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf48, (128, 128), (1, 128), 0), view_9, out=buf50)
            del buf48
            del view_9
            # Topologically Sorted Source Nodes: [view_61, permute_63, _scaled_dot_product_efficient_attention_backward_1], Original ATen: [aten.view, aten.permute, aten._scaled_dot_product_efficient_attention_backward]
            buf52 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf49, (4, 4, 32, 32), (128, 32, 512, 1), 0), view_6, view_7, view_8, None, getitem_2, getitem_3, getitem_4, getitem_5, 0.0, [True, True, True, False], True)
            del buf49
            del getitem_2
            del getitem_3
            del getitem_4
            del getitem_5
            del view_6
            del view_7
            del view_8
            buf53 = buf52[0]
            assert_size_stride(buf53, (4, 4, 32, 32), (4096, 32, 128, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf53, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf54 = buf52[1]
            assert_size_stride(buf54, (4, 4, 32, 32), (4096, 32, 128, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf54, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf55 = buf52[2]
            assert_size_stride(buf55, (4, 4, 32, 32), (4096, 32, 128, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf55, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf52
            buf56 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_default, clone_21, view_62, clone_22, view_63, clone_23, view_64, permute_64, clone_24, view_65, permute_65, clone_25, view_66, permute_66, clone_26, view_67, add_28, add_29, unsqueeze_5, permute_67, squeeze_3, clone_27, view_68, sum_24], Original ATen: [aten.select_backward, aten.clone, aten._unsafe_view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_7.run(buf55, buf54, buf53, buf56, 384, 128, stream=stream0)
            buf57 = empty_strided_cuda((32, 4, 3, 128), (1536, 384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_default, clone_21, view_62, clone_22, view_63, clone_23, view_64, permute_64, clone_24, view_65, permute_65, clone_25, view_66, permute_66, clone_26, view_67, add_28, add_29, unsqueeze_5, permute_67, squeeze_3, clone_27], Original ATen: [aten.select_backward, aten.clone, aten._unsafe_view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_select_backward_squeeze_transpose_unsqueeze_8.run(buf55, buf54, buf53, buf57, 49152, stream=stream0)
            del buf53
            del buf54
            buf58 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_default, clone_21, view_62, clone_22, view_63, clone_23, view_64, permute_64, clone_24, view_65, permute_65, clone_25, view_66, permute_66, clone_26, view_67, add_28, add_29, unsqueeze_5, permute_67, squeeze_3, clone_27, view_68, view_70, permute_68, mm_19], Original ATen: [aten.select_backward, aten.clone, aten._unsafe_view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf57, (384, 128), (1, 384), 0), view, out=buf58)
            del view
            buf59 = reinterpret_tensor(buf55, (128, 128), (128, 1), 0); del buf55  # reuse
            # Topologically Sorted Source Nodes: [full_default, clone_21, view_62, clone_22, view_63, clone_23, view_64, permute_64, clone_24, view_65, permute_65, clone_25, view_66, permute_66, clone_26, view_67, add_28, add_29, unsqueeze_5, permute_67, squeeze_3, clone_27, view_68, view_70, multi_head_attention_forward, permute_70, mm_20], Original ATen: [aten.select_backward, aten.clone, aten._unsafe_view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf57, (128, 384), (384, 1), 0), primals_7, out=buf59)
            del buf57
            del primals_7
            buf60 = empty_strided_cuda((4, 32, 1), (1, 4, 128), torch.float32)
            # Topologically Sorted Source Nodes: [view_71, permute_72, mul_59, sum_25], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_transpose_view_9.run(buf59, primals_4, buf60, 128, 128, stream=stream0)
            buf67 = reinterpret_tensor(buf39, (256, 128), (128, 1), 0); del buf39  # reuse
            # Topologically Sorted Source Nodes: [full_default_9], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_12.run(buf67, 32768, stream=stream0)
            buf64 = buf47; del buf47  # reuse
            # Topologically Sorted Source Nodes: [view_71, permute_72, mul_59, mul_60, hidden, layer_norm, mul_61, sum_26, mul_62, sub_19, sub_20, div_4, mul_63, add_30, full_default_6, eq_1, unsqueeze_7, where_2, full_default_9, index_put_1], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_layer_norm, aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_embedding_dense_backward_native_layer_norm_native_layer_norm_backward_transpose_view_13.run(buf64, buf59, primals_4, embedding, embedding_1, getitem_1, rsqrt, buf60, primals_1, buf67, 128, 128, stream=stream0)
            del primals_1
            del primals_4
            buf62 = reinterpret_tensor(buf60, (128, ), (1, ), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [view_71, permute_72, hidden, layer_norm, mul_64, sum_27], Original ATen: [aten.view, aten.transpose, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_14.run(buf59, embedding, embedding_1, getitem_1, rsqrt, buf62, 128, 128, stream=stream0)
            del embedding
            del embedding_1
            del getitem_1
            del rsqrt
            buf63 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_71, permute_72, sum_28], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_2.run(buf59, buf63, 128, 128, stream=stream0)
            buf65 = buf59; del buf59  # reuse
            # Topologically Sorted Source Nodes: [full_default_7], Original ATen: [aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_15.run(buf65, 16384, stream=stream0)
            # Topologically Sorted Source Nodes: [sum_29, view_72, eq, unsqueeze_6, full_default_6, where_1, full_default_7, index_put], Original ATen: [aten.sum, aten.view, aten.embedding_dense_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_sum_view_16.run(iota, buf64, buf65, 4096, stream=stream0)
            del buf64
            del iota
        return (None, buf67, buf65, buf62, buf63, reinterpret_tensor(buf56, (384, ), (1, ), 0), buf58, buf50, reinterpret_tensor(buf51, (128, ), (1, ), 0), buf45, buf46, buf41, reinterpret_tensor(buf42, (256, ), (1, ), 0), buf37, reinterpret_tensor(buf38, (128, ), (1, ), 0), buf33, buf34, reinterpret_tensor(buf27, (384, ), (1, ), 0), buf29, buf21, reinterpret_tensor(buf22, (128, ), (1, ), 0), buf16, buf17, buf12, reinterpret_tensor(buf13, (256, ), (1, ), 0), buf8, reinterpret_tensor(buf9, (128, ), (1, ), 0), buf5, buf6, buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((4, 32), (32, 1), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    iota = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.int64)
    embedding = rand_strided((4, 32, 128), (4096, 128, 1), device='cuda:0', dtype=torch.float32)
    embedding_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((4, 32, 1), (32, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((4, 32, 1), (32, 1, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_6 = rand_strided((4, 4, 32, 32), (128, 32, 512, 1), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((4, 4, 32, 32), (128, 32, 512, 1), device='cuda:0', dtype=torch.float32)
    view_8 = rand_strided((4, 4, 32, 32), (128, 32, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((4, 4, 32, 32), (4096, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((4, 4, 32), (128, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_4 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_5 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_9 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_2 = rand_strided((4, 32, 128), (4096, 128, 1), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((4, 32, 128), (4096, 128, 1), device='cuda:0', dtype=torch.float32)
    view_15 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((4, 4, 32, 32), (128, 32, 512, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((4, 4, 32, 32), (128, 32, 512, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((4, 4, 32, 32), (128, 32, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_10 = rand_strided((4, 4, 32, 32), (4096, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((4, 4, 32), (128, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_12 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_13 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_24 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((4, 32, 128), (4096, 128, 1), device='cuda:0', dtype=torch.float32)
    view_26 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    view_28 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_14 = rand_strided((4, 32, 128), (4096, 128, 1), device='cuda:0', dtype=torch.float32)
    view_30 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((4, 32, 1), (32, 1, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((4, 32, 1), (32, 1, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((4, 32, 1), (32, 1, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((4, 32, 1), (32, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((4, 32, 256), (8192, 256, 1), device='cuda:0', dtype=torch.float32)
    return [primals_1, primals_4, primals_7, primals_8, primals_10, primals_12, primals_14, primals_16, primals_19, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, iota, embedding, embedding_1, getitem_1, rsqrt, view, view_6, view_7, view_8, getitem_2, getitem_3, getitem_4, getitem_5, view_9, mul_2, view_11, addmm_1, view_13, mul_7, view_15, view_21, view_22, view_23, getitem_10, getitem_11, getitem_12, getitem_13, view_24, mul_9, view_26, addmm_4, view_28, mul_14, view_30, div, div_1, div_2, div_3, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
