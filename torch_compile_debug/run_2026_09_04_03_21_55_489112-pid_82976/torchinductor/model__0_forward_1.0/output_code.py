# AOT ID: ['0_forward']
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


# kernel path: /tmp/torchinductor_binly/tmpqlvsuo7k/if/cif4ejmgvn6fycj4rosp7ozibaemw4kqjhtijaqlo33h4rnpnkfk.py
# Topologically Sorted Source Nodes: [positions], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   positions => iota
# Graph fragment:
#   %iota : Tensor "i64[32][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   return %iota
triton_poi_fused_arange_0 = async_compile.triton('triton_poi_fused_arange_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmpqlvsuo7k/7u/c7ulsm2mfseafsk2wsnkfycy5net6nfsh6bdptszno6crevkgiku.py
# Topologically Sorted Source Nodes: [embedding_1], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   embedding_1 => embedding_1
# Graph fragment:
#   %primals_3 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %embedding_1 : Tensor "f32[32, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_3, %iota), kwargs = {})
#   return %embedding_1
triton_poi_fused_embedding_1 = async_compile.triton('triton_poi_fused_embedding_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 49152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmpqlvsuo7k/3a/c3aarvhpm7xkpydks2km5m4kfdrglybe33b4eiqrcxl6atlnlrux.py
# Topologically Sorted Source Nodes: [embedding, hidden, layer_norm, query, multi_head_attention_forward], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   embedding => embedding
#   hidden => add
#   layer_norm => add_1, add_2, mul, mul_1, rsqrt, sub_1, var_mean
#   multi_head_attention_forward => clone
#   query => permute
# Graph fragment:
#   %primals_1 : Tensor "i64[4, 32][32, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %primals_2 : Tensor "f32[256, 128][128, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %embedding : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=embedding]
#   %embedding_1 : Tensor "f32[32, 128][128, 1]cuda:0" = PlaceHolder[target=embedding_1]
#   %buf4 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=buf4]
#   %getitem_1 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %rsqrt : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %primals_4 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_4]
#   %primals_5 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_5]
#   %embedding : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_2, %primals_1), kwargs = {})
#   %add : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub_1 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %mul : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_4), kwargs = {})
#   %add_2 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_5), kwargs = {})
#   %permute : Tensor "f32[32, 4, 128][128, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_2, [1, 0, 2]), kwargs = {})
#   %clone : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   return %embedding,%getitem_1,%buf4,%rsqrt,%clone
triton_per_fused_add_clone_embedding_native_layer_norm_transpose_2 = async_compile.triton('triton_per_fused_add_clone_embedding_native_layer_norm_transpose_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_embedding_native_layer_norm_transpose_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 4, 'num_reduction': 4, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3072, 'r0_': 279552}}
)
@triton.jit
def triton_per_fused_add_clone_embedding_native_layer_norm_transpose_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x0 = xindex
    r0_1 = r0_index
    x2 = (xindex % 32)
    x3 = xindex // 32
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 128*x2), xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 256, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 256)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 256")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 128*tmp4), xmask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([1, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tmp25 = tl.full([1, 1], 128.0, tl.float32)
    tmp26 = (tmp24 / tmp25)
    tmp27 = tl.full([1, 1], 1e-05, tl.float32)
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp8 - tmp18
    tmp31 = tmp30 * tmp29
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(out_ptr0 + (r0_1 + 128*x0), tmp6, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp29, xmask)
    tl.store(out_ptr2 + (r0_1 + 128*x3 + 512*x2), tmp35, xmask)
    tl.store(out_ptr1 + (x0), tmp18, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmpqlvsuo7k/wf/cwf7tou5bargvct6gwsaesbpnlalqsgbi432uohngqzmgdc2sfvu.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_3, clone_1, permute_2, squeeze, unsqueeze_2, view_1, view_2
# Graph fragment:
#   %mm : Tensor "f32[128, 384][384, 1]cuda:0" = PlaceHolder[target=mm]
#   %primals_6 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=primals_6]
#   %view_1 : Tensor "f32[32, 4, 384][1536, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [32, 4, 384]), kwargs = {})
#   %add_3 : Tensor "f32[32, 4, 384][1536, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_6), kwargs = {})
#   %view_2 : Tensor "f32[32, 4, 3, 128][1536, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [32, 4, 3, 128]), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 32, 4, 3, 128][49152, 1536, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 0), kwargs = {})
#   %permute_2 : Tensor "f32[3, 32, 4, 1, 128][128, 1536, 384, 49152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_2, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 32, 4, 128][128, 1536, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_2, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3 = async_compile.triton('triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 591360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 128)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 384*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmpqlvsuo7k/gg/cgg2jqsus3buktc7bfoxpodxlxf2kfjkmas3zqnkidzclbwow3fe.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_3, clone_1, permute_2, permute_3, select, squeeze, unsqueeze_2, view_1, view_2, view_3, view_6
# Graph fragment:
#   %clone_1 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0" = PlaceHolder[target=clone_1]
#   %view_1 : Tensor "f32[32, 4, 384][1536, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [32, 4, 384]), kwargs = {})
#   %add_3 : Tensor "f32[32, 4, 384][1536, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_6), kwargs = {})
#   %view_2 : Tensor "f32[32, 4, 3, 128][1536, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [32, 4, 3, 128]), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 32, 4, 3, 128][49152, 1536, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 0), kwargs = {})
#   %permute_2 : Tensor "f32[3, 32, 4, 1, 128][128, 1536, 384, 49152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_2, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 32, 4, 128][128, 1536, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_2, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 0), kwargs = {})
#   %view_3 : Tensor "f32[32, 16, 32][512, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [32, 16, 32]), kwargs = {})
#   %permute_3 : Tensor "f32[16, 32, 32][32, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [1, 0, 2]), kwargs = {})
#   %view_6 : Tensor "f32[4, 4, 32, 32][128, 32, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_3, [4, 4, 32, 32]), kwargs = {})
#   return %view_6
triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_4 = async_compile.triton('triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 196608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmpqlvsuo7k/6w/c6wf6eetypylmnokinqsvw3dsoohwg4mfvihak2j375yhtqlz23o.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_3, clone_1, permute_2, permute_4, select_1, squeeze, unsqueeze_2, view_1, view_2, view_4, view_7
# Graph fragment:
#   %clone_1 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0" = PlaceHolder[target=clone_1]
#   %view_1 : Tensor "f32[32, 4, 384][1536, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [32, 4, 384]), kwargs = {})
#   %add_3 : Tensor "f32[32, 4, 384][1536, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_6), kwargs = {})
#   %view_2 : Tensor "f32[32, 4, 3, 128][1536, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [32, 4, 3, 128]), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 32, 4, 3, 128][49152, 1536, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 0), kwargs = {})
#   %permute_2 : Tensor "f32[3, 32, 4, 1, 128][128, 1536, 384, 49152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_2, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 32, 4, 128][128, 1536, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_2, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select_1 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 1), kwargs = {})
#   %view_4 : Tensor "f32[32, 16, 32][512, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [32, 16, 32]), kwargs = {})
#   %permute_4 : Tensor "f32[16, 32, 32][32, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [1, 0, 2]), kwargs = {})
#   %view_7 : Tensor "f32[4, 4, 32, 32][128, 32, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_4, [4, 4, 32, 32]), kwargs = {})
#   return %view_7
triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_5 = async_compile.triton('triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 196608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (16384 + x4), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmpqlvsuo7k/z5/cz5ubdowmbtuvqyqfil526tm7jjuy5kykwymhs7t3jid55fywr4k.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_3, clone_1, permute_2, permute_5, select_2, squeeze, unsqueeze_2, view_1, view_2, view_5, view_8
# Graph fragment:
#   %clone_1 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0" = PlaceHolder[target=clone_1]
#   %view_1 : Tensor "f32[32, 4, 384][1536, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [32, 4, 384]), kwargs = {})
#   %add_3 : Tensor "f32[32, 4, 384][1536, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_6), kwargs = {})
#   %view_2 : Tensor "f32[32, 4, 3, 128][1536, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [32, 4, 3, 128]), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 32, 4, 3, 128][49152, 1536, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 0), kwargs = {})
#   %permute_2 : Tensor "f32[3, 32, 4, 1, 128][128, 1536, 384, 49152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_2, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 32, 4, 128][128, 1536, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_2, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 32, 4, 128][16384, 512, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select_2 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 2), kwargs = {})
#   %view_5 : Tensor "f32[32, 16, 32][512, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_2, [32, 16, 32]), kwargs = {})
#   %permute_5 : Tensor "f32[16, 32, 32][32, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [1, 0, 2]), kwargs = {})
#   %view_8 : Tensor "f32[4, 4, 32, 32][128, 32, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_5, [4, 4, 32, 32]), kwargs = {})
#   return %view_8
triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_6 = async_compile.triton('triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 196608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (32768 + x4), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmpqlvsuo7k/zs/czsi4xq6iuk6kwvwfftbw3dp46btwc4cod72q3e56xx3wusdypnz.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.permute, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_2, permute_6
# Graph fragment:
#   %getitem_2 : Tensor "f32[4, 4, 32, 32][4096, 32, 128, 1]cuda:0" = PlaceHolder[target=getitem_2]
#   %permute_6 : Tensor "f32[32, 4, 4, 32][128, 4096, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_2, [2, 0, 1, 3]), kwargs = {})
#   %clone_2 : Tensor "f32[32, 4, 4, 32][512, 128, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_6,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_poi_fused_clone_permute_7 = async_compile.triton('triton_poi_fused_clone_permute_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_permute_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 196608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_permute_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 4)
    x2 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 4096*x1), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmpqlvsuo7k/ft/cft2ygjmx5c2k6y3fya355yzvtflepag6p2g7y4frio4s2sicep3.py
# Topologically Sorted Source Nodes: [hidden, multi_head_attention_forward, x, x_1, layer_norm_1, div_3], Original ATen: [aten.add, aten.addmm, aten.view, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   div_3 => div_3
#   hidden => add
#   layer_norm_1 => add_5, add_6, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
#   multi_head_attention_forward => add_tensor_3, view_10
#   x => permute_8
#   x_1 => add_4
# Graph fragment:
#   %embedding : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=embedding]
#   %embedding_1 : Tensor "f32[32, 128][128, 1]cuda:0" = PlaceHolder[target=embedding_1]
#   %primals_9 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_9]
#   %mm_default_3 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_3]
#   %getitem_7 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=getitem_7]
#   %buf21 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=buf21]
#   %mul_2 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=mul_2]
#   %primals_10 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_10]
#   %primals_11 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_11]
#   %add : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_tensor_3 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_9, %mm_default_3), kwargs = {})
#   %view_10 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_3, [32, 4, 128]), kwargs = {})
#   %permute_8 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [1, 0, 2]), kwargs = {})
#   %add_4 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %permute_8), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_5 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %sub_2 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_7), kwargs = {})
#   %mul_2 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_10), kwargs = {})
#   %add_6 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_11), kwargs = {})
#   %div_3 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 128), kwargs = {})
#   return %getitem_7,%buf21,%mul_2,%add_6,%div_3
triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_8 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 3, 'num_reduction': 4, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024, 'r0_': 411136}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_2), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (r0_2 + 128*x1 + 512*x0), xmask, other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr5 + (r0_2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp14 = tl.full([1, 1], 128, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = (tmp13 / tmp15)
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = tmp6 - tmp16
    tmp24 = tl.full([1, 1], 128.0, tl.float32)
    tmp25 = (tmp22 / tmp24)
    tmp26 = tl.full([1, 1], 1e-05, tl.float32)
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full([1, 1], 0.0078125, tl.float32)
    tmp35 = tmp28 * tmp34
    tl.store(out_ptr2 + (r0_2 + 128*x3), tmp29, xmask)
    tl.store(out_ptr3 + (r0_2 + 128*x3), tmp33, xmask)
    tl.store(out_ptr4 + (x3), tmp35, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmpqlvsuo7k/on/consbhu3y5lbksgptbjkqj6txnrgomtzcla5r6bt4jhu2j5qeidv.py
# Topologically Sorted Source Nodes: [linear, gelu], Original ATen: [aten.view, aten.gelu]
# Source node to ATen node mapping:
#   gelu => add_7, erf, mul_4, mul_5, mul_6
#   linear => view_12
# Graph fragment:
#   %addmm_1 : Tensor "f32[128, 256][256, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %view_12 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [4, 32, 256]), kwargs = {})
#   %mul_4 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_12, 0.5), kwargs = {})
#   %mul_5 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_12, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_5,), kwargs = {})
#   %add_7 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_6 : Tensor "f32[4, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_7), kwargs = {})
#   return %mul_6
triton_poi_fused_gelu_view_9 = async_compile.triton('triton_poi_fused_gelu_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 393216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_view_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0.5, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmpqlvsuo7k/pw/cpwwivmd4ugvi4lr3oi72crk6eep2pg7m25boh2rymprl5se3zei.py
# Topologically Sorted Source Nodes: [hidden, multi_head_attention_forward, x, x_1, x_2, x_3, layer_norm_2, query_1, multi_head_attention_forward_1, div_2], Original ATen: [aten.add, aten.addmm, aten.view, aten.transpose, aten.native_layer_norm, aten.clone, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   div_2 => div_2
#   hidden => add
#   layer_norm_2 => add_10, add_9, mul_7, mul_8, rsqrt_2, sub_3, var_mean_2
#   multi_head_attention_forward => add_tensor_3, view_10
#   multi_head_attention_forward_1 => clone_6
#   query_1 => permute_11
#   x => permute_8
#   x_1 => add_4
#   x_2 => add_tensor_2, view_14
#   x_3 => add_8
# Graph fragment:
#   %embedding : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=embedding]
#   %embedding_1 : Tensor "f32[32, 128][128, 1]cuda:0" = PlaceHolder[target=embedding_1]
#   %primals_9 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_9]
#   %mm_default_3 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_3]
#   %primals_15 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_15]
#   %mm_default_2 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_2]
#   %add_8 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=add_8]
#   %getitem_9 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=getitem_9]
#   %buf30 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=buf30]
#   %mul_7 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=mul_7]
#   %primals_16 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_16]
#   %primals_17 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_17]
#   %add : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_tensor_3 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_9, %mm_default_3), kwargs = {})
#   %view_10 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_3, [32, 4, 128]), kwargs = {})
#   %permute_8 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [1, 0, 2]), kwargs = {})
#   %add_4 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %permute_8), kwargs = {})
#   %add_tensor_2 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_15, %mm_default_2), kwargs = {})
#   %view_14 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_2, [4, 32, 128]), kwargs = {})
#   %add_8 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_14), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %sub_3 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_9), kwargs = {})
#   %mul_7 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_8 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %primals_16), kwargs = {})
#   %add_10 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %primals_17), kwargs = {})
#   %permute_11 : Tensor "f32[32, 4, 128][128, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_10, [1, 0, 2]), kwargs = {})
#   %clone_6 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_11,), kwargs = {memory_format: torch.contiguous_format})
#   %div_2 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 128), kwargs = {})
#   return %add_8,%getitem_9,%buf30,%mul_7,%div_2,%clone_6
triton_per_fused_add_addmm_clone_native_layer_norm_native_layer_norm_backward_transpose_view_10 = async_compile.triton('triton_per_fused_add_addmm_clone_native_layer_norm_native_layer_norm_backward_transpose_view_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_clone_native_layer_norm_native_layer_norm_backward_transpose_view_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 8, 'num_store': 4, 'num_reduction': 4, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024, 'r0_': 608256}}
)
@triton.jit
def triton_per_fused_add_addmm_clone_native_layer_norm_native_layer_norm_backward_transpose_view_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_2), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (r0_2 + 128*x1 + 512*x0), xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r0_2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr0 + (r0_2 + 128*x3), xmask, other=0.0)
    tmp36 = tl.load(in_ptr5 + (r0_2), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr6 + (r0_2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = tl.full([1, 1], 128, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = (tmp17 / tmp19)
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp25 = tl.where(xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None].to(tl.float32)
    tmp27 = tmp10 - tmp20
    tmp28 = tl.full([1, 1], 128.0, tl.float32)
    tmp29 = (tmp26 / tmp28)
    tmp30 = tl.full([1, 1], 1e-05, tl.float32)
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp34 = tl.full([1, 1], 0.0078125, tl.float32)
    tmp35 = tmp32 * tmp34
    tmp37 = tmp33 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r0_2 + 128*x3), tmp10, xmask)
    tl.store(out_ptr2 + (r0_2 + 128*x3), tmp33, xmask)
    tl.store(out_ptr3 + (x3), tmp35, xmask)
    tl.store(out_ptr4 + (r0_2 + 128*x1 + 512*x0), tmp39, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmpqlvsuo7k/ld/cld2e662zekvp7rah5sfqnkfo2wxmlirunh3bayugqbfvynf6wtp.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward_1, x_4, x_5, layer_norm_3, div_1], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   div_1 => div_1
#   layer_norm_3 => add_13, add_14, mul_10, mul_9, rsqrt_3, sub_4, var_mean_3
#   multi_head_attention_forward_1 => add_tensor_1, view_25
#   x_4 => permute_19
#   x_5 => add_12
# Graph fragment:
#   %add_8 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=add_8]
#   %primals_21 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_21]
#   %mm_default_1 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %getitem_15 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=getitem_15]
#   %buf47 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=buf47]
#   %mul_9 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=mul_9]
#   %primals_22 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_22]
#   %primals_23 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_23]
#   %add_tensor_1 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_21, %mm_default_1), kwargs = {})
#   %view_25 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [32, 4, 128]), kwargs = {})
#   %permute_19 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_25, [1, 0, 2]), kwargs = {})
#   %add_12 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %permute_19), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_12, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_13 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %sub_4 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %getitem_15), kwargs = {})
#   %mul_9 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_10 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %primals_22), kwargs = {})
#   %add_14 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %primals_23), kwargs = {})
#   %div_1 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 128), kwargs = {})
#   return %getitem_15,%buf47,%mul_9,%add_14,%div_1
triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_11 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 4, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024, 'r0_': 394752}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (r0_2 + 128*x1 + 512*x0), xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (r0_2), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 128.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full([1, 1], 0.0078125, tl.float32)
    tmp33 = tmp26 * tmp32
    tl.store(out_ptr2 + (r0_2 + 128*x3), tmp27, xmask)
    tl.store(out_ptr3 + (r0_2 + 128*x3), tmp31, xmask)
    tl.store(out_ptr4 + (x3), tmp33, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_binly/tmpqlvsuo7k/5b/c5bo7heakn7ahz3ub4sva6enb5csab7ohn74c75ag5bqpmpy3tr2.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward_1, x_4, x_5, x_6, x_7, layer_norm_4, div], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   div => div
#   layer_norm_4 => add_17, add_18, mul_14, mul_15, rsqrt_4, sub_5, var_mean_4
#   multi_head_attention_forward_1 => add_tensor_1, view_25
#   x_4 => permute_19
#   x_5 => add_12
#   x_6 => add_tensor, view_29
#   x_7 => add_16
# Graph fragment:
#   %add_8 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=add_8]
#   %primals_21 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_21]
#   %mm_default_1 : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %primals_27 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_27]
#   %mm_default : Tensor "f32[128, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %add_16 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=add_16]
#   %getitem_17 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=getitem_17]
#   %buf56 : Tensor "f32[4, 32, 1][32, 1, 128]cuda:0" = PlaceHolder[target=buf56]
#   %mul_14 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0" = PlaceHolder[target=mul_14]
#   %primals_28 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_28]
#   %primals_29 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_29]
#   %add_tensor_1 : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_21, %mm_default_1), kwargs = {})
#   %view_25 : Tensor "f32[32, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [32, 4, 128]), kwargs = {})
#   %permute_19 : Tensor "f32[4, 32, 128][128, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_25, [1, 0, 2]), kwargs = {})
#   %add_12 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %permute_19), kwargs = {})
#   %add_tensor : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_27, %mm_default), kwargs = {})
#   %view_29 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [4, 32, 128]), kwargs = {})
#   %add_16 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %view_29), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_16, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_17 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %sub_5 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, %getitem_17), kwargs = {})
#   %mul_14 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_4), kwargs = {})
#   %mul_15 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %primals_28), kwargs = {})
#   %add_18 : Tensor "f32[4, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %primals_29), kwargs = {})
#   %div : Tensor "f32[4, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 128), kwargs = {})
#   return %add_16,%getitem_17,%buf56,%mul_14,%add_18,%div
triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_12 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 3, 'num_reduction': 4, 'backend_hash': 'D2B7050BEBFFCFFED69FAD412C73FDD34C8A37458510E14856046944E93DFE6F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024, 'r0_': 460800}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 128*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_2), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (r0_2 + 128*x1 + 512*x0), xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (r0_2 + 128*x3), xmask, other=0.0)
    tmp32 = tl.load(in_ptr4 + (r0_2), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (r0_2), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([1, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tmp25 = tmp8 - tmp18
    tmp26 = tl.full([1, 1], 128.0, tl.float32)
    tmp27 = (tmp24 / tmp26)
    tmp28 = tl.full([1, 1], 1e-05, tl.float32)
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tl.full([1, 1], 0.0078125, tl.float32)
    tmp37 = tmp30 * tmp36
    tl.store(in_out_ptr0 + (r0_2 + 128*x3), tmp31, xmask)
    tl.store(out_ptr2 + (r0_2 + 128*x3), tmp35, xmask)
    tl.store(out_ptr3 + (x3), tmp37, xmask)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30 = args
        args.clear()
        assert_size_stride(primals_1, (4, 32), (32, 1))
        assert_size_stride(primals_2, (256, 128), (128, 1))
        assert_size_stride(primals_3, (128, 128), (128, 1))
        assert_size_stride(primals_4, (128, ), (1, ))
        assert_size_stride(primals_5, (128, ), (1, ))
        assert_size_stride(primals_6, (384, ), (1, ))
        assert_size_stride(primals_7, (384, 128), (128, 1))
        assert_size_stride(primals_8, (128, 128), (128, 1))
        assert_size_stride(primals_9, (128, ), (1, ))
        assert_size_stride(primals_10, (128, ), (1, ))
        assert_size_stride(primals_11, (128, ), (1, ))
        assert_size_stride(primals_12, (256, 128), (128, 1))
        assert_size_stride(primals_13, (256, ), (1, ))
        assert_size_stride(primals_14, (128, 256), (256, 1))
        assert_size_stride(primals_15, (128, ), (1, ))
        assert_size_stride(primals_16, (128, ), (1, ))
        assert_size_stride(primals_17, (128, ), (1, ))
        assert_size_stride(primals_18, (384, ), (1, ))
        assert_size_stride(primals_19, (384, 128), (128, 1))
        assert_size_stride(primals_20, (128, 128), (128, 1))
        assert_size_stride(primals_21, (128, ), (1, ))
        assert_size_stride(primals_22, (128, ), (1, ))
        assert_size_stride(primals_23, (128, ), (1, ))
        assert_size_stride(primals_24, (256, 128), (128, 1))
        assert_size_stride(primals_25, (256, ), (1, ))
        assert_size_stride(primals_26, (128, 256), (256, 1))
        assert_size_stride(primals_27, (128, ), (1, ))
        assert_size_stride(primals_28, (128, ), (1, ))
        assert_size_stride(primals_29, (128, ), (1, ))
        assert_size_stride(primals_30, (256, 128), (128, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((32, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [positions], Original ATen: [aten.arange]
            stream0 = get_raw_stream(0)
            triton_poi_fused_arange_0.run(buf0, 32, stream=stream0)
            buf2 = empty_strided_cuda((32, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [embedding_1], Original ATen: [aten.embedding]
            stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_1.run(primals_3, buf2, 4096, stream=stream0)
            del primals_3
            buf1 = empty_strided_cuda((4, 32, 128), (4096, 128, 1), torch.float32)
            buf3 = empty_strided_cuda((4, 32, 1), (32, 1, 1), torch.float32)
            buf4 = empty_strided_cuda((4, 32, 1), (32, 1, 128), torch.float32)
            buf6 = reinterpret_tensor(buf4, (4, 32, 1), (32, 1, 1), 0); del buf4  # reuse
            buf7 = empty_strided_cuda((32, 4, 128), (512, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [embedding, hidden, layer_norm, query, multi_head_attention_forward], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_clone_embedding_native_layer_norm_transpose_2.run(buf6, primals_1, primals_2, buf2, primals_4, primals_5, buf1, buf3, buf7, 128, 128, stream=stream0)
            del primals_2
            del primals_5
            buf8 = empty_strided_cuda((128, 384), (384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden, layer_norm, query, multi_head_attention_forward], Original ATen: [aten.add, aten.native_layer_norm, aten.transpose, aten.t, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf7, (128, 128), (128, 1), 0), reinterpret_tensor(primals_7, (128, 384), (1, 128), 0), out=buf8)
            buf9 = empty_strided_cuda((3, 32, 4, 128), (16384, 512, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3.run(buf8, primals_6, buf9, 49152, stream=stream0)
            del buf8
            del primals_6
            buf10 = empty_strided_cuda((4, 4, 32, 32), (128, 32, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_4.run(buf9, buf10, 16384, stream=stream0)
            buf11 = empty_strided_cuda((4, 4, 32, 32), (128, 32, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_5.run(buf9, buf11, 16384, stream=stream0)
            buf12 = empty_strided_cuda((4, 4, 32, 32), (128, 32, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf9, buf12, 16384, stream=stream0)
            del buf9
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
            buf13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf10, buf11, buf12, None, True, 0.0, True)
            buf14 = buf13[0]
            assert_size_stride(buf14, (4, 4, 32, 32), (4096, 32, 128, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf14, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf15 = buf13[1]
            assert_size_stride(buf15, (4, 4, 32), (128, 32, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf15, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf16 = buf13[2]
            assert_size_stride(buf16, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf16, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf17 = buf13[3]
            assert_size_stride(buf17, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf17, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf13
            buf18 = empty_strided_cuda((32, 4, 4, 32), (512, 128, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.permute, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_permute_7.run(buf14, buf18, 16384, stream=stream0)
            buf19 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.permute, aten.clone, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf18, (128, 128), (128, 1), 0), reinterpret_tensor(primals_8, (128, 128), (1, 128), 0), out=buf19)
            buf23 = empty_strided_cuda((4, 32, 128), (4096, 128, 1), torch.float32)
            buf24 = empty_strided_cuda((4, 32, 128), (4096, 128, 1), torch.float32)
            buf64 = empty_strided_cuda((4, 32, 1), (32, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden, multi_head_attention_forward, x, x_1, layer_norm_1, div_3], Original ATen: [aten.add, aten.addmm, aten.view, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_8.run(buf1, buf2, primals_9, buf19, primals_10, primals_11, buf23, buf24, buf64, 128, 128, stream=stream0)
            del primals_11
            buf25 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_1, linear], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(primals_13, reinterpret_tensor(buf24, (128, 128), (128, 1), 0), reinterpret_tensor(primals_12, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf25)
            del primals_13
            buf26 = empty_strided_cuda((4, 32, 256), (8192, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, gelu], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_9.run(buf25, buf26, 32768, stream=stream0)
            buf27 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, gelu, x_2], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf26, (128, 256), (256, 1), 0), reinterpret_tensor(primals_14, (256, 128), (1, 256), 0), out=buf27)
            buf28 = reinterpret_tensor(buf27, (4, 32, 128), (4096, 128, 1), 0); del buf27  # reuse
            buf32 = empty_strided_cuda((4, 32, 128), (4096, 128, 1), torch.float32)
            buf63 = empty_strided_cuda((4, 32, 1), (32, 1, 1), torch.float32)
            buf33 = empty_strided_cuda((32, 4, 128), (512, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden, multi_head_attention_forward, x, x_1, x_2, x_3, layer_norm_2, query_1, multi_head_attention_forward_1, div_2], Original ATen: [aten.add, aten.addmm, aten.view, aten.transpose, aten.native_layer_norm, aten.clone, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_clone_native_layer_norm_native_layer_norm_backward_transpose_view_10.run(buf28, buf1, buf2, primals_9, buf19, primals_15, primals_16, primals_17, buf32, buf63, buf33, 128, 128, stream=stream0)
            del primals_15
            del primals_17
            del primals_9
            buf34 = empty_strided_cuda((128, 384), (384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_2, query_1, multi_head_attention_forward_1], Original ATen: [aten.native_layer_norm, aten.transpose, aten.t, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf33, (128, 128), (128, 1), 0), reinterpret_tensor(primals_19, (128, 384), (1, 128), 0), out=buf34)
            buf35 = empty_strided_cuda((3, 32, 4, 128), (16384, 512, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_3.run(buf34, primals_18, buf35, 49152, stream=stream0)
            del buf34
            del primals_18
            buf36 = reinterpret_tensor(buf19, (4, 4, 32, 32), (128, 32, 512, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_4.run(buf35, buf36, 16384, stream=stream0)
            buf37 = empty_strided_cuda((4, 4, 32, 32), (128, 32, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_5.run(buf35, buf37, 16384, stream=stream0)
            buf38 = empty_strided_cuda((4, 4, 32, 32), (128, 32, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf35, buf38, 16384, stream=stream0)
            del buf35
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._scaled_dot_product_efficient_attention]
            buf39 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf36, buf37, buf38, None, True, 0.0, True)
            buf40 = buf39[0]
            assert_size_stride(buf40, (4, 4, 32, 32), (4096, 32, 128, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf40, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf41 = buf39[1]
            assert_size_stride(buf41, (4, 4, 32), (128, 32, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf41, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf42 = buf39[2]
            assert_size_stride(buf42, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf42, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf43 = buf39[3]
            assert_size_stride(buf43, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf43, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf39
            buf44 = empty_strided_cuda((32, 4, 4, 32), (512, 128, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.permute, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_permute_7.run(buf40, buf44, 16384, stream=stream0)
            buf45 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.permute, aten.clone, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf44, (128, 128), (128, 1), 0), reinterpret_tensor(primals_20, (128, 128), (1, 128), 0), out=buf45)
            buf49 = empty_strided_cuda((4, 32, 128), (4096, 128, 1), torch.float32)
            buf50 = empty_strided_cuda((4, 32, 128), (4096, 128, 1), torch.float32)
            buf62 = empty_strided_cuda((4, 32, 1), (32, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1, x_4, x_5, layer_norm_3, div_1], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_11.run(buf28, primals_21, buf45, primals_22, primals_23, buf49, buf50, buf62, 128, 128, stream=stream0)
            del primals_23
            buf51 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_3, linear_2], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(primals_25, reinterpret_tensor(buf50, (128, 128), (128, 1), 0), reinterpret_tensor(primals_24, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf51)
            del primals_25
            buf52 = empty_strided_cuda((4, 32, 256), (8192, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2, gelu_1], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_9.run(buf51, buf52, 32768, stream=stream0)
            buf53 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2, gelu_1, x_6], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf52, (128, 256), (256, 1), 0), reinterpret_tensor(primals_26, (256, 128), (1, 256), 0), out=buf53)
            buf54 = buf28; del buf28  # reuse
            buf58 = buf54; del buf54  # reuse
            buf59 = empty_strided_cuda((4, 32, 128), (4096, 128, 1), torch.float32)
            buf61 = empty_strided_cuda((4, 32, 1), (32, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1, x_4, x_5, x_6, x_7, layer_norm_4, div], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_transpose_view_12.run(buf58, primals_21, buf45, primals_27, buf53, primals_28, primals_29, buf59, buf61, 128, 128, stream=stream0)
            del buf45
            del buf53
            del primals_21
            del primals_27
            del primals_29
            buf60 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_4, linear_4], Original ATen: [aten.native_layer_norm, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf59, (128, 128), (128, 1), 0), reinterpret_tensor(primals_30, (128, 256), (1, 128), 0), out=buf60)
        return (reinterpret_tensor(buf60, (4, 32, 256), (8192, 256, 1), 0), primals_1, primals_4, primals_7, primals_8, primals_10, primals_12, primals_14, primals_16, primals_19, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, buf0, buf1, buf2, buf3, buf6, reinterpret_tensor(buf7, (128, 128), (128, 1), 0), buf10, buf11, buf12, buf14, buf15, buf16, buf17, reinterpret_tensor(buf18, (128, 128), (128, 1), 0), buf23, reinterpret_tensor(buf24, (128, 128), (128, 1), 0), buf25, reinterpret_tensor(buf26, (128, 256), (256, 1), 0), buf32, reinterpret_tensor(buf33, (128, 128), (128, 1), 0), buf36, buf37, buf38, buf40, buf41, buf42, buf43, reinterpret_tensor(buf44, (128, 128), (128, 1), 0), buf49, reinterpret_tensor(buf50, (128, 128), (128, 1), 0), buf51, reinterpret_tensor(buf52, (128, 256), (256, 1), 0), buf58, reinterpret_tensor(buf59, (128, 128), (128, 1), 0), buf61, buf62, buf63, buf64, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((4, 32), (32, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
