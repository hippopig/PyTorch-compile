class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4, 32]", primals_2: "f32[256, 128]", primals_3: "f32[128, 128]", primals_4: "f32[128]", primals_5: "f32[128]", primals_6: "f32[384]", primals_7: "f32[384, 128]", primals_8: "f32[128, 128]", primals_9: "f32[128]", primals_10: "f32[128]", primals_11: "f32[128]", primals_12: "f32[256, 128]", primals_13: "f32[256]", primals_14: "f32[128, 256]", primals_15: "f32[128]", primals_16: "f32[128]", primals_17: "f32[128]", primals_18: "f32[384]", primals_19: "f32[384, 128]", primals_20: "f32[128, 128]", primals_21: "f32[128]", primals_22: "f32[128]", primals_23: "f32[128]", primals_24: "f32[256, 128]", primals_25: "f32[256]", primals_26: "f32[128, 256]", primals_27: "f32[128]", primals_28: "f32[128]", primals_29: "f32[128]", primals_30: "f32[256, 128]"):
        # File: /home/binly/workspace/PyTorch-compile/model.py:47 in forward, code: positions = torch.arange(sequence_length, device=token_ids.device)
        iota: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/binly/workspace/PyTorch-compile/model.py:48 in forward, code: hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        embedding: "f32[4, 32, 128]" = torch.ops.aten.embedding.default(primals_2, primals_1);  primals_2 = None
        embedding_1: "f32[32, 128]" = torch.ops.aten.embedding.default(primals_3, iota);  primals_3 = None
        add: "f32[4, 32, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1)

        # File: /home/binly/workspace/PyTorch-compile/model.py:58 in forward, code: hidden = self.transformer(hidden, mask=causal_mask, is_causal=True)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 32, 1]" = var_mean[0]
        getitem_1: "f32[4, 32, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[4, 32, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[4, 32, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub_1: "f32[4, 32, 128]" = torch.ops.aten.sub.Tensor(add, getitem_1)
        mul: "f32[4, 32, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_1: "f32[4, 32, 128]" = torch.ops.aten.mul.Tensor(mul, primals_4);  mul = None
        add_2: "f32[4, 32, 128]" = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None
        permute: "f32[32, 4, 128]" = torch.ops.aten.permute.default(add_2, [1, 0, 2]);  add_2 = None
        permute_1: "f32[128, 384]" = torch.ops.aten.permute.default(primals_7, [1, 0])
        clone: "f32[32, 4, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view: "f32[128, 128]" = torch.ops.aten.reshape.default(clone, [128, 128]);  clone = None
        mm: "f32[128, 384]" = torch.ops.aten.mm.default(view, permute_1);  permute_1 = None
        view_1: "f32[32, 4, 384]" = torch.ops.aten.reshape.default(mm, [32, 4, 384]);  mm = None
        add_3: "f32[32, 4, 384]" = torch.ops.aten.add.Tensor(view_1, primals_6);  view_1 = primals_6 = None
        view_2: "f32[32, 4, 3, 128]" = torch.ops.aten.reshape.default(add_3, [32, 4, 3, 128]);  add_3 = None
        unsqueeze_2: "f32[1, 32, 4, 3, 128]" = torch.ops.aten.unsqueeze.default(view_2, 0);  view_2 = None
        permute_2: "f32[3, 32, 4, 1, 128]" = torch.ops.aten.permute.default(unsqueeze_2, [3, 1, 2, 0, 4]);  unsqueeze_2 = None
        squeeze: "f32[3, 32, 4, 128]" = torch.ops.aten.squeeze.dim(permute_2, -2);  permute_2 = None
        clone_1: "f32[3, 32, 4, 128]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select: "f32[32, 4, 128]" = torch.ops.aten.select.int(clone_1, 0, 0)
        select_1: "f32[32, 4, 128]" = torch.ops.aten.select.int(clone_1, 0, 1)
        select_2: "f32[32, 4, 128]" = torch.ops.aten.select.int(clone_1, 0, 2);  clone_1 = None
        view_3: "f32[32, 16, 32]" = torch.ops.aten.reshape.default(select, [32, 16, 32]);  select = None
        permute_3: "f32[16, 32, 32]" = torch.ops.aten.permute.default(view_3, [1, 0, 2]);  view_3 = None
        view_4: "f32[32, 16, 32]" = torch.ops.aten.reshape.default(select_1, [32, 16, 32]);  select_1 = None
        permute_4: "f32[16, 32, 32]" = torch.ops.aten.permute.default(view_4, [1, 0, 2]);  view_4 = None
        view_5: "f32[32, 16, 32]" = torch.ops.aten.reshape.default(select_2, [32, 16, 32]);  select_2 = None
        permute_5: "f32[16, 32, 32]" = torch.ops.aten.permute.default(view_5, [1, 0, 2]);  view_5 = None
        view_6: "f32[4, 4, 32, 32]" = torch.ops.aten.reshape.default(permute_3, [4, 4, 32, 32]);  permute_3 = None
        view_7: "f32[4, 4, 32, 32]" = torch.ops.aten.reshape.default(permute_4, [4, 4, 32, 32]);  permute_4 = None
        view_8: "f32[4, 4, 32, 32]" = torch.ops.aten.reshape.default(permute_5, [4, 4, 32, 32]);  permute_5 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_6, view_7, view_8, None, True, 0.0, True)
        getitem_2: "f32[4, 4, 32, 32]" = _scaled_dot_product_efficient_attention[0]
        getitem_3: "f32[4, 4, 32]" = _scaled_dot_product_efficient_attention[1]
        getitem_4: "i64[]" = _scaled_dot_product_efficient_attention[2]
        getitem_5: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
        permute_6: "f32[32, 4, 4, 32]" = torch.ops.aten.permute.default(getitem_2, [2, 0, 1, 3])
        clone_2: "f32[32, 4, 4, 32]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_9: "f32[128, 128]" = torch.ops.aten.reshape.default(clone_2, [128, 128]);  clone_2 = None
        permute_7: "f32[128, 128]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        mm_default_3: "f32[128, 128]" = torch.ops.aten.mm.default(view_9, permute_7);  permute_7 = None
        add_tensor_3: "f32[128, 128]" = torch.ops.aten.add.Tensor(primals_9, mm_default_3);  primals_9 = mm_default_3 = None
        view_10: "f32[32, 4, 128]" = torch.ops.aten.reshape.default(add_tensor_3, [32, 4, 128]);  add_tensor_3 = None
        permute_8: "f32[4, 32, 128]" = torch.ops.aten.permute.default(view_10, [1, 0, 2]);  view_10 = None
        add_4: "f32[4, 32, 128]" = torch.ops.aten.add.Tensor(add, permute_8);  add = permute_8 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_6: "f32[4, 32, 1]" = var_mean_1[0]
        getitem_7: "f32[4, 32, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[4, 32, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_1: "f32[4, 32, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2: "f32[4, 32, 128]" = torch.ops.aten.sub.Tensor(add_4, getitem_7);  getitem_7 = None
        mul_2: "f32[4, 32, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_3: "f32[4, 32, 128]" = torch.ops.aten.mul.Tensor(mul_2, primals_10)
        add_6: "f32[4, 32, 128]" = torch.ops.aten.add.Tensor(mul_3, primals_11);  mul_3 = primals_11 = None
        view_11: "f32[128, 128]" = torch.ops.aten.reshape.default(add_6, [128, 128]);  add_6 = None
        permute_9: "f32[128, 256]" = torch.ops.aten.permute.default(primals_12, [1, 0])
        addmm_1: "f32[128, 256]" = torch.ops.aten.addmm.default(primals_13, view_11, permute_9);  primals_13 = permute_9 = None
        view_12: "f32[4, 32, 256]" = torch.ops.aten.reshape.default(addmm_1, [4, 32, 256])
        mul_4: "f32[4, 32, 256]" = torch.ops.aten.mul.Tensor(view_12, 0.5)
        mul_5: "f32[4, 32, 256]" = torch.ops.aten.mul.Tensor(view_12, 0.7071067811865476);  view_12 = None
        erf: "f32[4, 32, 256]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_7: "f32[4, 32, 256]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[4, 32, 256]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
        view_13: "f32[128, 256]" = torch.ops.aten.reshape.default(mul_6, [128, 256]);  mul_6 = None
        permute_10: "f32[256, 128]" = torch.ops.aten.permute.default(primals_14, [1, 0])
        mm_default_2: "f32[128, 128]" = torch.ops.aten.mm.default(view_13, permute_10);  permute_10 = None
        add_tensor_2: "f32[128, 128]" = torch.ops.aten.add.Tensor(primals_15, mm_default_2);  primals_15 = mm_default_2 = None
        view_14: "f32[4, 32, 128]" = torch.ops.aten.reshape.default(add_tensor_2, [4, 32, 128]);  add_tensor_2 = None
        add_8: "f32[4, 32, 128]" = torch.ops.aten.add.Tensor(add_4, view_14);  add_4 = view_14 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_8: "f32[4, 32, 1]" = var_mean_2[0]
        getitem_9: "f32[4, 32, 1]" = var_mean_2[1];  var_mean_2 = None
        add_9: "f32[4, 32, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2: "f32[4, 32, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_3: "f32[4, 32, 128]" = torch.ops.aten.sub.Tensor(add_8, getitem_9);  getitem_9 = None
        mul_7: "f32[4, 32, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_8: "f32[4, 32, 128]" = torch.ops.aten.mul.Tensor(mul_7, primals_16)
        add_10: "f32[4, 32, 128]" = torch.ops.aten.add.Tensor(mul_8, primals_17);  mul_8 = primals_17 = None
        permute_11: "f32[32, 4, 128]" = torch.ops.aten.permute.default(add_10, [1, 0, 2]);  add_10 = None
        permute_12: "f32[128, 384]" = torch.ops.aten.permute.default(primals_19, [1, 0])
        clone_6: "f32[32, 4, 128]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_15: "f32[128, 128]" = torch.ops.aten.reshape.default(clone_6, [128, 128]);  clone_6 = None
        mm_1: "f32[128, 384]" = torch.ops.aten.mm.default(view_15, permute_12);  permute_12 = None
        view_16: "f32[32, 4, 384]" = torch.ops.aten.reshape.default(mm_1, [32, 4, 384]);  mm_1 = None
        add_11: "f32[32, 4, 384]" = torch.ops.aten.add.Tensor(view_16, primals_18);  view_16 = primals_18 = None
        view_17: "f32[32, 4, 3, 128]" = torch.ops.aten.reshape.default(add_11, [32, 4, 3, 128]);  add_11 = None
        unsqueeze_3: "f32[1, 32, 4, 3, 128]" = torch.ops.aten.unsqueeze.default(view_17, 0);  view_17 = None
        permute_13: "f32[3, 32, 4, 1, 128]" = torch.ops.aten.permute.default(unsqueeze_3, [3, 1, 2, 0, 4]);  unsqueeze_3 = None
        squeeze_1: "f32[3, 32, 4, 128]" = torch.ops.aten.squeeze.dim(permute_13, -2);  permute_13 = None
        clone_7: "f32[3, 32, 4, 128]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        select_3: "f32[32, 4, 128]" = torch.ops.aten.select.int(clone_7, 0, 0)
        select_4: "f32[32, 4, 128]" = torch.ops.aten.select.int(clone_7, 0, 1)
        select_5: "f32[32, 4, 128]" = torch.ops.aten.select.int(clone_7, 0, 2);  clone_7 = None
        view_18: "f32[32, 16, 32]" = torch.ops.aten.reshape.default(select_3, [32, 16, 32]);  select_3 = None
        permute_14: "f32[16, 32, 32]" = torch.ops.aten.permute.default(view_18, [1, 0, 2]);  view_18 = None
        view_19: "f32[32, 16, 32]" = torch.ops.aten.reshape.default(select_4, [32, 16, 32]);  select_4 = None
        permute_15: "f32[16, 32, 32]" = torch.ops.aten.permute.default(view_19, [1, 0, 2]);  view_19 = None
        view_20: "f32[32, 16, 32]" = torch.ops.aten.reshape.default(select_5, [32, 16, 32]);  select_5 = None
        permute_16: "f32[16, 32, 32]" = torch.ops.aten.permute.default(view_20, [1, 0, 2]);  view_20 = None
        view_21: "f32[4, 4, 32, 32]" = torch.ops.aten.reshape.default(permute_14, [4, 4, 32, 32]);  permute_14 = None
        view_22: "f32[4, 4, 32, 32]" = torch.ops.aten.reshape.default(permute_15, [4, 4, 32, 32]);  permute_15 = None
        view_23: "f32[4, 4, 32, 32]" = torch.ops.aten.reshape.default(permute_16, [4, 4, 32, 32]);  permute_16 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_21, view_22, view_23, None, True, 0.0, True)
        getitem_10: "f32[4, 4, 32, 32]" = _scaled_dot_product_efficient_attention_1[0]
        getitem_11: "f32[4, 4, 32]" = _scaled_dot_product_efficient_attention_1[1]
        getitem_12: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
        getitem_13: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
        permute_17: "f32[32, 4, 4, 32]" = torch.ops.aten.permute.default(getitem_10, [2, 0, 1, 3])
        clone_8: "f32[32, 4, 4, 32]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        view_24: "f32[128, 128]" = torch.ops.aten.reshape.default(clone_8, [128, 128]);  clone_8 = None
        permute_18: "f32[128, 128]" = torch.ops.aten.permute.default(primals_20, [1, 0])
        mm_default_1: "f32[128, 128]" = torch.ops.aten.mm.default(view_24, permute_18);  permute_18 = None
        add_tensor_1: "f32[128, 128]" = torch.ops.aten.add.Tensor(primals_21, mm_default_1);  primals_21 = mm_default_1 = None
        view_25: "f32[32, 4, 128]" = torch.ops.aten.reshape.default(add_tensor_1, [32, 4, 128]);  add_tensor_1 = None
        permute_19: "f32[4, 32, 128]" = torch.ops.aten.permute.default(view_25, [1, 0, 2]);  view_25 = None
        add_12: "f32[4, 32, 128]" = torch.ops.aten.add.Tensor(add_8, permute_19);  add_8 = permute_19 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_14: "f32[4, 32, 1]" = var_mean_3[0]
        getitem_15: "f32[4, 32, 1]" = var_mean_3[1];  var_mean_3 = None
        add_13: "f32[4, 32, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_3: "f32[4, 32, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_4: "f32[4, 32, 128]" = torch.ops.aten.sub.Tensor(add_12, getitem_15);  getitem_15 = None
        mul_9: "f32[4, 32, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_10: "f32[4, 32, 128]" = torch.ops.aten.mul.Tensor(mul_9, primals_22)
        add_14: "f32[4, 32, 128]" = torch.ops.aten.add.Tensor(mul_10, primals_23);  mul_10 = primals_23 = None
        view_26: "f32[128, 128]" = torch.ops.aten.reshape.default(add_14, [128, 128]);  add_14 = None
        permute_20: "f32[128, 256]" = torch.ops.aten.permute.default(primals_24, [1, 0])
        addmm_4: "f32[128, 256]" = torch.ops.aten.addmm.default(primals_25, view_26, permute_20);  primals_25 = permute_20 = None
        view_27: "f32[4, 32, 256]" = torch.ops.aten.reshape.default(addmm_4, [4, 32, 256])
        mul_11: "f32[4, 32, 256]" = torch.ops.aten.mul.Tensor(view_27, 0.5)
        mul_12: "f32[4, 32, 256]" = torch.ops.aten.mul.Tensor(view_27, 0.7071067811865476);  view_27 = None
        erf_1: "f32[4, 32, 256]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_15: "f32[4, 32, 256]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[4, 32, 256]" = torch.ops.aten.mul.Tensor(mul_11, add_15);  mul_11 = add_15 = None
        view_28: "f32[128, 256]" = torch.ops.aten.reshape.default(mul_13, [128, 256]);  mul_13 = None
        permute_21: "f32[256, 128]" = torch.ops.aten.permute.default(primals_26, [1, 0])
        mm_default: "f32[128, 128]" = torch.ops.aten.mm.default(view_28, permute_21);  permute_21 = None
        add_tensor: "f32[128, 128]" = torch.ops.aten.add.Tensor(primals_27, mm_default);  primals_27 = mm_default = None
        view_29: "f32[4, 32, 128]" = torch.ops.aten.reshape.default(add_tensor, [4, 32, 128]);  add_tensor = None
        add_16: "f32[4, 32, 128]" = torch.ops.aten.add.Tensor(add_12, view_29);  add_12 = view_29 = None

        # File: /home/binly/workspace/PyTorch-compile/model.py:59 in forward, code: return self.lm_head(self.final_norm(hidden))
        var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_16: "f32[4, 32, 1]" = var_mean_4[0]
        getitem_17: "f32[4, 32, 1]" = var_mean_4[1];  var_mean_4 = None
        add_17: "f32[4, 32, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_4: "f32[4, 32, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_5: "f32[4, 32, 128]" = torch.ops.aten.sub.Tensor(add_16, getitem_17);  add_16 = getitem_17 = None
        mul_14: "f32[4, 32, 128]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = None
        mul_15: "f32[4, 32, 128]" = torch.ops.aten.mul.Tensor(mul_14, primals_28)
        add_18: "f32[4, 32, 128]" = torch.ops.aten.add.Tensor(mul_15, primals_29);  mul_15 = primals_29 = None
        permute_22: "f32[128, 256]" = torch.ops.aten.permute.default(primals_30, [1, 0])
        view_30: "f32[128, 128]" = torch.ops.aten.reshape.default(add_18, [128, 128]);  add_18 = None
        mm_2: "f32[128, 256]" = torch.ops.aten.mm.default(view_30, permute_22);  permute_22 = None
        view_31: "f32[4, 32, 256]" = torch.ops.aten.reshape.default(mm_2, [4, 32, 256]);  mm_2 = None
        div: "f32[4, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 128);  rsqrt_4 = None

        # File: /home/binly/workspace/PyTorch-compile/model.py:58 in forward, code: hidden = self.transformer(hidden, mask=causal_mask, is_causal=True)
        div_1: "f32[4, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 128);  rsqrt_3 = None
        div_2: "f32[4, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
        div_3: "f32[4, 32, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
        return (view_31, primals_1, primals_4, primals_7, primals_8, primals_10, primals_12, primals_14, primals_16, primals_19, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, iota, embedding, embedding_1, getitem_1, rsqrt, view, view_6, view_7, view_8, getitem_2, getitem_3, getitem_4, getitem_5, view_9, mul_2, view_11, addmm_1, view_13, mul_7, view_15, view_21, view_22, view_23, getitem_10, getitem_11, getitem_12, getitem_13, view_24, mul_9, view_26, addmm_4, view_28, mul_14, view_30, div, div_1, div_2, div_3)
