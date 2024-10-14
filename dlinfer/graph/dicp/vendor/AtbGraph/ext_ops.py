import math

import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Sequence

torch._dynamo.config.suppress_errors = False

# atb mm
@torch._custom_op.impl.custom_op('atb::linear')
def linear(a: Tensor, b: Tensor, bias: Tensor, trans_a: bool, trans_b: bool) -> Tensor:
    ...


@linear.impl_abstract()
def atb_linear_abstract(a, b, bias, trans_a, trans_b):
    if trans_a:
        a = a.t()
    if trans_b:
        b = b.t()
    return torch.matmul(a, b)


@linear.impl(['cpu', 'cuda'])
def atb_linear_impl(a, b, bias, trans_a, trans_b):
    if trans_a:
        a = a.t()
    if trans_b:
        b = b.t()
    out = torch.matmul(a, b)
    if bias:
        out = out + bias
    return out


# atb mm
@torch._custom_op.impl.custom_op('atb::add')
def add(a: Tensor, b: Tensor) -> Tensor:
    ...


@add.impl_abstract()
def add_abstract(a, b,):
    return a + b


@add.impl(['cpu', 'cuda'])
def add_impl(a, b, bias, trans_a, trans_b):
    return a + b


# atb mm
@torch._custom_op.impl.custom_op('atb::bmm')
def bmm(a: Tensor, b: Tensor) -> Tensor:
    ...


@bmm.impl_abstract()
def bmm_abstract(a, b,):
    return a @ b


@bmm.impl(['cpu', 'cuda'])
def bmm_impl(a, b, bias, trans_a, trans_b):
    return a @ b


# atb fused_mm_mm_add
@torch._custom_op.impl.custom_op('atb::rope')
def rope(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor, seqlen: Tensor) -> tuple[Tensor, Tensor]:
    ...

@rope.impl_abstract()
def rope_abstract(query, key, cos, sin, seqlen):
    return query, key


@rope.impl(['cpu', 'cuda'])
def rope_impl(query, key, cos, sin, seqlen):
    return query, key


@torch._custom_op.impl.custom_op('atb::context_attention')
def context_attention(query: Tensor, key: Tensor, value: Tensor, key_cache: Tensor, value_cache: Tensor, seqlen: Tensor, mask: Tensor, num_q_heads: int, num_kv_heads: int) -> Tensor:
    ...

@context_attention.impl_abstract()
def context_attention_abstract(query, key, value, key_cache: Tensor, value_cache: Tensor, seqlen, mask, num_q_heads: int, num_kv_heads: int):
    return query


@context_attention.impl(['cpu', 'cuda'])
def context_attention_impl(query, key, value, key_cache, value_cache, seqlen, mask, num_q_heads: int, num_kv_heads: int):
    return query


@torch._custom_op.impl.custom_op('atb::fill_kv_cache')
def fill_kv_cache(key: Tensor, value: Tensor, key_cache: Tensor, value_cache: Tensor, kv_indices: Tensor) -> tuple[Tensor, Tensor]:
    ...

@fill_kv_cache.impl_abstract()
def fill_kv_cache_abstract(key, value, key_cache, value_cache, kv_indices):
    return key_cache, value_cache


@fill_kv_cache.impl(['cpu', 'cuda'])
def fill_kv_cache_impl(key, value, key_cache, value_cache, kv_indices):
    return key_cache, value_cache

@torch._custom_op.impl.custom_op('atb::paged_attention_decode')
def paged_attention_decode(query: Tensor, key_cache: Tensor, value_cache: Tensor, block_table: Tensor, context_len: Tensor, maks: Tensor, num_q_heads: int, num_kv_heads: int) -> Tensor:
    ...

@paged_attention_decode.impl_abstract()
def paged_attention_decode_abstract(query, key_cache, value_cache, block_table, context_len, mask, num_q_heads: int, num_kv_heads: int):
    return query


@paged_attention_decode.impl(['cpu', 'cuda'])
def paged_attention_decode_impl(query, key_cache, value_cache, block_table, context_len, mask, num_q_heads: int, num_kv_heads: int):
    return query

@torch._custom_op.impl.custom_op('atb::add_rms_norm')
def add_rms_norm(x1: Tensor, x2: Tensor, gamma: Tensor, epsilon: float) -> tuple[Tensor, Tensor]:
    ...

@add_rms_norm.impl_abstract()
def add_rms_norm_abstract(x1, x2, gamma, epsilon):
    return x1 + x2, x1 + x2


@add_rms_norm.impl(['cpu', 'cuda'])
def add_rms_norm_impl(x1, x2, gamma, epsilon):
    return x1 + x2, x1 + x2

@torch._custom_op.impl.custom_op('atb::silu_and_mul')
def silu_and_mul(gate_up: Tensor) -> Tensor:
    ...

@silu_and_mul.impl_abstract()
def silu_and_mul_abstract(gate_up):
    gate, up = torch.tensor_split(gate_up, 2, dim=-1)
    return gate


@silu_and_mul.impl(['cpu', 'cuda'])
def silu_and_mul_impl(gate_up):
    gate, up = torch.tensor_split(gate_up, 2, dim=-1)
    return gate


@torch._custom_op.impl.custom_op('atb::mlp_gate')
def mlp_gate(input: Tensor, gate_up: Tensor, down: Tensor) -> Tensor:
    ...

@mlp_gate.impl_abstract()
def mlp_gate_abstract(input , gate_up, down):
    return input


@mlp_gate.impl(['cpu', 'cuda'])
def mlp_gate_impl(input, gate_up, down):
    return input


@torch._custom_op.impl.custom_op('atb::atb_context_attention')
def atb_context_attention(query: Tensor,
                          key: Tensor,
                          value: Tensor,
                          k_cache: Tensor,
                          v_cache: Tensor,
                          kv_start_indices_1d: Tensor,
                          kv_seqlens_int: Tensor,
                          mask: Tensor,
                          num_heads: int,
                          num_kv_heads: int,
                          kv_head_size: int,
                          block_size: int) -> Tensor:
    ...

@atb_context_attention.impl_abstract()
def atb_context_attention_abstract(query: Tensor,
                                   key: Tensor,
                                   value: Tensor,
                                   k_cache: Tensor,
                                   v_cache: Tensor,
                                   kv_start_indices_1d: Tensor,
                                   kv_seqlens_int: Tensor,
                                   mask: Tensor,
                                   num_heads: int,
                                   num_kv_heads: int,
                                   kv_head_size: int,
                                   block_size: int):
    return query

@atb_context_attention.impl(['cpu', 'cuda'])
def atb_context_attention_impl(query: Tensor,
                               key: Tensor,
                               value: Tensor,
                               k_cache: Tensor,
                               v_cache: Tensor,
                               kv_start_indices_1d: Tensor,
                               kv_seqlens_int: Tensor,
                               mask: Tensor,
                               num_heads: int,
                               num_kv_heads: int,
                               kv_head_size: int,
                               block_size: int):
    return query



@torch._custom_op.impl.custom_op('atb::atb_paged_attention')
def atb_paged_attention(query: Tensor,
                          key: Tensor,
                          value: Tensor,
                          k_cache: Tensor,
                          v_cache: Tensor,
                          kv_start_indices_1d: Tensor,
                          kv_seqlens_int: Tensor,
                          mask: Tensor,
                          block_offset: Tensor,
                          num_heads: int,
                          num_kv_heads: int,
                          kv_head_size: int,
                          block_size: int) -> Tensor:
    ...

@atb_paged_attention.impl_abstract()
def atb_paged_attention_abstract(query: Tensor,
                                   key: Tensor,
                                   value: Tensor,
                                   k_cache: Tensor,
                                   v_cache: Tensor,
                                   kv_start_indices_1d: Tensor,
                                   kv_seqlens_int: Tensor,
                                   mask: Tensor,
                                   block_offset: Tensor,
                                   num_heads: int,
                                   num_kv_heads: int,
                                   kv_head_size: int,
                                   block_size: int):
    return query

@atb_paged_attention.impl(['cpu', 'cuda'])
def atb_paged_attention_impl(query: Tensor,
                               key: Tensor,
                               value: Tensor,
                               k_cache: Tensor,
                               v_cache: Tensor,
                               kv_start_indices_1d: Tensor,
                               kv_seqlens_int: Tensor,
                               mask: Tensor,
                               block_offset: Tensor,
                               num_heads: int,
                               num_kv_heads: int,
                               kv_head_size: int,
                               block_size: int):
    return query

@torch._custom_op.impl.custom_op('atb::add_and_rms_norm')
def add_and_rms_norm(x1: Tensor, x2: Tensor, gamma: Tensor, epsilon: float) -> Tensor:
    ...

@add_and_rms_norm.impl_abstract()
def add_and_rms_norm_abstract(x1, x2, gamma, epsilon):
    return x1 + x2


@add_and_rms_norm.impl(['cpu', 'cuda'])
def add_and_rms_norm_impl(x1, x2, gamma, epsilon):
    return x1 + x2


@torch._custom_op.impl.custom_op('atb::view_and_rope')
def view_and_rope(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor, seqlen: Tensor, view_size: int) -> tuple[Tensor, Tensor]:
    ...

@view_and_rope.impl_abstract()
def view_and_rope_abstract(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor, seqlen: Tensor, view_size: int) -> tuple[Tensor, Tensor]:
    query = torch.ops.aten.view.default(query, [-1, view_size])
    return query


@view_and_rope.impl(['cpu', 'cuda'])
def view_and_rope_impl(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor, seqlen: Tensor, view_size: int) -> tuple[Tensor, Tensor]:
    query = torch.ops.aten.view.default(query, [-1, view_size])
    return query
