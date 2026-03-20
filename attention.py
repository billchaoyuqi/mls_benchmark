"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None

# ============================================================================
# Triton Kernels for Attention (FlashAttention Fused + Autotune)
# ============================================================================

@triton.autotune(
    configs=[
        # Prefill: large tiles, more pipeline stages
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=8, num_stages=2),
        # Medium sequences
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32},  num_warps=4, num_stages=2),
        # Decode (small Seq_Q)
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16},  num_warps=2, num_stages=2),
    ],
    key=['Seq_Q', 'Seq_K'],
)
@triton.jit
def flash_attn_fused_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    Seq_Q, Seq_K, Head_Dim,
    is_causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 程序 ID: 处理的 Batch*Head 索引，以及 Q 的块索引
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    # 计算块内偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # 定位当前 Batch 和 Head 的基础指针
    q_ptrs = Q + pid_bh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + pid_bh * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + pid_bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    o_ptrs = Out + pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    # 1. 加载 Query
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < Seq_Q) & (offs_d[None, :] < Head_Dim), other=0.0)

    # 2. 初始化 Online Softmax 变量
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # 3. 遍历 K 和 V
    for start_n in range(0, Seq_K, BLOCK_N):
        curr_k_ptrs = k_ptrs + start_n * stride_kn
        curr_v_ptrs = v_ptrs + start_n * stride_vn

        # 加载 Key
        k = tl.load(curr_k_ptrs, mask=((start_n + offs_n)[:, None] < Seq_K) & (offs_d[None, :] < Head_Dim), other=0.0)

        # Q @ K^T
        qk = tl.dot(q, tl.trans(k), input_precision="tf32") * sm_scale

        # 边界掩码
        qk = tl.where((start_n + offs_n)[None, :] < Seq_K, qk, -float('inf'))

        # 因果掩码 (Causal Mask)
        if is_causal:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], qk, -float('inf'))

        # Online Softmax 计算
        m_ij = tl.max(qk, 1)
        m_next = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_next[:, None])
        l_ij = tl.sum(p, 1)

        # 更新累加器缩放系数
        alpha = tl.exp(m_i - m_next)
        acc = acc * alpha[:, None]

        # 加载 Value
        v = tl.load(curr_v_ptrs, mask=((start_n + offs_n)[:, None] < Seq_K) & (offs_d[None, :] < Head_Dim), other=0.0)

        # 累加 Attention Output
        acc += tl.dot(p.to(v.dtype), v, input_precision="tf32")

        # 更新统计量
        l_i = l_i * alpha + l_ij
        m_i = m_next

    # 4. 最终归一化与存储
    acc = acc / l_i[:, None]
    tl.store(o_ptrs, acc, mask=(offs_m[:, None] < Seq_Q) & (offs_d[None, :] < Head_Dim))


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: Optional[int] = None, head_dim: Optional[int] = None):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(q, k, v, attention_mask, is_causal, self.scale)

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(batch, num_kv_heads, num_repeats, seq_len, head_dim)
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)

def next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 0 else 1

MAX_ATTENTION_DIM = 256

# 全局变量控制打印，防止刷屏
_PRINTED_ATTN_AUTOTUNE = True  # suppress autotune noise

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    global _PRINTED_ATTN_AUTOTUNE
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    # 判断是否能够走 Triton 内核。带有外部 Mask 的操作 (如 Encoder 填充对齐) 退回到 Torch 以保正确性
    # Decode 阶段通常没有 Mask (只有 is_causal=True), 会走 Triton 性能爆发路线
    use_triton = q.is_cuda and attention_mask is None

    if use_triton:
        q = q.to(torch.float32).contiguous()
        k = k.to(torch.float32).contiguous()
        v = v.to(torch.float32).contiguous()

        output = torch.empty_like(q)
        head_dim_padded = next_power_of_two(head_dim)

        # 改为使用 lambda 接收 autotune 的动态 META
        grid = lambda META: (batch * num_heads, triton.cdiv(seq_q, META['BLOCK_M']))

        flash_attn_fused_kernel[grid](
            q, k, v, output,
            float(scale),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            seq_q, seq_k, head_dim,
            is_causal=is_causal,
            BLOCK_D=head_dim_padded,
            # 注意：删除了显式传入的 BLOCK_M, BLOCK_N, num_warps, num_stages
        )

        # 打印调优结果
        if not _PRINTED_ATTN_AUTOTUNE and flash_attn_fused_kernel.best_config is not None:
            print(f"[Autotune] FlashAttention Kernel for Seq_Q={seq_q}, Seq_K={seq_k} selected config: {flash_attn_fused_kernel.best_config}")
            _PRINTED_ATTN_AUTOTUNE = True

        return output

    # Fallback to pure PyTorch logic
    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        mask = torch.triu(torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device), diagonal=1) * -1e9
        scores = scores + mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask

    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    return torch.einsum("bnqk,bnkd->bnqd", attn_weights, v).to(q.dtype)


# ============================================================================
# PyTorch Reference: MHA with RoPE (Ground Truth)
# ============================================================================

def mha_rope_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: Optional[int] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Pure PyTorch reference: applies RoPE then scaled dot-product attention.

    This serves as the ground truth for correctness verification of the
    Triton fused kernel. No Triton, no tricks — just plain PyTorch.

    Args:
        q, k, v: [batch, heads, seq, head_dim]
        cos, sin: [seq, rotary_dim] or [seq, rotary_dim//2] (first half only)
        rotary_dim: number of dimensions to rotate (default: head_dim)
        is_causal: apply causal mask
        scale: attention scale (default: 1/sqrt(head_dim))
    """
    head_dim = q.shape[-1]
    if rotary_dim is None:
        rotary_dim = head_dim
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    half_dim = rotary_dim // 2

    # cos/sin are [seq, half_dim] (non-duplicated cache format).
    cos_h = cos.float() if cos.dtype != torch.float32 else cos
    sin_h = sin.float() if sin.dtype != torch.float32 else sin

    def rotate(x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation to a single Q or K tensor."""
        seq = x.shape[-2]
        x1 = x[..., :half_dim].float()           # first half of rotary dims
        x2 = x[..., half_dim:rotary_dim].float()  # second half of rotary dims
        c = cos_h[:seq][None, None]               # [1, 1, seq, half_dim]
        s = sin_h[:seq][None, None]
        x1_rot = x1 * c - x2 * s
        x2_rot = x2 * c + x1 * s
        if rotary_dim < head_dim:
            # Partial RoPE: pass through remaining dimensions unchanged
            return torch.cat([x1_rot, x2_rot, x[..., rotary_dim:].float()], dim=-1)
        return torch.cat([x1_rot, x2_rot], dim=-1)

    q_rot = rotate(q)
    k_rot = rotate(k)

    # Scaled dot-product attention (numerically stable)
    seq_q, seq_k = q.shape[-2], k.shape[-2]
    scores = torch.einsum("bnqd,bnkd->bnqk", q_rot, k_rot) * scale

    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_q, seq_k, dtype=torch.bool, device=q.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask[None, None], float('-inf'))

    scores = scores - scores.amax(dim=-1, keepdim=True)  # stability
    attn = torch.exp(scores)
    attn = attn / attn.sum(dim=-1, keepdim=True)

    return torch.einsum("bnqk,bnkd->bnqd", attn, v.float()).to(q.dtype)


# ============================================================================
# Triton Fused RoPE + FlashAttention Kernel
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16},  num_warps=2, num_stages=2),
    ],
    key=['Seq_Q', 'Seq_K'],
)
@triton.jit
def flash_attn_rope_fused_kernel(
    Q, K, V, Cos, Sin, Out,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    stride_ob, stride_oh, stride_om, stride_od,
    Seq_Q, Seq_K, Head_Dim, Rotary_Half_Dim,
    is_causal: tl.constexpr,
    HAS_PASS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    Fused RoPE + FlashAttention Triton kernel.

    Key idea: instead of writing rotated Q/K back to HBM then re-reading
    them for attention, apply RoPE inline as tiles are loaded. This fuses
    two separate kernels into one and saves:
        2 * (read Q + read K + write Q_rot + write K_rot)
    = 4 * batch * heads * seq * head_dim * sizeof(dtype)  bytes of HBM traffic.

    QK^T is computed in pieces to accommodate the RoPE split:
        QK^T = q1_rot @ k1_rot.T  +  q2_rot @ k2_rot.T
             [+ q_pass @ k_pass.T  if HAS_PASS]
    Total arithmetic is identical to the unfused version.

    Args:
        Cos, Sin: [seq, Rotary_Half_Dim] — the NON-duplicated half of the cache.
        HAS_PASS: constexpr True when head_dim > rotary_dim (partial RoPE).
        BLOCK_R:  constexpr next_power_of_2(Rotary_Half_Dim).
        BLOCK_D:  constexpr next_power_of_2(Head_Dim).
    """
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_r = tl.arange(0, BLOCK_R)   # indices within [0, Rotary_Half_Dim)

    # Combined batch*head base offset (works for contiguous tensors)
    q_base = Q   + pid_bh * stride_qh
    k_base = K   + pid_bh * stride_kh
    v_base = V   + pid_bh * stride_vh
    o_base = Out + pid_bh * stride_oh

    m_mask = offs_m < Seq_Q
    r_mask = offs_r < Rotary_Half_Dim

    # ------------------------------------------------------------------ #
    # Load Q blocks and apply RoPE                                         #
    # ------------------------------------------------------------------ #
    # q1: Q[offs_m, 0 : Rotary_Half_Dim]
    q1 = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_r[None, :] * stride_qd,
        mask=m_mask[:, None] & r_mask[None, :], other=0.0,
    ).to(tl.float32)

    # q2: Q[offs_m, Rotary_Half_Dim : 2*Rotary_Half_Dim]
    q2 = tl.load(
        q_base + offs_m[:, None] * stride_qm + (Rotary_Half_Dim + offs_r[None, :]) * stride_qd,
        mask=m_mask[:, None] & r_mask[None, :], other=0.0,
    ).to(tl.float32)

    # cos/sin for query positions  [BLOCK_M, BLOCK_R]
    q_cos = tl.load(
        Cos + offs_m[:, None] * stride_cos_s + offs_r[None, :] * stride_cos_d,
        mask=m_mask[:, None] & r_mask[None, :], other=1.0,
    ).to(tl.float32)
    q_sin = tl.load(
        Sin + offs_m[:, None] * stride_sin_s + offs_r[None, :] * stride_sin_d,
        mask=m_mask[:, None] & r_mask[None, :], other=0.0,
    ).to(tl.float32)

    # RoPE: [x1*cos - x2*sin,  x2*cos + x1*sin]
    q1_rot = q1 * q_cos - q2 * q_sin   # [BLOCK_M, BLOCK_R]
    q2_rot = q2 * q_cos + q1 * q_sin   # [BLOCK_M, BLOCK_R]

    # Pass-through dims of Q (compiled away when HAS_PASS=False)
    if HAS_PASS:
        offs_pass = 2 * BLOCK_R + tl.arange(0, BLOCK_D - 2 * BLOCK_R)
        q_pass = tl.load(
            q_base + offs_m[:, None] * stride_qm + offs_pass[None, :] * stride_qd,
            mask=m_mask[:, None] & (offs_pass[None, :] < Head_Dim), other=0.0,
        ).to(tl.float32)   # [BLOCK_M, BLOCK_D - 2*BLOCK_R]

    # ------------------------------------------------------------------ #
    # Online softmax state + output accumulator                           #
    # ------------------------------------------------------------------ #
    m_i  = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i  = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # ------------------------------------------------------------------ #
    # Iterate over K / V tiles                                            #
    # ------------------------------------------------------------------ #
    for start_n in range(0, Seq_K, BLOCK_N):
        n_mask = (start_n + offs_n) < Seq_K

        # k1, k2: rotary halves of K
        k1 = tl.load(
            k_base + (start_n + offs_n)[:, None] * stride_kn + offs_r[None, :] * stride_kd,
            mask=n_mask[:, None] & r_mask[None, :], other=0.0,
        ).to(tl.float32)
        k2 = tl.load(
            k_base + (start_n + offs_n)[:, None] * stride_kn + (Rotary_Half_Dim + offs_r[None, :]) * stride_kd,
            mask=n_mask[:, None] & r_mask[None, :], other=0.0,
        ).to(tl.float32)

        # cos/sin for key positions  [BLOCK_N, BLOCK_R]
        k_cos = tl.load(
            Cos + (start_n + offs_n)[:, None] * stride_cos_s + offs_r[None, :] * stride_cos_d,
            mask=n_mask[:, None] & r_mask[None, :], other=1.0,
        ).to(tl.float32)
        k_sin = tl.load(
            Sin + (start_n + offs_n)[:, None] * stride_sin_s + offs_r[None, :] * stride_sin_d,
            mask=n_mask[:, None] & r_mask[None, :], other=0.0,
        ).to(tl.float32)

        k1_rot = k1 * k_cos - k2 * k_sin   # [BLOCK_N, BLOCK_R]
        k2_rot = k2 * k_cos + k1 * k_sin   # [BLOCK_N, BLOCK_R]

        # QK^T computed in two (or three) tile-matmul pieces
        # Each piece: [BLOCK_M, BLOCK_R] @ [BLOCK_R, BLOCK_N]
        qk = tl.dot(q1_rot, tl.trans(k1_rot), input_precision="tf32") + \
             tl.dot(q2_rot, tl.trans(k2_rot), input_precision="tf32")

        if HAS_PASS:
            k_pass = tl.load(
                k_base + (start_n + offs_n)[:, None] * stride_kn + offs_pass[None, :] * stride_kd,
                mask=n_mask[:, None] & (offs_pass[None, :] < Head_Dim), other=0.0,
            ).to(tl.float32)
            qk += tl.dot(q_pass, tl.trans(k_pass), input_precision="tf32")

        qk *= sm_scale

        # Out-of-bounds keys → -inf
        qk = tl.where(n_mask[None, :], qk, -float('inf'))

        # Causal mask (compiled away when is_causal=False)
        if is_causal:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], qk, -float('inf'))

        # Online softmax update
        m_ij  = tl.max(qk, 1)
        m_next = tl.maximum(m_i, m_ij)
        p      = tl.exp(qk - m_next[:, None])
        l_ij   = tl.sum(p, 1)
        alpha  = tl.exp(m_i - m_next)
        acc    = acc * alpha[:, None]

        # Load V (full head_dim) and accumulate attention output
        v = tl.load(
            v_base + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & (offs_d[None, :] < Head_Dim), other=0.0,
        ).to(tl.float32)
        acc += tl.dot(p.to(v.dtype), v, input_precision="tf32")

        l_i = l_i * alpha + l_ij
        m_i = m_next

    # ------------------------------------------------------------------ #
    # Normalize and store output                                          #
    # ------------------------------------------------------------------ #
    acc = acc / l_i[:, None]
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc,
        mask=m_mask[:, None] & (offs_d[None, :] < Head_Dim),
    )


# ============================================================================
# Wrapper: Fused RoPE + Attention (with PyTorch fallback)
# ============================================================================

_PRINTED_FUSED_AUTOTUNE = True  # suppress autotune noise


def scaled_dot_product_attention_with_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Fused RoPE + scaled dot-product attention.

    On CUDA without an external attention_mask, uses the Triton fused kernel
    which applies RoPE inline during FlashAttention, saving two full HBM
    read/write passes over Q and K compared to the unfused approach.

    Falls back to ``mha_rope_torch`` (PyTorch) when CUDA is unavailable or an
    external mask is present.

    Args:
        q, k, v: [batch, heads, seq, head_dim]  (any dtype; converted to fp32)
        cos, sin: [seq, rotary_dim] from RotaryEmbedding (duplicated cache format)
        rotary_dim: dimensions to rotate (default: head_dim → full RoPE)
        attention_mask: optional additive mask; forces PyTorch fallback if given
        is_causal: apply causal mask
        scale: softmax scale (default: 1/sqrt(head_dim))
    """
    global _PRINTED_FUSED_AUTOTUNE
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if rotary_dim is None:
        rotary_dim = head_dim
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    # Fall back to PyTorch for CPU or when an explicit mask is supplied
    if not q.is_cuda or attention_mask is not None:
        return mha_rope_torch(q, k, v, cos, sin, rotary_dim, is_causal, scale)

    # --- Triton fused path ---
    q = q.to(torch.float32).contiguous()
    k = k.to(torch.float32).contiguous()
    v = v.to(torch.float32).contiguous()

    # cos/sin are now [seq, half_dim] (non-duplicated cache format).
    # .to(fp32) is usually a no-op; .contiguous() is a no-op on the row-slice.
    half_dim = rotary_dim // 2
    cos_half = cos.to(torch.float32).contiguous()
    sin_half = sin.to(torch.float32).contiguous()

    output = torch.empty_like(q)
    head_dim_padded = next_power_of_two(head_dim)
    block_r         = next_power_of_two(half_dim)
    has_pass        = bool(rotary_dim < head_dim)

    grid = lambda META: (batch * num_heads, triton.cdiv(seq_q, META['BLOCK_M']))

    flash_attn_rope_fused_kernel[grid](
        q, k, v, cos_half, sin_half, output,
        float(scale),
        q.stride(0),       q.stride(1),       q.stride(2),       q.stride(3),
        k.stride(0),       k.stride(1),       k.stride(2),       k.stride(3),
        v.stride(0),       v.stride(1),       v.stride(2),       v.stride(3),
        cos_half.stride(0), cos_half.stride(1),
        sin_half.stride(0), sin_half.stride(1),
        output.stride(0),  output.stride(1),  output.stride(2),  output.stride(3),
        seq_q, seq_k, head_dim, half_dim,
        is_causal=is_causal,
        HAS_PASS=has_pass,
        BLOCK_D=head_dim_padded,
        BLOCK_R=block_r,
    )

    if not _PRINTED_FUSED_AUTOTUNE and flash_attn_rope_fused_kernel.best_config is not None:
        print(f"[Autotune] Fused RoPE+Attn for Seq_Q={seq_q}, Seq_K={seq_k}: "
              f"{flash_attn_rope_fused_kernel.best_config}")
        _PRINTED_FUSED_AUTOTUNE = True

    return output


# ============================================================================
# Test & Benchmark
# ============================================================================

if __name__ == "__main__":
    import time
    import sys
    from rope import RotaryEmbedding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("CUDA not available — Triton tests skipped.")
        sys.exit(0)

    def allclose(a, b, atol=1e-3, rtol=1e-3, label=""):
        ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
        status = "PASS" if ok else "FAIL"
        if not ok:
            diff = (a.float() - b.float()).abs()
            print(f"  [{label}] max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}")
        return ok, status

    def bench(fn, warmup=3, iters=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000  # ms

    # ------------------------------------------------------------------
    # 1. Correctness: existing FlashAttention (no RoPE)
    # ------------------------------------------------------------------
    print("\n=== 1. Existing FlashAttention correctness ===")
    B, H, S, D = 2, 4, 64, 64
    q = torch.randn(B, H, S, D, device=device)
    k = torch.randn(B, H, S, D, device=device)
    v = torch.randn(B, H, S, D, device=device)

    out_triton = scaled_dot_product_attention(q, k, v)
    out_torch  = scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu()).to(device)
    _, s = allclose(out_triton, out_torch, label="basic attn")
    print(f"  basic attention:   {s}")

    out_triton_c = scaled_dot_product_attention(q, k, v, is_causal=True)
    out_torch_c  = scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), is_causal=True).to(device)
    _, s = allclose(out_triton_c, out_torch_c, label="causal attn")
    print(f"  causal attention:  {s}")

    # ------------------------------------------------------------------
    # 2. Correctness: PyTorch reference mha_rope_torch
    # ------------------------------------------------------------------
    print("\n=== 2. PyTorch RoPE+Attn reference correctness ===")
    rope_full    = RotaryEmbedding(dim=D, max_position_embeddings=256)
    cos, sin     = rope_full(q)

    ref = mha_rope_torch(q, k, v, cos, sin, rotary_dim=D)
    print(f"  mha_rope_torch output shape: {ref.shape}  dtype: {ref.dtype}")

    # Cross-check against manual unfused: apply_rotary_pos_emb + scaled_dot_product_attention
    from rope import apply_rotary_pos_emb
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, D)
    ref2 = scaled_dot_product_attention(q_rot, k_rot, v)
    _, s = allclose(ref, ref2, label="ref vs unfused")
    print(f"  mha_rope_torch vs unfused:   {s}")

    # ------------------------------------------------------------------
    # 3. Correctness: Fused Triton kernel — full RoPE
    # ------------------------------------------------------------------
    print("\n=== 3. Fused Triton kernel — full RoPE (head_dim=64, rotary_dim=64) ===")
    fused = scaled_dot_product_attention_with_rope(q, k, v, cos, sin, rotary_dim=D)
    _, s  = allclose(fused, ref, label="fused vs torch ref")
    print(f"  fused Triton vs PyTorch ref: {s}")

    _, s = allclose(fused, ref2, label="fused vs unfused")
    print(f"  fused Triton vs unfused:     {s}")

    # ------------------------------------------------------------------
    # 4. Correctness: Fused Triton kernel — partial RoPE (50 %)
    # ------------------------------------------------------------------
    print("\n=== 4. Fused Triton kernel — partial RoPE (head_dim=64, rotary_dim=32) ===")
    B2, H2, S2, D2 = 2, 4, 64, 64
    rotary_dim_partial = 32
    q2 = torch.randn(B2, H2, S2, D2, device=device)
    k2 = torch.randn(B2, H2, S2, D2, device=device)
    v2 = torch.randn(B2, H2, S2, D2, device=device)

    rope_partial = RotaryEmbedding(dim=D2, partial_rotary_factor=0.5, max_position_embeddings=256)
    cos_p, sin_p = rope_partial(q2)

    ref_p   = mha_rope_torch(q2, k2, v2, cos_p, sin_p, rotary_dim=rotary_dim_partial)
    fused_p = scaled_dot_product_attention_with_rope(q2, k2, v2, cos_p, sin_p, rotary_dim=rotary_dim_partial)
    _, s    = allclose(fused_p, ref_p, label="partial fused vs ref")
    print(f"  fused Triton vs PyTorch ref (partial): {s}")

    # ------------------------------------------------------------------
    # 5. Correctness: causal + full RoPE
    # ------------------------------------------------------------------
    print("\n=== 5. Fused Triton kernel — causal + full RoPE ===")
    ref_causal   = mha_rope_torch(q, k, v, cos, sin, rotary_dim=D, is_causal=True)
    fused_causal = scaled_dot_product_attention_with_rope(q, k, v, cos, sin, rotary_dim=D, is_causal=True)
    _, s         = allclose(fused_causal, ref_causal, label="causal fused vs ref")
    print(f"  causal fused vs PyTorch ref: {s}")

    # ------------------------------------------------------------------
    # 6. Benchmark: unfused vs fused at various sequence lengths
    # ------------------------------------------------------------------
    print("\n=== 6. Benchmark: unfused (RoPE + FlashAttn) vs fused ===")
    print(f"  {'seq':>6}  {'unfused (ms)':>14}  {'fused (ms)':>12}  {'speedup':>9}")
    print("  " + "-" * 48)

    for seq in [64, 128, 256, 512, 1024]:
        qb = torch.randn(2, 8, seq, 64, device=device, dtype=torch.float32)
        kb = torch.randn(2, 8, seq, 64, device=device, dtype=torch.float32)
        vb = torch.randn(2, 8, seq, 64, device=device, dtype=torch.float32)
        rope_b = RotaryEmbedding(dim=64, max_position_embeddings=seq + 64)
        cos_b, sin_b = rope_b(qb)

        def unfused():
            qr, kr = apply_rotary_pos_emb(qb, kb, cos_b, sin_b)
            return scaled_dot_product_attention(qr, kr, vb)

        def fused_fn():
            return scaled_dot_product_attention_with_rope(qb, kb, vb, cos_b, sin_b)

        t_unfused = bench(unfused)
        t_fused   = bench(fused_fn)
        speedup   = t_unfused / t_fused
        print(f"  {seq:>6}  {t_unfused:>14.3f}  {t_fused:>12.3f}  {speedup:>8.2f}x")

    print("\nAll tests complete.")