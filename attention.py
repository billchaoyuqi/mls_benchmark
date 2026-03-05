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
        # 针对长序列的配置 (Prefill)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        # 针对中等序列的配置
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        # 针对极短序列的配置 (Decode 阶段 M=1)
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_warps=2, num_stages=2),
    ],
    key=['Seq_Q', 'Seq_K'], # 当序列长度变化时触发重新调优
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
        qk = tl.dot(q, tl.trans(k)) * sm_scale

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
        acc += tl.dot(p.to(v.dtype), v)

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
_PRINTED_ATTN_AUTOTUNE = False

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


if __name__ == "__main__":
    print("Testing Triton Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")
    print("\nTriton Attention working!")