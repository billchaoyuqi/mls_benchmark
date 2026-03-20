"""
Triton Rotary Position Embeddings (RoPE)
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Filled TODO sections to implement RoPE using Triton kernels
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for RoPE
# ============================================================================

@triton.jit
def compute_freqs_kernel(
    positions_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    seq_len,
    half_dim,
    stride_pos,
    stride_inv,
    stride_cos0,
    stride_cos1,
    stride_sin0,
    stride_sin1,
    BLOCK: tl.constexpr,
):
    """
    Compute cos and sin for rotary embeddings.
    Grid: (seq_len,)
    """
    # 每个 program 处理序列中的一个位置 (position)
    pid = tl.program_id(0)

    # 1. 确定列偏移（处理 dim // 2 的维度）
    offs = tl.arange(0, BLOCK)
    mask = offs < half_dim

    # 2. 加载当前位置索引和对应的逆频率向量
    # positions_ptr 通常是 [0, 1, 2, ..., seq_len-1]
    pos = tl.load(positions_ptr + pid * stride_pos)
    inv_freq = tl.load(inv_freq_ptr + offs * stride_inv, mask=mask, other=0.0)

    # 3. 计算频率: freqs = pos * inv_freq
    freqs = pos * inv_freq

    # 4. 计算 cos 和 sin
    cos_val = tl.cos(freqs)
    sin_val = tl.sin(freqs)

    # 5. 存储结果到 cache  [seq, half_dim] — 不再复制第二半
    # 调用方只需 half_dim 列，不用重复。节省 2x cache 内存并避免后续 .contiguous() 分配。
    tl.store(cos_ptr + pid * stride_cos0 + offs * stride_cos1, cos_val, mask=mask)
    tl.store(sin_ptr + pid * stride_sin0 + offs * stride_sin1, sin_val, mask=mask)


# ============================================================================
# RoPE Classes
# ============================================================================

class RotaryEmbedding:
    """Rotary Position Embedding using Triton."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        partial_rotary_factor: float = 1.0,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        self.rotary_dim = int(dim * partial_rotary_factor)
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)

        # 预计算 inv_freq 并存为 Tensor
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.inv_freq = inv_freq

        self._update_cache(max_position_embeddings)

    def _update_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Pre-compute cos and sin using Triton kernel."""
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2
        if device is None:
            device = self.inv_freq.device

        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        # Cache stores [seq, half_dim] only — no duplicated second half.
        # Callers only ever need the first half; removing the duplicate:
        #   - halves cache memory usage
        #   - eliminates the [:, :half_dim].contiguous() allocation on every forward call
        cos_cache = torch.empty((seq_len, half_dim), dtype=torch.float32, device=device)
        sin_cache = torch.empty((seq_len, half_dim), dtype=torch.float32, device=device)

        if device.type == "cuda":
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)

            block = triton.next_power_of_2(half_dim)
            compute_freqs_kernel[(seq_len,)](
                positions,
                self.inv_freq,
                cos_cache,
                sin_cache,
                seq_len,
                half_dim,
                positions.stride(0),
                self.inv_freq.stride(0),
                cos_cache.stride(0),
                cos_cache.stride(1),
                sin_cache.stride(0),
                sin_cache.stride(1),
                BLOCK=block,
            )
        else:
            # CPU fallback
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            freqs = positions[:, None] * self.inv_freq[None, :]
            cos_cache[:] = torch.cos(freqs)
            sin_cache[:] = torch.sin(freqs)

        self.cos_cached = cos_cache
        self.sin_cached = sin_cache

    def __call__(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin for given positions."""
        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len, device=x.device)
        elif self.cos_cached.device != x.device:
            self._update_cache(self.max_seq_len_cached, device=x.device)

        if position_ids is not None:
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
            if cos.ndim == 3 and cos.shape[0] == 1:
                cos = cos[0]
                sin = sin[0]
        else:
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)

        return cos, sin


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ROPE_DIM = 256


# ============================================================================
# Triton kernel: fused RoPE application (replaces _apply_rope_single)
# Eliminates torch.cat and intermediate tensor allocations.
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16}, num_warps=2),
    ],
    key=['Seq_Len', 'Rotary_Half_Dim'],
)
@triton.jit
def apply_rope_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Out_ptr,
    stride_xh, stride_xs, stride_xd,
    stride_cs, stride_cd,
    stride_oh, stride_os, stride_od,
    Seq_Len, Head_Dim, Rotary_Half_Dim,
    HAS_PASS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    Apply RoPE to X in a single kernel pass.

    Grid: (batch * heads, ceil(Seq_Len / BLOCK_M))

    Avoids the torch.cat + intermediate allocations that the pure-PyTorch
    path (_apply_rope_single) requires. Writes output directly to Out_ptr.

    Cos/Sin shape: [Seq_Len, Rotary_Half_Dim]  (non-duplicated half).
    """
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_s = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_r = tl.arange(0, BLOCK_R)

    s_mask = offs_s < Seq_Len
    r_mask = offs_r < Rotary_Half_Dim

    x_base = X_ptr   + pid_h * stride_xh
    o_base = Out_ptr + pid_h * stride_oh

    # Load first half [BLOCK_M, BLOCK_R]
    x1 = tl.load(
        x_base + offs_s[:, None] * stride_xs + offs_r[None, :] * stride_xd,
        mask=s_mask[:, None] & r_mask[None, :], other=0.0,
    ).to(tl.float32)

    # Load second half [BLOCK_M, BLOCK_R]
    x2 = tl.load(
        x_base + offs_s[:, None] * stride_xs + (Rotary_Half_Dim + offs_r[None, :]) * stride_xd,
        mask=s_mask[:, None] & r_mask[None, :], other=0.0,
    ).to(tl.float32)

    # Load cos / sin [BLOCK_M, BLOCK_R]
    cos = tl.load(
        Cos_ptr + offs_s[:, None] * stride_cs + offs_r[None, :] * stride_cd,
        mask=s_mask[:, None] & r_mask[None, :], other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        Sin_ptr + offs_s[:, None] * stride_cs + offs_r[None, :] * stride_cd,
        mask=s_mask[:, None] & r_mask[None, :], other=0.0,
    ).to(tl.float32)

    # Apply RoPE rotation
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x2 * cos + x1 * sin

    # Store rotated halves directly — no torch.cat needed
    tl.store(
        o_base + offs_s[:, None] * stride_os + offs_r[None, :] * stride_od,
        x1_rot, mask=s_mask[:, None] & r_mask[None, :],
    )
    tl.store(
        o_base + offs_s[:, None] * stride_os + (Rotary_Half_Dim + offs_r[None, :]) * stride_od,
        x2_rot, mask=s_mask[:, None] & r_mask[None, :],
    )

    # Copy pass-through dims (partial RoPE only; compiled away when HAS_PASS=False)
    if HAS_PASS:
        offs_pass = 2 * BLOCK_R + tl.arange(0, BLOCK_D - 2 * BLOCK_R)
        x_pass = tl.load(
            x_base + offs_s[:, None] * stride_xs + offs_pass[None, :] * stride_xd,
            mask=s_mask[:, None] & (offs_pass[None, :] < Head_Dim), other=0.0,
        )
        tl.store(
            o_base + offs_s[:, None] * stride_os + offs_pass[None, :] * stride_od,
            x_pass, mask=s_mask[:, None] & (offs_pass[None, :] < Head_Dim),
        )


def _apply_rope_triton(
    x: torch.Tensor,
    cos_half: torch.Tensor,
    sin_half: torch.Tensor,
    half_dim: int,
    head_dim: int,
) -> torch.Tensor:
    """Fused Triton RoPE — single kernel, no torch.cat."""
    batch, heads, seq, _ = x.shape
    has_pass = (head_dim > half_dim * 2)
    x_2d  = x.reshape(batch * heads, seq, head_dim).contiguous()
    out_2d = torch.empty_like(x_2d)
    block_r = next_power_of_two(half_dim)
    block_d = next_power_of_two(head_dim)
    grid = lambda META: (batch * heads, triton.cdiv(seq, META['BLOCK_M']))
    apply_rope_kernel[grid](
        x_2d, cos_half, sin_half, out_2d,
        x_2d.stride(0),   x_2d.stride(1),   x_2d.stride(2),
        cos_half.stride(0), cos_half.stride(1),
        out_2d.stride(0),  out_2d.stride(1),  out_2d.stride(2),
        seq, head_dim, half_dim,
        HAS_PASS=has_pass,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
    )
    return out_2d.reshape(batch, heads, seq, head_dim)


def _apply_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    half_dim: int,
    head_dim: int,
) -> torch.Tensor:
    """Apply RoPE to a single tensor (Q or K). Uses Triton on CUDA.

    cos/sin are expected to be [seq_cached, half_dim] — the non-duplicated
    cache format.  A simple row-slice cos[:seq] is always contiguous (no copy).
    """
    seq_len = x.shape[-2]
    if x.is_cuda:
        # Row-slice only: cos[:seq] is always a contiguous view — zero allocation.
        return _apply_rope_triton(x, cos[:seq_len], sin[:seq_len], half_dim, head_dim)
    # CPU fallback
    c = cos[:seq_len][None, None]   # [1, 1, seq, half_dim]
    s = sin[:seq_len][None, None]
    x1_rot = x[..., :half_dim] * c - x[..., half_dim:half_dim*2] * s
    x2_rot = x[..., half_dim:half_dim*2] * c + x[..., :half_dim] * s
    if head_dim > half_dim * 2:
        return torch.cat([x1_rot, x2_rot, x[..., half_dim*2:]], dim=-1)
    return torch.cat([x1_rot, x2_rot], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings.

    cos/sin shape: [seq, half_dim]  (non-duplicated cache format).
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape

    if rotary_dim is None:
        rotary_dim = head_dim

    half_dim = rotary_dim // 2

    # cos/sin are already [seq, half_dim]; no column truncation needed.
    # Ensure fp32 — usually a no-op since the cache is stored as fp32.
    if cos.dtype != torch.float32:
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

    q_out = _apply_rope_single(q, cos, sin, half_dim, head_dim)
    k_out = _apply_rope_single(k, cos, sin, half_dim, head_dim)

    return q_out.to(q.dtype), k_out.to(k.dtype)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to partial dimensions."""
    return apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)


if __name__ == "__main__":
    print("Testing Triton RoPE...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping Triton test.")
    else:
        batch_size = 2
        num_heads = 4
        seq_len = 16
        head_dim = 64

        rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=1024)

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        cos, sin = rope(q)
        print(f"Cos shape: {cos.shape}")
        print(f"Sin shape: {sin.shape}")

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        print(f"Q rotated shape: {q_rot.shape}")
        print(f"K rotated shape: {k_rot.shape}")

        print("\nTesting partial RoPE (50%):")
        rope_partial = RotaryEmbedding(dim=head_dim, partial_rotary_factor=0.5)
        cos_p, sin_p = rope_partial(q)
        q_rot_p, k_rot_p = apply_partial_rotary_pos_emb(q, k, cos_p, sin_p, head_dim // 2)
        print(f"Q rotated (partial) shape: {q_rot_p.shape}")

        print("\nTriton RoPE working!")