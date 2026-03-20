"""
Triton Neural Network Layers
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement core layers using Triton kernels
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl


# ============================================================================
# Helper Functions
# ============================================================================

def get_stream():
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


def pad_to_multiple(size: int, multiple: int) -> int:
    return ((size + multiple - 1) // multiple) * multiple


def next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 0 else 1


# ============================================================================
# Triton Kernels (With Autotuning)
# ============================================================================

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride_x, stride_y, hidden_size, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size
    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / hidden_size
    r_std = tl.rsqrt(var + eps)
    y = x * r_std * w
    tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)


@triton.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, stride_x, stride_y, hidden_size, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size
    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / hidden_size
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / hidden_size
    std = tl.sqrt(var + eps)
    y = (x_centered / std) * w + b
    tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)


@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid_x
    tl.store(y_ptr + offs, y, mask=mask)


# --- Autotuned Linear Kernel ---
@triton.autotune(
    configs=[
        # Prefill / large-batch (big M)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        # Small-M (decode, M≈1-64)
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        # Fallback
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel_tf32(
        a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
                    mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="tf32")
    # Fuse bias add — eliminates a separate elementwise kernel launch
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# --- Autotuned Linear+GELU Kernel ---
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_gelu_kernel(
        a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
                    mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="tf32")
    # Fuse bias before activation — eliminates a separate elementwise kernel launch
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    sqrt_2_over_pi = 0.7978845608028654
    acc3 = acc * acc * acc
    inner = sqrt_2_over_pi * (acc + 0.044715 * acc3)
    acc = acc * 0.5 * (1.0 + tl.libdevice.tanh(inner))

    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# --- Autotuned SwiGLU Fused Kernel ---
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def swiglu_fused_kernel(
        a_ptr, gate_ptr, up_ptr, c_ptr, M, N, K,
        stride_am, stride_ak, stride_gk, stride_gn, stride_uk, stride_un, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
                    mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        gate_w = tl.load(gate_ptr + (k + offs_k[:, None]) * stride_gk + offs_n[None, :] * stride_gn,
                         mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        up_w = tl.load(up_ptr + (k + offs_k[:, None]) * stride_uk + offs_n[None, :] * stride_un,
                       mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        gate_acc += tl.dot(a, gate_w, input_precision="tf32")
        up_acc += tl.dot(a, up_w, input_precision="tf32")

    sigmoid = 1.0 / (1.0 + tl.exp(-gate_acc))
    gate_act = gate_acc * sigmoid
    out = gate_act * up_acc

    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def embedding_kernel(indices_ptr, weight_ptr, output_ptr, embedding_dim, stride_w0, stride_w1, stride_out0,
                     BLOCK_SIZE: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    idx = tl.load(indices_ptr + pid0)
    offs = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < embedding_dim
    w = tl.load(weight_ptr + idx * stride_w0 + offs * stride_w1, mask=mask, other=0.0)
    tl.store(output_ptr + pid0 * stride_out0 + offs, w, mask=mask)


@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x_safe = x - x_max
    numerator = tl.exp(x_safe)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator
    tl.store(y_ptr + row * stride_y + offs, y, mask=mask)


# ============================================================================
# Layer Classes
# ============================================================================

def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


class RMSNorm:
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.ones(hidden_size, dtype=torch.float32)
        self.use_triton = _is_power_of_two(hidden_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if self.use_triton and x.is_cuda:
            batch_size = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(batch_size, self.hidden_size).contiguous()
            x_flat = x_flat.to(torch.float32)
            output = torch.empty_like(x_flat)
            if self.weight.device != x.device:
                self.weight = self.weight.to(x.device)
            block = next_power_of_two(self.hidden_size)
            rmsnorm_kernel[(batch_size,)](
                x_flat, self.weight, output, x_flat.stride(0), output.stride(0),
                self.hidden_size, self.eps, BLOCK_SIZE=block,
            )
            return output.reshape(original_shape)
        x_float = x.to(torch.float32)
        variance = torch.mean(x_float * x_float, dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.eps)
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        return (self.weight * x_normed).to(x.dtype)


class LayerNorm:
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.ones(hidden_size, dtype=torch.float32)
        self.bias = torch.zeros(hidden_size, dtype=torch.float32)
        self.use_triton = _is_power_of_two(hidden_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if self.use_triton and x.is_cuda:
            batch_size = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(batch_size, self.hidden_size).contiguous()
            x_flat = x_flat.to(torch.float32)
            output = torch.empty_like(x_flat)
            if self.weight.device != x.device:
                self.weight = self.weight.to(x.device)
            if self.bias.device != x.device:
                self.bias = self.bias.to(x.device)
            block = next_power_of_two(self.hidden_size)
            layernorm_kernel[(batch_size,)](
                x_flat, self.weight, self.bias, output, x_flat.stride(0), output.stride(0),
                self.hidden_size, self.eps, BLOCK_SIZE=block,
            )
            return output.reshape(original_shape)
        x_float = x.to(torch.float32)
        mean = torch.mean(x_float, dim=-1, keepdim=True)
        variance = torch.var(x_float, dim=-1, keepdim=True, unbiased=False)
        x_normed = (x_float - mean) * torch.rsqrt(variance + self.eps)
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        if self.bias.device != x.device:
            self.bias = self.bias.to(x.device)
        return (self.weight * x_normed + self.bias).to(x.dtype)


def gelu(x: torch.Tensor) -> torch.Tensor:
    original_shape = x.shape
    total = int(np.prod(x.shape))
    block = 256
    x_flat = x.reshape(-1).contiguous().to(torch.float32)
    output = torch.empty_like(x_flat)
    grid = (triton.cdiv(total, block),)
    if x.is_cuda:
        gelu_kernel[grid](x_flat, output, total, BLOCK_SIZE=block)
        return output[:total].reshape(original_shape).to(x.dtype)
    return torch.nn.functional.gelu(x)


def silu(x: torch.Tensor) -> torch.Tensor:
    original_shape = x.shape
    total = int(np.prod(x.shape))
    block = 256
    x_flat = x.reshape(-1).contiguous().to(torch.float32)
    output = torch.empty_like(x_flat)
    grid = (triton.cdiv(total, block),)
    if x.is_cuda:
        silu_kernel[grid](x_flat, output, total, BLOCK_SIZE=block)
        return output[:total].reshape(original_shape).to(x.dtype)
    return torch.nn.functional.silu(x)


def get_activation(name: str):
    activations = {"gelu": gelu, "silu": silu}
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


class Linear:
    BACKEND = "triton"

    # 我们只用作占位符，因为实际运算时由 Autotune 决定
    TILE_K = 32

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight = torch.zeros((out_features, in_features), dtype=torch.float32)
        self.bias_param = torch.zeros(out_features, dtype=torch.float32) if bias else None
        self._weight_t_padded = None

        self._printed_autotune = True  # suppress autotune noise

    def _ensure_weight_prepared(self, device):
        if self._weight_t_padded is None or self._weight_t_padded.device != device:
            K = self.in_features
            N = self.out_features
            K_pad = pad_to_multiple(K, self.TILE_K)
            N_pad = pad_to_multiple(N, self.TILE_K)  # 为了适应最大到128的调优池，可以设为一个较大的倍数，比如直接使用转置即可

            weight_t = self.weight.t().contiguous().to(device)
            if K_pad > K or N_pad > N:
                weight_pad = torch.zeros((K_pad, N_pad), dtype=torch.float32, device=device)
                weight_pad[:K, :N] = weight_t
                self._weight_t_padded = weight_pad
            else:
                self._weight_t_padded = weight_t

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if Linear.BACKEND in ("torch", "cublas"):
            return self._forward_torch(x)
        if x.is_cuda:
            return self._forward_triton(x)
        return self._forward_torch(x)

    def _forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        batch_dims = original_shape[:-1]
        M = int(np.prod(batch_dims))
        x_2d = x.reshape(M, self.in_features).to(torch.float32)
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        output = x_2d @ self.weight.t()
        if self.has_bias and self.bias_param is not None:
            if self.bias_param.device != x.device:
                self.bias_param = self.bias_param.to(x.device)
            output = output + self.bias_param
        return output.reshape(*batch_dims, self.out_features)

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        batch_dims = original_shape[:-1]
        M = int(np.prod(batch_dims))
        K = self.in_features
        N = self.out_features
        # Avoid redundant copy: only call .contiguous() when actually needed.
        x_2d = x.reshape(M, K)
        if x_2d.dtype != torch.float32 or not x_2d.is_contiguous():
            x_2d = x_2d.to(torch.float32).contiguous()
        self._ensure_weight_prepared(x.device)
        output = torch.empty((M, N), dtype=torch.float32, device=x.device)

        has_bias = self.has_bias and self.bias_param is not None
        if has_bias and self.bias_param.device != x.device:
            self.bias_param = self.bias_param.to(x.device)

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )
        # bias_ptr: pass bias tensor when HAS_BIAS, else pass output as dummy (never read)
        linear_kernel_tf32[grid](
            x_2d, self._weight_t_padded,
            self.bias_param if has_bias else output,
            output,
            M, N, K,
            x_2d.stride(0), x_2d.stride(1),
            self._weight_t_padded.stride(0), self._weight_t_padded.stride(1),
            output.stride(0), output.stride(1),
            HAS_BIAS=has_bias,
        )
        return output.reshape(*batch_dims, self.out_features)


class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.zeros((num_embeddings, embedding_dim), dtype=torch.float32)

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        original_shape = input_ids.shape
        batch_size = int(np.prod(original_shape))
        if self.weight.device != input_ids.device:
            self.weight = self.weight.to(input_ids.device)
        if not input_ids.is_cuda:
            flat = input_ids.reshape(-1).to(torch.int64)
            output = self.weight.index_select(0, flat)
            return output.reshape(*original_shape, self.embedding_dim)
        indices_flat = input_ids.reshape(-1).to(torch.int32).contiguous()
        output = torch.empty((batch_size, self.embedding_dim), dtype=torch.float32, device=indices_flat.device)
        block = 256
        grid = (batch_size, triton.cdiv(self.embedding_dim, block))
        embedding_kernel[grid](
            indices_flat, self.weight, output, self.embedding_dim,
            self.weight.stride(0), self.weight.stride(1), output.stride(0), BLOCK_SIZE=block,
        )
        return output.reshape(*original_shape, self.embedding_dim)


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    if axis != -1 and axis != len(x.shape) - 1:
        x = torch.movedim(x, axis, -1)
    original_shape = x.shape
    batch_size = int(np.prod(x.shape[:-1]))
    seq_len = x.shape[-1]
    x_flat = x.reshape(batch_size, seq_len).to(torch.float32).contiguous()
    output = torch.empty_like(x_flat)
    if x.is_cuda:
        block = next_power_of_two(seq_len)
        softmax_kernel[(batch_size,)](
            x_flat, output, x_flat.stride(0), output.stride(0), seq_len, BLOCK_SIZE=block,
        )
        result = output.reshape(original_shape)
    else:
        result = torch.softmax(x, dim=-1)
    if axis != -1 and axis != len(original_shape) - 1:
        result = torch.movedim(result, -1, axis)
    return result


class MLP:
    FUSED = True
    TILE_K = 32

    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "silu", bias: bool = False,
                 use_gating: bool = True):
        self.use_gating = use_gating
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=bias)
        self._gate_weight_t = None
        self._up_weight_t = None
        self._printed_autotune = True  # suppress autotune noise

    def _prepare_fused_weights(self, device):
        if self._gate_weight_t is None or self._gate_weight_t.device != device:
            # Direct transpose + device move — no torch.zeros allocation + copy.
            self._gate_weight_t = self.gate_proj.weight.t().to(device).contiguous()
            self._up_weight_t   = self.up_proj.weight.t().to(device).contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gating and MLP.FUSED and x.is_cuda:
            return self._forward_fused(x)
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        self._prepare_fused_weights(x.device)
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.hidden_size)
        if x_2d.dtype != torch.float32 or not x_2d.is_contiguous():
            x_2d = x_2d.to(torch.float32).contiguous()
        M = x_2d.shape[0]
        K = self.hidden_size
        N = self.intermediate_size

        intermediate = torch.empty((M, N), dtype=torch.float32, device=x.device)

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )
        swiglu_fused_kernel[grid](
            x_2d, self._gate_weight_t, self._up_weight_t, intermediate,
            M, N, self.hidden_size,
            x_2d.stride(0), x_2d.stride(1),
            self._gate_weight_t.stride(0), self._gate_weight_t.stride(1),
            self._up_weight_t.stride(0), self._up_weight_t.stride(1),
            intermediate.stride(0), intermediate.stride(1),
        )
        return self.down_proj(intermediate.reshape(*orig_shape[:-1], N))


class EncoderMLP:
    FUSED = True
    TILE_K = 32

    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "gelu", bias: bool = True):
        self.fc1 = Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = get_activation(activation)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias_enabled = bias
        self.activation = activation
        self._fc1_weight_t = None
        self._printed_autotune = True  # suppress autotune noise

    def _prepare_fused_weights(self, device):
        if self._fc1_weight_t is None or self._fc1_weight_t.device != device:
            # Direct transpose + device move — no torch.zeros allocation + copy.
            self._fc1_weight_t = self.fc1.weight.t().to(device).contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if EncoderMLP.FUSED and self.activation == "gelu" and x.is_cuda:
            return self._forward_fused(x)
        return self._forward_standard(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        self._prepare_fused_weights(x.device)
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.hidden_size)
        if x_2d.dtype != torch.float32 or not x_2d.is_contiguous():
            x_2d = x_2d.to(torch.float32).contiguous()
        M = x_2d.shape[0]
        K = self.hidden_size
        N = self.intermediate_size

        has_bias = self.bias_enabled and self.fc1.bias_param is not None
        if has_bias and self.fc1.bias_param.device != x.device:
            self.fc1.bias_param = self.fc1.bias_param.to(x.device)

        intermediate = torch.empty((M, N), dtype=torch.float32, device=x.device)

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )
        linear_gelu_kernel[grid](
            x_2d, self._fc1_weight_t,
            self.fc1.bias_param if has_bias else intermediate,
            intermediate,
            M, N, K,
            x_2d.stride(0), x_2d.stride(1),
            self._fc1_weight_t.stride(0), self._fc1_weight_t.stride(1),
            intermediate.stride(0), intermediate.stride(1),
            HAS_BIAS=has_bias,
        )
        return self.fc2(intermediate.reshape(*orig_shape[:-1], self.intermediate_size))
