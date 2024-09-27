import torch
import torch.utils.benchmark as benchmark
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl
from typing import Set, Tuple, List, Dict, Any, Callable
import pandas
from tqdm import tqdm
import argparse
import os

###
### Activations is taken from https://github.com/dtunai/triton-activations/blob/main/triton_activations/functions.py
###

@triton.jit
def silu_activation_fwd_kernel(x : tl.Tensor) -> tl.Tensor:
    """
    Computes the element-wise function: {silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}
    """
    return x * (1.0 / (1.0 + tl.exp(-x)))


GELU_COEFF = ( 2.0 / torch.pi ) ** 0.5


###
### Adapted to always use approximation
###
@triton.jit
def gelu_activation_fwd_kernel(x : tl.Tensor) -> tl.Tensor:
    """
    Computes the approximate formulation of GeLU: {gelu}(x) = \frac{x}{2} \left(1 + \mathrm{tanh} \left( \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)
    """
	return 0.5 * x * ( 1.0 + tl.libdevice.tanh( GELU_COEFF * (x + 0.044715 * x * x * x) ) )


###
### Configurations
###
minimal_autotune_configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_K': BK}, num_stages=s, num_warps=w) \
    for BM in [32, 64, 128, 256]\
    for BK in [32, 64, 128, 256]\
    for s in [3, 7]\
    for w in [4, 8]\
]


static_config = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=4)
]


###
### Fused FC Layer
### TODO: Add batch dimension blocking?
### M = # outputs
### N = B
### K = # inputs
@triton.autotune(minimal_autotune_configs, key=['OUT_CHANNELS', 'IN_CHANNELS', 'BATCH_SIZE'])
@triton.jit
def _parallel_linear_fwd_kernel_fp32(
        W,     # (M, K)
        X,     # (B, K) (must transpose)
        D,     # (B, M)
        stride_wm, stride_wk,
        stride_xb, stride_xk,,
        stride_db, stride_dm,
		IN_CHANNELS: tl.constexpr,
		OUT_CHANNELS: tl.constexpr,
		BLOCK_M: tl.constexpr,
        BATCH_SIZE: tl.constexpr, 
        BLOCK_K: tl.constexpr,
):
    ### Which program are we?
	program_m_id = tl.program_id(axis=0)
    program_k_id = tl.program_id(axis=1)

    ### Compute any offsets
    K_offset = program_k_id * BLOCK_K
	M_offset = program_m_id * BLOCK_M

    ### Create pointers!
    W_block_ptr = tl.make_block_ptr(
        base=W + (stride_wm * program_m_id) + (stride_wk * program_k_id),
        shape=(IN_CHANNELS, OUT_CHANNELS),
        strides=(stride_wk, stride_wm),
        offsets=(K_offset, 0),
        block_shape=(BLOCK_K, BLOCK_M),
        order=(1, 0),
    )

    XT_block_ptr = tl.make_block_ptr(
        base=X + (stride_xk * K_offset),
        shape=(IN_CHANNELS, BATCH_SIZE),
        strides=(stride_xk, stride_xb),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BATCH_SIZE),
        order=(0, 1),
    )

    D_block_ptr = tl.make_block_ptr(
        base=D + (stride_dm * program_m_id),
        shape=(BATCH_SIZE, OUT_CHANNELS),
        strides=(stride_db, stride_dm),
        offsets=(0, M_offset),
        block_shape=(BATCH_SIZE, BLOCK_M),
        order=(1, 0),
    )

    ### Accumulation term
    accum = tl.zeros((BATCH_SIZE, BLOCK_M), dtype=tl.float32)

	### Load W into shared memory
    w = tl.load(W_block_ptr)

    for k in range(0, tl.cdiv(IN_CHANNELS, BLOCK_K)):
		### Load chunk of x
        xt = tl.load(XT_block_ptr, padding_option="zero", boundary_check=(0,))

        ### Compute psi(K.T) @ V for numerator
        accum = tl.dot(w, xt, acc=accum)
    
        ### Advance pointers
        XT_block_ptr = tl.advance(XT_block_ptr, (BLOCK_K, 0))

    ### Epilogue
    tl.store(D_block_ptr, accum, boundary_check=(0,))



###
### Multi-headed Attention Forward
###
def parallel_linear_fwd_fp32(
    W : torch.Tensor, # (IN_CHANNELS, OUT_CHANNELS)
    X : torch.Tensor, # (BATCH_SIZE, IN_CHANNELS)
) -> torch.Tensor:    
    ### Get input shapes
    IN_CHANNELS, OUT_CHANNELS = W.shape
    B, _ = X.shape

    ### Input Checking
    assert W.is_contiguous() and X.is_contiguous(), "Must have contiguous tensors"

    ### Allocates output destination
    D = torch.empty((BATCH_SIZE, OUT_CHANNELS), device=W.device, dtype=torch.float32, requires_grad=False).contiguous()

    ### 3D Kernel launch grid
    grid = lambda META: (triton.cdiv(IN_CHANNELS, META['BLOCK_M']), 1, 1)

    ### Launch it!
    kernel = _linear_fwd_kernel[grid](
        ### Standard params
        W, X, D
        W.stride(0), W.stride(1),
        X.stride(0), X.stride(1),
        D.stride(0), D.stride(1),
        ### Metaparameters go here
        BATCH_SIZE,
		
    )

    return D


if __name__ == "__main__":
