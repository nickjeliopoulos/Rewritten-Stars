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
def silu_activation_fwd_kernel(x : tl.tensor) -> tl.tensor:
    """
    Computes the element-wise function: {silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}
    """
    return x * (1.0 / (1.0 + tl.exp(-x)))


GELU_COEFF = ( 2.0 / torch.pi ) ** 0.5


###
### Adapted to always use approximation
###
@triton.jit
def gelu_activation_fwd_kernel(x : tl.tensor) -> tl.tensor:
    """
    Computes the approximate formulation of GeLU: {gelu}(x) = \frac{x}{2} \left(1 + \mathrm{tanh} \left( \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)
    """
    return 0.5 * x * ( 1.0 + tl.libdevice.tanh( GELU_COEFF * (x + 0.044715 * x * x * x) ) )


###
### Configurations
###
minimal_autotune_configs = [
    triton.Config({'BLOCK_O': BM, 'BLOCK_I': BK}, num_stages=s, num_warps=w) \
    for BM in [32, 64, 128, 256]\
    for BK in [32, 64, 128, 256]\
    for s in [3, 7]\
    for w in [4, 8]\
]


static_config = [
    triton.Config({'BLOCK_O': 128, 'BLOCK_I': 128}, num_stages=3, num_warps=4)
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
        W,     # (O, I) (must transpose)
        X,     # (B, I) 
        D,     # (B, O)
        stride_wo, stride_wi,
        stride_xb, stride_xi,
        stride_db, stride_do,
        IN_CHANNELS: tl.constexpr,
        OUT_CHANNELS: tl.constexpr,
        BATCH_SIZE: tl.constexpr,
        ### Metaparams
        BLOCK_O: tl.constexpr,
        BLOCK_I: tl.constexpr,
):
    ### Which program are we?
    # program_I_id = tl.program_id(axis=1)
    program_O_id = tl.program_id(axis=0)

    ### Compute any offsets
    O_offset = program_O_id * BLOCK_O
    # I_offset = program_I_id * BLOCK_I 
    I_offset = 0

    ### Create pointers!
    WT_block_ptr = tl.make_block_ptr(
        base=W + (stride_wo * program_O_id),
        shape=(IN_CHANNELS, OUT_CHANNELS),
        strides=(stride_wi, stride_wo),
        offsets=(0, O_offset),
        block_shape=(BLOCK_I, BLOCK_O),
        order=(0, 1),
    )

    X_block_ptr = tl.make_block_ptr(
        base=X + ( 0 ),
        shape=(BATCH_SIZE, IN_CHANNELS),
        strides=(stride_xb, stride_xi),
        offsets=(0, 0),
        block_shape=(BATCH_SIZE, BLOCK_I),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        base=D + (stride_do * program_O_id),
        shape=(BATCH_SIZE, OUT_CHANNELS),
        strides=(stride_db, stride_do),
        offsets=(0, O_offset),
        block_shape=(BATCH_SIZE, BLOCK_O),
        order=(1, 0),
    )

    ### Accumulation term
    accum = tl.zeros((BATCH_SIZE, BLOCK_O), dtype=tl.float32)

    for k in range(0, tl.cdiv(IN_CHANNELS, BLOCK_I)):
        ### Load w
        wt = tl.load(WT_block_ptr)

        ### Load chunk of x
        x = tl.load(X_block_ptr, padding_option="zero", boundary_check=(0,))

        accum = tl.dot(x, wt, acc=accum)
    
        ### Advance pointers
        X_block_ptr = tl.advance(X_block_ptr, (0, BLOCK_I))
        WT_block_ptr = tl.advance(WT_block_ptr, (BLOCK_I, 0))

    ### Epilogue
    tl.store(D_block_ptr, accum, boundary_check=(0,))


###
### Multi-headed Attention Forward
###
def parallel_linear_fwd_fp32(
    W : torch.Tensor, # (OUT_CHANNELS, IN_CHANNELS)
    X : torch.Tensor, # (BATCH_SIZE, IN_CHANNELS)
    D : torch.Tensor = None, # (BATCH_SIZE, OUT_CHANNELS)
) -> torch.Tensor:    
    ### Get input shapes
    OUT_CHANNELS, IN_CHANNELS  = W.shape
    BATCH_SIZE, _ = X.shape

    ### Input Checking
    assert W.is_contiguous() and X.is_contiguous(), "Must have contiguous tensors"
    assert W.device == X.device, "Must have same device"
    assert W.dtype == X.dtype, "Must have same dtype"

    ### Allocates output destination
    if not D:
        D = torch.empty((BATCH_SIZE, OUT_CHANNELS), device=W.device, dtype=W.dtype, requires_grad=False).contiguous()

    ### 3D Kernel launch grid
    grid = lambda META: (triton.cdiv(OUT_CHANNELS, META['BLOCK_O']), 1, 1)

    ### Launch it!
    kernel = _parallel_linear_fwd_kernel_fp32[grid](
        ### Standard params
        W, X, D,
        W.stride(0), W.stride(1),
        X.stride(0), X.stride(1),
        D.stride(0), D.stride(1),
        IN_CHANNELS,
        OUT_CHANNELS,
        ### Metaparameters go here
        BATCH_SIZE,
    )

    return D


if __name__ == "__main__":
    device = torch.device("cuda:0")
    B, I, O = 16, 64, 128

    W = torch.randn((O,I), device=device, dtype=torch.float32, requires_grad=False)
    X = torch.randn((B,I), device=device, dtype=torch.float32, requires_grad=False)

    triton_linear = parallel_linear_fwd_fp32(W, X)
    torchf_linear = torch.nn.functional.linear(X, W)

    print(f"Triton shape: {triton_linear.shape}")
    print(f"Triton: {triton_linear}")

    print(f"torch.F shape: {torchf_linear.shape}")
    print(f"torch.F: {torchf_linear}")