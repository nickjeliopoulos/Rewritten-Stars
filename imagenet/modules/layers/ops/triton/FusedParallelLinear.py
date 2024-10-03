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
    return ( x ) / (1.0 + tl.exp(-x))


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
    for BM in [32]\
    for BK in [32]\
    for s in [2, 4]\
    for w in [4, 8]\
]


static_config = [
    triton.Config({'BLOCK_O': 128, 'BLOCK_I': 128}, num_stages=3, num_warps=4)
]


###
### Fused FC Layer
### Current design: Interleave computation
### TODO: Need to test whether assigning different programs a single FC is better, of if interleaving is more eeffective
### TODO: Unsure of which dim to check for storing d_accum in D
###
@triton.autotune(minimal_autotune_configs, key=['OUT_CHANNELS', 'HEIGHT', 'WIDTH', 'IN_CHANNELS', 'BATCH_SIZE'])
@triton.jit
def _parallel_linear_fwd_kernel_fp32_add(
        W,       # (2, O, I)
        X,       # (B, I, H, W) 
        D,       # (B, O, H, W)
        stride_wp, stride_wo, stride_wi,
        stride_xb, stride_xi, stride_xh, stride_xw,
        stride_db, stride_do, stride_dh, stride_dw,
        IN_CHANNELS: tl.constexpr,
        OUT_CHANNELS: tl.constexpr,
        BATCH_SIZE: tl.constexpr,
        HEIGHT : tl.constexpr,
        WIDTH : tl.constexpr,
        ### Metaparams
        BLOCK_O: tl.constexpr,
        BLOCK_I: tl.constexpr,
):
    ### Kernel-wide constants
    dtype = tl.float32

    ### Which program are we?
    # program_I_id = tl.program_id(axis=1)
    program_O_id = tl.program_id(axis=0)
    program_H_id = tl.program_id(axis=1)
    program_W_id = tl.program_id(axis=2)

    ### Compute any offsets
    O_offset = program_O_id * BLOCK_O

    ### Create pointers!
    ### TODO: Coalesce memory loading 
    WT1_block_ptr = tl.make_block_ptr(
        base=W + ( stride_wp * 0 ),
        shape=(IN_CHANNELS, OUT_CHANNELS),
        strides=(stride_wi, stride_wo),
        offsets=(0, O_offset),
        block_shape=(BLOCK_I, BLOCK_O),
        order=(0, 1),
    )

    WT2_block_ptr = tl.make_block_ptr(
        base=W + ( stride_wp * 1 ),
        shape=(IN_CHANNELS, OUT_CHANNELS),
        strides=(stride_wi, stride_wo),
        offsets=(0, O_offset),
        block_shape=(BLOCK_I, BLOCK_O),
        order=(0, 1),
    )

    X_block_ptr = tl.make_block_ptr(
        base=X + (stride_xh * program_H_id) + (stride_xw * program_W_id),
        shape=(BATCH_SIZE, IN_CHANNELS),
        strides=(stride_xb, stride_xi),
        offsets=(0, 0),
        block_shape=(BATCH_SIZE, BLOCK_I),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        base=D + (stride_dh * program_H_id) + (stride_dw * program_W_id),
        shape=(BATCH_SIZE, OUT_CHANNELS),
        strides=(stride_db, stride_do),
        offsets=(0, O_offset),
        block_shape=(BATCH_SIZE, BLOCK_O),
        order=(1, 0),
    )

    ### Accumulation terms
    t0_accum = tl.zeros((BATCH_SIZE, BLOCK_O), dtype=dtype)
    t1_accum = tl.zeros((BATCH_SIZE, BLOCK_O), dtype=dtype)

    for k in range(0, tl.cdiv(IN_CHANNELS, BLOCK_I)):
        ### Load chunk of x
        x = tl.load(X_block_ptr, padding_option="zero", boundary_check=(0,1))

        ### Load w1, w2
        wt1 = tl.load(WT1_block_ptr, padding_option="zero", boundary_check=(0,1))
        wt2 = tl.load(WT2_block_ptr, padding_option="zero", boundary_check=(0,1))

        t0_accum = tl.dot(x, wt1, acc=t0_accum, input_precision="ieee")
        t1_accum = tl.dot(x, wt2, acc=t1_accum, input_precision="ieee")

        ### Advance pointers
        X_block_ptr = tl.advance(X_block_ptr, (0, BLOCK_I))
        WT1_block_ptr = tl.advance(WT1_block_ptr, (BLOCK_I, 0))
        WT2_block_ptr = tl.advance(WT2_block_ptr, (BLOCK_I, 0))


    ### Epilogue
    tl.store(D_block_ptr, silu_activation_fwd_kernel(t0_accum) + t1_accum, boundary_check=(0,1))


###
### Fused FC Layer
###
@triton.autotune(minimal_autotune_configs, key=['OUT_CHANNELS', 'HEIGHT', 'WIDTH', 'IN_CHANNELS', 'BATCH_SIZE'])
@triton.jit
def _parallel_linear_fwd_kernel_fp32_mul(
        W,       # (2, O, I)
        X,       # (B, I, H, W) 
        D,       # (B, O, H, W)
        stride_wp, stride_wo, stride_wi,
        stride_xb, stride_xi, stride_xh, stride_xw,
        stride_db, stride_do, stride_dh, stride_dw,
        IN_CHANNELS: tl.constexpr,
        OUT_CHANNELS: tl.constexpr,
        BATCH_SIZE: tl.constexpr,
        HEIGHT : tl.constexpr,
        WIDTH : tl.constexpr,
        ### Metaparams
        BLOCK_O: tl.constexpr,
        BLOCK_I: tl.constexpr,
):
    ### Kernel-wide constants
    dtype = tl.float32

    ### Which program are we?
    # program_I_id = tl.program_id(axis=1)
    program_O_id = tl.program_id(axis=0)
    program_H_id = tl.program_id(axis=1)
    program_W_id = tl.program_id(axis=2)

    ### Compute any offsets
    O_offset = program_O_id * BLOCK_O

    ### Create pointers!
    ### TODO: Coalesce memory loading 
    WT1_block_ptr = tl.make_block_ptr(
        base=W + ( stride_wp * 0 ),
        shape=(IN_CHANNELS, OUT_CHANNELS),
        strides=(stride_wi, stride_wo),
        offsets=(0, O_offset),
        block_shape=(BLOCK_I, BLOCK_O),
        order=(0, 1),
    )

    WT2_block_ptr = tl.make_block_ptr(
        base=W + ( stride_wp * 1 ),
        shape=(IN_CHANNELS, OUT_CHANNELS),
        strides=(stride_wi, stride_wo),
        offsets=(0, O_offset),
        block_shape=(BLOCK_I, BLOCK_O),
        order=(0, 1),
    )

    X_block_ptr = tl.make_block_ptr(
        base=X + (stride_xh * program_H_id) + (stride_xw * program_W_id),
        shape=(BATCH_SIZE, IN_CHANNELS),
        strides=(stride_xb, stride_xi),
        offsets=(0, 0),
        block_shape=(BATCH_SIZE, BLOCK_I),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        base=D + (stride_dh * program_H_id) + (stride_dw * program_W_id),
        shape=(BATCH_SIZE, OUT_CHANNELS),
        strides=(stride_db, stride_do),
        offsets=(0, O_offset),
        block_shape=(BATCH_SIZE, BLOCK_O),
        order=(1, 0),
    )

    ### Accumulation terms
    t0_accum = tl.zeros((BATCH_SIZE, BLOCK_O), dtype=dtype)
    t1_accum = tl.zeros((BATCH_SIZE, BLOCK_O), dtype=dtype)

    for k in range(0, tl.cdiv(IN_CHANNELS, BLOCK_I)):
        ### Load chunk of x
        x = tl.load(X_block_ptr, padding_option="zero", boundary_check=(0,1))

        ### Load w1, w2
        wt1 = tl.load(WT1_block_ptr, padding_option="zero", boundary_check=(0,1))
        wt2 = tl.load(WT2_block_ptr, padding_option="zero", boundary_check=(0,1))

        t0_accum = tl.dot(x, wt1, acc=t0_accum, input_precision="ieee")
        t1_accum = tl.dot(x, wt2, acc=t1_accum, input_precision="ieee")

        ### Advance pointers
        X_block_ptr = tl.advance(X_block_ptr, (0, BLOCK_I))
        WT1_block_ptr = tl.advance(WT1_block_ptr, (BLOCK_I, 0))
        WT2_block_ptr = tl.advance(WT2_block_ptr, (BLOCK_I, 0))


    ### Epilogue
    tl.store(D_block_ptr, silu_activation_fwd_kernel(t0_accum) * t1_accum, boundary_check=(0,1))


###
### Multi-headed Attention Forward
###
def parallel_linear_fwd_fp32(
    W : torch.Tensor, # (2, OUT_CHANNELS, IN_CHANNELS)
    X : torch.Tensor, # (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    D : torch.Tensor = None, # (BATCH_SIZE, OUT_CHANNELS, HEIGHT, WIDTH)
    op : str = "add",
) -> torch.Tensor:    
    ### Get input shapes
    _, OUT_CHANNELS, _  = W.shape
    BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH = X.shape

    ### Input Checking
    assert W.is_contiguous() and X.is_contiguous(), "Must have contiguous tensors"
    assert W.device == X.device, "Must have same device"
    assert W.dtype == X.dtype, "Must have same dtype"
    assert op in ["add", "mul"], "Invalid op"

    ### Allocates output destination
    if not D:
        D = torch.empty((BATCH_SIZE, OUT_CHANNELS, HEIGHT, WIDTH), device=W.device, dtype=W.dtype, requires_grad=False).contiguous()

    ### NOTE: 3D Kernel launch grid, we can change the 2nd dimension in case we decide to not do interleaved computation
    ### NOTE: Can use jobid to decide behavior by doing that
    grid = lambda META: (triton.cdiv(OUT_CHANNELS, META['BLOCK_O']), HEIGHT, WIDTH)

    ### Launch it!
    if op == "add":
        kernel = _parallel_linear_fwd_kernel_fp32_add[grid](
            ### Standard params
            W, X, D,
            W.stride(0), W.stride(1), W.stride(2),
            X.stride(0), X.stride(1), X.stride(2), X.stride(3),
            D.stride(0), D.stride(1), D.stride(2), D.stride(3),
            IN_CHANNELS,
            OUT_CHANNELS,
            BATCH_SIZE,
            HEIGHT,
            WIDTH,
        )
    elif op == "mul":
        kernel = _parallel_linear_fwd_kernel_fp32_mul[grid](
            ### Standard params
            W, X, D,
            W.stride(0), W.stride(1), W.stride(2),
            X.stride(0), X.stride(1), X.stride(2), X.stride(3),
            D.stride(0), D.stride(1), D.stride(2), D.stride(3),
            IN_CHANNELS,
            OUT_CHANNELS,
            BATCH_SIZE,
            HEIGHT,
            WIDTH,
        )   

    return D


if __name__ == "__main__":
    def benchmark_latency(W, X, op : str, N : int = 128):
        def _torch_mul(W, X):
            A = torch.nn.functional.conv2d(X, W[0,:,:,None,None])
            B = torch.nn.functional.conv2d(X, W[1,:,:,None,None])
            C = torch.nn.functional.silu(A) * B
            return C

        def _torch_add(W, X):
            A = torch.nn.functional.conv2d(X, W[0,:,:,None,None])
            B = torch.nn.functional.conv2d(X, W[1,:,:,None,None])
            C = torch.nn.functional.silu(A) + B
            return C

        t0 = benchmark.Timer(
            stmt="f(W,X)",
            globals={'f' : _torch_mul if op == "mul" else _torch_add, 'W' : W, 'X' : X},
        )

        torch_measurement = t0.timeit(N)

        t1 = benchmark.Timer(
            stmt="f(W,X,None,op)",
            globals={'f' : parallel_linear_fwd_fp32, 'W' : W, 'X' : X, 'op' : op},
        )

        triton_measurement = t1.timeit(N)

        return (torch_measurement.median * 1e6, triton_measurement.median * 1e6)


    torch.set_printoptions(sci_mode=True)

    device = torch.device("cuda:0")
    B, I, O, HEIGHT, WIDTH = 16, 64, 128, 8, 8
    W = torch.randn((2,O,I), device=device, dtype=torch.float32, requires_grad=False)
    X = torch.randn((B,I,HEIGHT,WIDTH), device=device, dtype=torch.float32, requires_grad=False)

    print(f"=== Interleaved Linear Mul ===")

    triton_parallel_linear_mul = parallel_linear_fwd_fp32(W, X, op="mul")
    torchf_linear_1 = torch.nn.functional.conv2d(X, W[0,:,:,None,None])
    torchf_linear_2 = torch.nn.functional.conv2d(X, W[1,:,:,None,None])
    torchf_linear_mul = torch.nn.functional.silu(torchf_linear_1) * torchf_linear_2

    torch_us, triton_us = benchmark_latency(W, X, op="mul")

    print(f"max diff: {torch.max(torchf_linear_mul-triton_parallel_linear_mul)}")
    print(f"latency torch / triton: {torch_us:.3f} / {triton_us:.3f}")

    print(f"\n=== Interleaved Linear Add ===")

    triton_parallel_linear_add = parallel_linear_fwd_fp32(W, X, op="add")
    torchf_linear_1 = torch.nn.functional.conv2d(X, W[0,:,:,None,None])
    torchf_linear_2 = torch.nn.functional.conv2d(X, W[1,:,:,None,None])
    torchf_linear_add = torch.nn.functional.silu(torchf_linear_1) + torchf_linear_2

    torch_us, triton_us = benchmark_latency(W, X, op="add")

    print(f"max diff: {torch.max(triton_parallel_linear_add-torchf_linear_add)}")
    print(f"latency torch / triton: {torch_us:.3f} / {triton_us:.3f}")
    
    print(f"\n=== Done! ===")