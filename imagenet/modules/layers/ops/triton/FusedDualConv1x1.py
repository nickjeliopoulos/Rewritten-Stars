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
autotune_large_workload_config = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK}, num_stages=s, num_warps=w) \
    for BM in [128, 256]\
    for BN in [64, 128]\
    for BK in [64, 128]\
    for s in [2, 4]\
    for w in [4, 8]\
]


autotune_small_workload_config = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for BK in [32, 64]\
    for s in [2, 4]\
    for w in [4, 8]\
]


static_config = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK}, num_stages=s, num_warps=w) \
    for BM in [64]\
    for BN in [32]\
    for BK in [32]\
    for s in [2]\
    for w in [4]\
]

@triton.autotune(static_config, key=['OUT_CHANNELS', 'BATCH_SIZE', 'HEIGHT', 'WIDTH'])
@triton.jit
def _dual_gemm_conv1x1_fwd_kernel_fp32(
    X,  # Input tensor: (B, I, H, W)
    W,  # Weight tensor: (2, O, I)
    D,  # Output tensor: (B, O, H, W)
    stride_wo, stride_wi, stride_wp,
    stride_xb, stride_xi, stride_xh, stride_xw,
    stride_db, stride_do, stride_dh, stride_dw,
    BATCH_SIZE: tl.constexpr,
    IN_CHANNELS: tl.constexpr,
    OUT_CHANNELS: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    OP : tl.constexpr,
    BLOCK_M: tl.constexpr,  # Tile size for batch and spatial dimensions
    BLOCK_N: tl.constexpr,  # Tile size for output channels
    BLOCK_K: tl.constexpr,  # Tile size for input channels
):
    ### Define block pointers
    pid_m = tl.program_id(0)  # Handles the m dimension (batch and spatial)
    pid_n = tl.program_id(1)  # Handles the n dimension (output channels)

    ### Compute block starting positions
    batch_id = pid_m // (HEIGHT * WIDTH)
    hw_id = pid_m % (HEIGHT * WIDTH)
    h_id = hw_id // WIDTH
    w_id = hw_id % WIDTH

    ### GEMM Problem Coordinates
    M = BATCH_SIZE * HEIGHT * WIDTH
    N = OUT_CHANNELS
    K = IN_CHANNELS

    ### GEMM Offsets
    offset_M = pid_m * BLOCK_M
    offset_N = pid_n * BLOCK_N

    ### Input Pointer
    X_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(M, K),
        strides=(stride_xw, stride_xi),
        offsets=(offset_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )

    ### W block pointer is 3D
    WT1_block_ptr = tl.make_block_ptr(
        base=W + (0 * stride_wp),
        shape=(K, N),
        strides=(stride_wi, stride_wo),
        offsets=(0, offset_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )

    WT2_block_ptr = tl.make_block_ptr(
        base=W + (1 * stride_wp),
        shape=(K, N),
        strides=(stride_wi, stride_wo),
        offsets=(0, offset_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )

    ### Destination Pointer
    D_block_ptr = tl.make_block_ptr(
        base=D,
        shape=(M, N),
        strides=(stride_dw, stride_do),
        offsets=(offset_M, offset_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    ### Initialize accumulator
    acc_w0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_w1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    ### Loop over all tiles of K dimension
    for k in range(0, tl.cdiv(IN_CHANNELS, BLOCK_K)):
        ### Grab X
        x = tl.load(X_block_ptr, padding_option="zero", boundary_check=(0,1))
        ### Load, Dot, Load, Dot
        wt1 = tl.load(WT1_block_ptr, padding_option="zero", boundary_check=(0,1))
        acc_w0 = tl.dot(x, wt1, acc=acc_w0, input_precision="ieee")
        wt2 = tl.load(WT2_block_ptr, padding_option="zero", boundary_check=(0,1))
        acc_w1 = tl.dot(x, wt2, acc=acc_w1, input_precision="ieee")

        ### Advance the block pointers for the next iteration in GEMM K dimension
        X_block_ptr = tl.advance(X_block_ptr, (0, BLOCK_K))
        WT1_block_ptr = tl.advance(WT1_block_ptr, (BLOCK_K, 0))
        WT2_block_ptr = tl.advance(WT2_block_ptr, (BLOCK_K, 0))

    ### Epilogue
    acc_w0 = silu_activation_fwd_kernel(acc_w0)

    ### Switch behavior based on op input
    ### op == 1 --> multiply
    ### op == 0 --> add
    if OP & 1:
        acc_w0 *= acc_w1
    else:
        acc_w0 += acc_w1

    ### Epilogue
    tl.store(D_block_ptr, acc_w0, boundary_check=(0,1))


###
### Multi-headed Attention Forward
###
def dual_conv1x1_fwd_fp32(
    W : torch.Tensor, # (OUT_CHANNELS, IN_CHANNELS, 2)
    X : torch.Tensor, # (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    D : torch.Tensor = None, # (BATCH_SIZE, OUT_CHANNELS, HEIGHT, WIDTH)
    op : str = "add",
) -> torch.Tensor:
    ### Get input shapes
    _, OUT_CHANNELS, _  = W.shape
    BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH = X.shape

    ### Input Checking
    assert W.device == X.device and "cuda" in str(X.device), "Must have same CUDA device"
    assert W.dtype == X.dtype, "Must have same dtype"
    assert op in ["add", "mul"], "Invalid op"

    ### Allocates output and check input
    if not D:
        D = torch.empty((BATCH_SIZE, OUT_CHANNELS, HEIGHT, WIDTH), device=W.device, dtype=W.dtype, requires_grad=False)#.contiguous(memory_format=torch.channels_last)
    else:
        assert "cuda" in str(D.device), "D must be CUDA device"
        assert D.dtype == X.dtype, "D must have same dtype has X"

    ### Implicit 2D GEMM Grid
    grid = lambda META: (
        triton.cdiv(BATCH_SIZE * HEIGHT * WIDTH, META['BLOCK_M']),
        triton.cdiv(OUT_CHANNELS, META['BLOCK_N']),
        1,
    )

    kernel = _dual_gemm_conv1x1_fwd_kernel_fp32[grid](
        ### Standard params
        W, X, D,
        W.stride(0), W.stride(1), W.stride(2),
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        D.stride(0), D.stride(1), D.stride(2), D.stride(3),
        BATCH_SIZE,
        IN_CHANNELS,
        OUT_CHANNELS,
        HEIGHT,
        WIDTH,
        OP=1 if op=="mul" else 0,
        ### Metaparameters
    )   

    return D

###
### TODO IMPORTANT! - Make sure W has the right shape (we are now implementing it as O,I,2 instead of 2,O,I)
###
if __name__ == "__main__":
    def benchmark_latency(W, X, op : str, N : int = 64):
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
            globals={'f' : dual_conv1x1_fwd_fp32, 'W' : W, 'X' : X, 'op' : op},
        )

        triton_measurement = t1.timeit(N)

        return (torch_measurement.median * 1e6, triton_measurement.median * 1e6)


    torch.set_printoptions(sci_mode=True)

    device = torch.device("cuda:0")
    B, I, O, HEIGHT, WIDTH = 32, 128, 256, 32, 32
    W = torch.randn((2,O,I), device=device, dtype=torch.float32, requires_grad=False)
    X = torch.randn((B,I,HEIGHT,WIDTH), device=device, dtype=torch.float32, requires_grad=False)#.contiguous(memory_format=torch.channels_last)

    print(f"=== Interleaved Linear Mul ===")

    triton_parallel_linear_mul = dual_conv1x1_fwd_fp32(W, X, op="mul")
    torchf_linear_1 = torch.nn.functional.conv2d(X, W[0,:,:,None,None])
    torchf_linear_2 = torch.nn.functional.conv2d(X, W[1,:,:,None,None])
    torchf_linear_mul = torch.nn.functional.silu(torchf_linear_1) * torchf_linear_2

    torch_us, triton_us = benchmark_latency(W, X, op="mul")

    print(f"max diff: {torch.max(torchf_linear_mul-triton_parallel_linear_mul)}")
    print(f"latency (us) torch / triton: {torch_us:.3f} / {triton_us:.3f}")

    print(f"\n=== Interleaved Linear Add ===")

    triton_parallel_linear_add = dual_conv1x1_fwd_fp32(W, X, op="add")
    torchf_linear_1 = torch.nn.functional.conv2d(X, W[0,:,:,None,None])
    torchf_linear_2 = torch.nn.functional.conv2d(X, W[1,:,:,None,None])
    torchf_linear_add = torch.nn.functional.silu(torchf_linear_1) + torchf_linear_2

    torch_us, triton_us = benchmark_latency(W, X, op="add")

    print(f"max diff: {torch.max(triton_parallel_linear_add-torchf_linear_add)}")
    print(f"latency (us) torch / triton: {torch_us:.3f} / {triton_us:.3f}")
    
    print(f"\n=== Done! ===")