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


static_config = [
    triton.Config({'BLOCK_NB' : nb, 'BLOCK_Q' : bq}, num_stages=s, num_warps=w) \
    for nb in [16]\
    for bq in [16]\
    for s in [2]\
    for w in [4]\
]


@triton.autotune(static_config, key=['N', 'B', 'Q'])
@triton.jit
def _collapsed_dim_add(
    X,  # Input tensor: (N, B, Q)
    D,  # Output tensor: (N, B, Q)
    stride_xn, stride_xb, stride_xq,
    stride_dn, stride_db, stride_dq,
    N : tl.constexpr,
    B : tl.constexpr,
    Q : tl.constexpr,
    BLOCK_NB: tl.constexpr,  # Tile size for batch and spatial dimensions
    BLOCK_Q : tl.constexpr,
):
    ### Define block pointers
    pid_nb = tl.program_id(0)

    offset_NB = pid_nb * BLOCK_NB

    ### Input Pointer
    X_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(N*B, Q),
        strides=(stride_xb, stride_xq),
        offsets=(offset_NB, 0),
        block_shape=(BLOCK_NB, BLOCK_Q),
        order=(1, 0)
    )
    
    ### Destination Pointer
    D_block_ptr = tl.make_block_ptr(
        base=D,
        shape=(N*B, Q),
        strides=(stride_db, stride_dq),
        offsets=(offset_NB, 0),
        block_shape=(BLOCK_NB, BLOCK_Q),
        order=(1, 0)
    )

    ### Loop over all tiles of K dimension
    for k in range(0, tl.cdiv(Q, BLOCK_Q)):
        ### Grab X
        x = tl.load(X_block_ptr, padding_option="zero", boundary_check=(0,1))

        ### Store 2X in D
        tl.store(D_block_ptr, x + x, boundary_check=(0,1))

        ### Advance the block pointers for the next iteration in GEMM K dimension
        X_block_ptr = tl.advance(X_block_ptr, (0, BLOCK_Q))
        D_block_ptr = tl.advance(D_block_ptr, (0, BLOCK_Q))


###
### Multi-headed Attention Forward
###
def collapsed_dim_add(
    X : torch.Tensor,
    D : torch.Tensor = None,
) -> torch.Tensor:
    ### Get input shapes
    N, B, Q = X.shape

    ### Allocates output and check input
    if not D:
        D = torch.empty((N, B, Q), device=X.device, dtype=X.dtype, requires_grad=False)

    ### Implicit 2D GEMM Grid
    grid = lambda META: (
        triton.cdiv(N * B, META['BLOCK_NB']),
        1,
        1,
    )

    kernel = _collapsed_dim_add[grid](
        ### Standard params
        X, D,
        X.stride(0), X.stride(1), X.stride(2),
        D.stride(0), D.stride(1), D.stride(2),
        N,
        B,
        Q,
        ### Metaparameters
    )   

    return D


if __name__ == "__main__":
    device = torch.device("cuda:0")
    N, B, Q = 32, 32, 32
    X = torch.ones((N, B, Q), device=device, dtype=torch.float32, requires_grad=False)
    D = collapsed_dim_add(X, None)

    print(f"X: {X[0,0,:]}")
    print(f"============")
    print(f"D: {D[0,0,:]}")
    print(f"============")
    print(f"MaxAbsDiff wrt reference X+X: {torch.max(torch.abs((X + X) - D))}")