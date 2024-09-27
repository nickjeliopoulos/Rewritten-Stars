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


