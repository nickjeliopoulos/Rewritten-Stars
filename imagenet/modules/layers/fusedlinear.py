import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from .ops.triton.FusedParallelLinear import parallel_linear_fwd_fp32


###
### Fused Linear Projections of input x followed by SiLU for one projection
###
class FusedParallelLinear(nn.Module):
	def __init__(self, dim : int, mlp_ratio : int = 3, bias : bool = False, dtype : torch.dtype = torch.float32):
		super().__init__()
		### Linear Weights
		self.W = torch.empty(size=(2, dim, mlp_ratio * dim), dtype=dtype)
		trunc_normal_(self.W, std=0.02)


	def forward(self, x : torch.Tensor, d : torch.Tensor = None, op : str = "mul") -> torch.Tensor:
		# print(f"x.shape {x.shape}")
		"""
		x : input tensor
		d : (optional) destination tensor
		op : "mul" or "add"
		"""
		return parallel_linear_fwd_fp32(self.W, x, d, op)