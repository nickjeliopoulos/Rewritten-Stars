#pragma once
#include <torch/extension.h>
#include "cutlass/cutlass.h"
#include "cutlass/cutlass.h"

namespace stars::operators{
    namespace{
    }
    /*
    W: (O, I, 2)
    X: (B, I, H, W) with channels_last format
    D: (B, O, H, W) with channels_last format
    */
    torch::Tensor parallel_linear_fwd_sm80_fp32_mul(
        torch::Tensor W,
        torch::Tensor X,
        torch::Tensor D,
    );

    /*
    W: (O, I, 2)
    X: (B, H, W, I) with channels_last format
    D: (B, H, W, O) with channels_last format
    */
    torch::Tensor parallel_linear_fwd_sm80_fp32_add(
        torch::Tensor W,
        torch::Tensor X,
        torch::Tensor D,
    );

    // PyBind!
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
		m.doc() = "CUDA + CUTLASS Operators for Rewrite-The-Stars";
        m.def("parallel_linear_fwd_sm80_fp32_add", &parallel_linear_fwd_sm80_fp32_add, "Fused Parallel Linear Layer (add)");
        m.def("parallel_linear_fwd_sm80_fp32_mul", &parallel_linear_fwd_sm80_fp32_mul, "Fused Parallel Linear Layer (mul)");
    }
}