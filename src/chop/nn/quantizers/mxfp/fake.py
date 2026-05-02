"""
Fake MXFP quantization operations.
"""

import torch
from torch import Tensor

from .meta import MXFPMeta


def extract_mxfp_components(
    tensor: Tensor, mxfp_meta: MXFPMeta, percentile: float = 1.0
) -> tuple[Tensor, Tensor]:
    """
    Returns:
        scales_uint: uint8, biased shared exponent per block (n_blocks, 1)
        elements_fp: fp32, denormalized minifloat-quantized values (n_blocks, B)
                     already divided by 2^shared_exp; compose multiplies it back.
    """
    tensor = tensor.float()
    B = mxfp_meta.block_size
    assert tensor.numel() % B == 0

    n_blocks = tensor.numel() // B

    sc_exp_bias = (1 << (mxfp_meta.scale_exp_bits - 1)) - 1
    sc_exp_max_biased = (1 << mxfp_meta.scale_exp_bits) - 1 - sc_exp_bias
    sc_exp_min_biased = -sc_exp_bias

    E = mxfp_meta.element_exp_bits
    M = mxfp_meta.element_frac_bits
    el_exp_bias = (1 << (E - 1)) - 1
    el_exp_max_biased = (1 << E) - 1 - el_exp_bias
    el_exp_min_biased = -el_exp_bias

    x = tensor.flatten().reshape(n_blocks, B)

    # Per-block shared scale: ceil(log2(per-block max)) clamped to scale range.
    per_block_max = x.abs().quantile(percentile, dim=1, keepdim=True) + 1e-9
    shared_exp = per_block_max.log2().ceil().to(torch.int32)
    shared_exp = shared_exp.clamp(sc_exp_min_biased, sc_exp_max_biased)

    # Encode shared scale as uint8.
    scales_uint = (shared_exp + sc_exp_bias).to(torch.uint8)

    # Normalize tensor by per-block scale; values now have max ~1.0 in magnitude.
    scales_fp = torch.exp2(shared_exp.float())
    q_normalized = x / scales_fp

    # Denormalized minifloat quantization (no implicit leading 1).
    sign = torch.sign(q_normalized + 1e-9)
    val = q_normalized.abs()
    el_exp = torch.ceil(torch.log2(val + 1e-9))
    el_exp = el_exp.clamp(el_exp_min_biased, el_exp_max_biased)
    mantissa = val / torch.exp2(el_exp)
    shift = float(1 << M)
    mantissa_int = (mantissa * shift).round()
    mantissa_int = mantissa_int.clamp(0, (1 << M) - 1)
    mantissa_q = mantissa_int / shift

    # Mask: very small values stay zero (avoid spurious tiny artifacts).
    is_zero = val < 1e-12
    elements_fp = torch.where(
        is_zero, torch.zeros_like(q_normalized), sign * torch.exp2(el_exp) * mantissa_q
    )

    return scales_uint, elements_fp.float()


def compose_mxfp_tensor(
    scales: Tensor,
    elements: Tensor,
    mxfp_meta: MXFPMeta,
    output_dtype: torch.dtype,
) -> Tensor:
    """
    Recompose tensor from (scales_uint8, elements_fp32). Multiplies the
    minifloat-quantized normalized values by 2^shared_exp.
    """
    assert scales.dtype == torch.uint8
    sc_exp_bias = (1 << (mxfp_meta.scale_exp_bits - 1)) - 1
    shared_exp = scales.to(torch.int32) - sc_exp_bias
    scales_fp = torch.exp2(shared_exp.float())

    dequantized = elements.float() * scales_fp
    dequantized = dequantized.flatten().to(output_dtype)
    return dequantized
