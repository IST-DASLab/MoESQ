from .quant import Quantizer, NvFp4Quantizer, NVFP4_BLOCK_SIZE, quantize, quantize_nvfp4, FP4_CODEBOOK, FP4_MAX
from .gptq import GPTQ, make_quantizer, rtn_quantize, random_quantize
from .obr import OBR
from .jsq import JSQ


def make_initializer(config, layer, name, device, dtype):
    if config.init.method == "obr":
        return OBR(layer, name, config, device, dtype)
    if config.init.method == "jsq":
        return JSQ(layer, name, config, device, dtype)
    return GPTQ(layer, name, config, device, dtype)


__all__ = [
    "Quantizer", "NvFp4Quantizer", "NVFP4_BLOCK_SIZE",
    "quantize", "quantize_nvfp4", "FP4_CODEBOOK", "FP4_MAX",
    "GPTQ", "OBR", "JSQ", "make_initializer", "make_quantizer", "rtn_quantize", "random_quantize",
]
