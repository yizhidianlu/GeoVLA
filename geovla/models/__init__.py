# -*- coding: utf-8 -*-
"""geovla.models — model variants."""
from geovla.models.gate1_models import VariantA, VariantB, VariantC, trainable_param_count

# Variant D (Qwen+LoRA+SigLIP) is loaded lazily — its imports require peft + the
# downloaded Qwen weights, which the gate1 entrypoints should not pay for.
def _lazy_d_rgb(*args, **kwargs):
    from geovla.models.qwen_vla import VariantD_RGB
    return VariantD_RGB(*args, **kwargs)


def _lazy_d_rgbp(*args, **kwargs):
    from geovla.models.qwen_vla import VariantD_RGBProprio
    return VariantD_RGBProprio(*args, **kwargs)


VariantD_RGB = _lazy_d_rgb
VariantD_RGBProprio = _lazy_d_rgbp


def _lazy_e_rgb(*args, **kwargs):
    from geovla.models.openvla_wrap import VariantE_RGB
    return VariantE_RGB(*args, **kwargs)


def _lazy_e_rgbp(*args, **kwargs):
    from geovla.models.openvla_wrap import VariantE_RGBProprio
    return VariantE_RGBProprio(*args, **kwargs)


VariantE_RGB = _lazy_e_rgb
VariantE_RGBProprio = _lazy_e_rgbp

__all__ = ["VariantA", "VariantB", "VariantC",
           "VariantD_RGB", "VariantD_RGBProprio",
           "VariantE_RGB", "VariantE_RGBProprio",
           "trainable_param_count"]
