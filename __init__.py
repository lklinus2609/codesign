# GBC (Gradient-Based Co-Design) for G1 Humanoid
# Entry point: codesign_g1_unified.py

from .g1_mjcf_modifier import G1MJCFModifier, SYMMETRIC_PAIRS, NUM_DESIGN_PARAMS

__all__ = [
    "G1MJCFModifier",
    "SYMMETRIC_PAIRS",
    "NUM_DESIGN_PARAMS",
]
