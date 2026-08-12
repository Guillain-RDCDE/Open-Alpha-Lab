"""Study 868 — Global Curve-Slope Carry.

A steep yield curve is supposed to pay a duration holder to sit still — *roll + carry*.
Cross-sectionally across bond markets: rank US and international sovereign-bond ETFs by a
curve-steepness / carry proxy (a realized yield-to-duration measure) and go **long the
high-carry / steep-curve** sleeves, **short the flat / low-carry** ones. This package
holds the data layer (:mod:`curve_slope_carry.data`) and the strategy + inference
(:mod:`curve_slope_carry.strategy`).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
