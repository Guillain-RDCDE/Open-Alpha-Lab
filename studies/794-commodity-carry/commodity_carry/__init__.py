"""Study 794 — Commodity Carry (cross-sectional roll yield).

Does an ex-ante commodity carry signal — read from the **real futures term
structure** (EIA WTI RCLC1-4 + Henry Hub RNGC1-4) — predict which commodity earns
more, cross-sectionally? Returns are the **investable ETF** total returns (USO for
WTI front, UNG for gas front); a second ETF pair (USO vs the 12-month-laddered USL)
is the direct roll-drag proxy.

PROXY CAVEAT (stated everywhere): clean, broad historical commodity term structure is
not freely available, so this study is a **two-name energy proxy** for the general
cross-sectional carry premium (Gorton-Rouwenhorst 2006; Erb-Harvey 2006; Koijen et al.
2018). A two-name cross-section is the thinnest possible and is underpowered by
construction — read the verdict with that in mind.
"""

from . import data, strategy  # noqa: F401
