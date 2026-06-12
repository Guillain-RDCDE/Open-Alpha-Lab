"""Study 65 — Scorecard: Piotroski's F-score on tradable large caps.

The F-score (Piotroski 2000) sums nine fundamental-health signals into a 0–9 score; high-F firms are
meant to beat low-F firms. On large-cap S&P survivors the spread doesn't hold — long high-F / short
low-F lost ~3.4%/yr (insignificant, wrong sign), because the original effect is a small-cap *value*
phenomenon: on profitable, crowded large-cap names the high scores are already fully priced.
"""
from . import data, strategy  # noqa: F401
