"""Study 1023 — The Cushing Glut.

Brent and WTI are both light sweet crude. One is loaded at Sullom Voe and priced off the North
Sea; the other is delivered into tanks at Cushing, Oklahoma. A barrel is a barrel, the argument
goes, so the gap between the two prices is pinned by what it costs to move oil between the two
places — and any stretch beyond that is a gift to whoever fades it.

For twenty-three years of this tape that was very nearly true. Then US shale production rose
faster than pipelines out of Cushing could be built, the tanks filled, and WTI fell to a $20+
discount that took years to close and never returned to its old home. This study measures the
tether, locates the moment it moved with a data-driven break test, and books what a disciplined
spread-fader would have lived through.

- :mod:`cushing.data` — the frozen monthly Brent/WTI tape from ``arch`` (SHA-256 pinned via
  :mod:`quantlab.bundled`) and a deterministic synthetic OU-spread generator with a planted
  level break.
- :mod:`cushing.strategy` — stationarity and cointegration tests, the sup-F / Bai–Perron break
  search, AR(1) half-lives, the real-time z-score fader, and the pre-registered verdict.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
