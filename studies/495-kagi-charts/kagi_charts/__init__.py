"""Study 495 — Kagi Charts (yin/yang line reversals).

A mechanical, falsifiable encoding of the Japanese Kagi chart. A Kagi line runs in one
direction until price reverses by at least a fixed amount (the *reversal* threshold);
the line then turns and switches **thickness** when it breaks a prior shoulder/waist —
a thick **yang** line (a higher high broken) vs a thin **yin** line (a lower low broken).
The folklore says you go **long on a yang switch** (thick = uptrend confirmed) and **flat
on yin** (thin = downtrend). We test that switch as a forward-return study against a
drift-matched random-entry baseline and a threshold-scramble placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
