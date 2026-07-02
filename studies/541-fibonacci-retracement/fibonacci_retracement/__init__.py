"""Study 541 — Fibonacci-Retracement.

Do the 38.2% / 61.8% (and 23.6% / 50%) Fibonacci retracement levels really mark where a pullback
stops and price reverses — or is any round-ish retracement fraction just as good?

Public surface:
- ``data.synthetic_tape`` / ``data.load_real`` / ``data.load_basket`` / ``data.fingerprint``
- ``strategy`` — the ZigZag swing detector, the retracement-touch reversal test, the placebo of
  arbitrary fractions, costs, the robustness sweep and the seed-robust synthetic positive control.
"""

from __future__ import annotations

from . import data, strategy

__all__ = ["data", "strategy"]
