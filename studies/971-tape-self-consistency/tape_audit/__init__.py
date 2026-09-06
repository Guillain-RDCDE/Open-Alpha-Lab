"""Study 971 — Does the Tape Agree With Itself?.

Every study on this desk starts by trusting a free data feed. This one audits it:
does the daily tape compound to the weekly tape, does the monthly tape agree with either, does
price plus dividends reconstruct the total-return series, do the split adjustments line up, and
are any sessions simply missing? None of these questions has an opinion in it — each has a right
answer that can be checked.

- :mod:`tape_audit.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`tape_audit.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
