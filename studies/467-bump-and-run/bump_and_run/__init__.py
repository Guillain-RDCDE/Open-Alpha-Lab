"""Study 467 — Bump-and-Run Reversal (BARR).

A mechanical, falsifiable encoding of Thomas Bulkowski's *bump-and-run reversal*: a
gentle **lead-in trendline**, a speculative **bump** (price steepens and surges away
from the line), then a **break back below the trendline** that is supposed to forecast a
reversal. The folklore rule is to **short** the trendline break (entered next close). We
test that as a forward-return study against random-entry and shuffled-geometry placebos,
with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
