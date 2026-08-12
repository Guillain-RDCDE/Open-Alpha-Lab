"""Study 874 — IPO-Price Anchoring.

Behavioural claim: investors **anchor** on an IPO's headline **offer price** (the round
number splashed across the prospectus and the ticker's first-day story). Two testable
consequences fall out of that anchor:

* **Anchoring pull.** The offer price is a psychological reference; a stock stretched far
  *above* its offer should feel "expensive" and get pulled back down, one that has fallen
  *below* should feel "cheap" and get pulled back up — i.e. the forward return should be
  **negatively** related to the current gap-from-offer. A slow reversion toward the anchor.
* **Below-offer drag.** Crossing *below* the offer breaks the IPO's founding promise
  ("everyone who bought the deal is now under water"); disposition/loss-aversion lore says
  such names carry a **persistent drag** until they reclaim the line.

* ``data``     — a CURATED table of ~45 well-known recent US IPOs (ticker, public **offer /
                 reference price**, first-trade date), hard-coded from the public record, plus
                 the yfinance daily-close tape for those names and ``SPY`` (the market leg),
                 cached under this study's own ``_cache/``; and a deterministic seeded
                 synthetic panel with a TUNABLE planted anchoring pull (null at ``edge=0``).
* ``strategy`` — the name-month event panel (gap-from-offer, below-offer flag, forward
                 market-adjusted return), a Fama-MacBeth cross-sectional anchoring slope, a
                 below-offer / above-offer basket spread, the HAC/Newey-West inference
                 primitives, a permutation placebo, a costed timer, and the synthetic control.

Low N (~45 names, heavily one-cohort) → low power → the honest prior is **None**. The
synthetic control only proves the machinery is unbiased; the stamp comes from the real tape.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
