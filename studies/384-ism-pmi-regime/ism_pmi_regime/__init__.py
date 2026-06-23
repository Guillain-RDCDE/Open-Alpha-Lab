"""Study 384 — ISM-PMI-Regime ("only own stocks when manufacturing PMI > 50").

Macro-nowcasting folklore: the ISM Manufacturing PMI's 50 line is supposed to be a
regime switch for equities — own stocks above 50 (expansion), sit in cash below 50
(contraction). The true ISM survey is proprietary (ISM sells it) and the usual free
fallback (FRED ``NAPM`` / regional-Fed diffusion indices) is unreachable from this
environment, so we CONSTRUCT a transparent **PMI proxy**: a monthly diffusion index
across a fixed industrial/manufacturing basket (the share of names with positive
3-month momentum, on a 0–100 PMI-like axis), and test the above-50 regime against the
below-50 regime and against the unconditional base rate.

The decisive finding is statistical: the above-50 *excess* over plain buy-and-hold is
small, and most of the "PMI works" story is the market's own up-drift plus the fact
that PMI is above 50 most of the time anyway — own-stocks-above-50 is close to
own-stocks.

See :mod:`ism_pmi_regime.data` (real basket loader + PMI proxy + deterministic synthetic
control) and :mod:`ism_pmi_regime.strategy` (regime split, Welch t + block-placebo null,
the deployable timing strategy, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
