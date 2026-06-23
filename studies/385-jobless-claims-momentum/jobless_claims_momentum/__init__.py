"""Study 385 — Jobless-Claims-Momentum (does rising claims call downturns early?).

The macro-nowcasting folklore: initial unemployment claims are a leading indicator, so
when the 4-week-MA of claims turns **up**, an equity downturn is supposedly on the way —
an early-warning you can trade. We rebuild the believers' signal on the monthly claims
tape (a hardcoded, never-revised snapshot of FRED ``IC4WSA``, since FRED is firewalled
here) and measure forward 1/3/6/12-month SPY returns conditional on rising vs falling
claims momentum, against the unconditional base rate, with a Welch t, a placebo null, a
lead/lag scan, and a tradable timing overlay.

The decisive finding is about *timing and tradability*, not direction: claims and the
market are tangled around recessions, but a monthly claims uptick is too coincident,
noisy and infrequent — and the COVID-2020 spike too dominant — to be an early-warning
you can allocate to.

See :mod:`jobless_claims_momentum.data` (hardcoded claims snapshot + SPY loader +
deterministic synthetic control) and :mod:`jobless_claims_momentum.strategy`
(momentum signal, forward-return inference, placebo null, lead/lag, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
