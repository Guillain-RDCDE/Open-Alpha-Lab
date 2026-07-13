"""Study 756 — Challenger-Layoffs (does a job-cut spike precede equity weakness?).

The macro-nowcasting folklore: the Challenger, Gray & Christmas monthly Job Cuts Report —
the widely-cited tally of layoffs *announced* by U.S. employers — is a leading indicator,
so when announced job cuts **spike**, an equity (and labour-market) downturn is supposedly
on the way. We rebuild the believers' signal on the monthly Challenger tape (a hardcoded,
clearly-labelled *approximate* snapshot, since Challenger's series is proprietary with no
free feed) and measure forward 1/3/6/12-month SPY returns conditional on a job-cut spike
vs a calm month, against the unconditional base rate, with a Welch t, a Newey-West HAC t,
a placebo null, a lead/lag scan, and a tradable timing overlay.

The decisive finding is about *timing and tradability*, not direction: job cuts and the
market are tangled around recessions, but an announced-cut spike is too coincident, noisy
and infrequent — and the COVID-2020 spike too dominant — to be an early-warning you can
allocate to.

See :mod:`challenger_layoffs.data` (hardcoded Challenger snapshot + SPY loader +
deterministic synthetic control) and :mod:`challenger_layoffs.strategy`
(spike signal, forward-return inference, HAC t, placebo null, lead/lag, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
