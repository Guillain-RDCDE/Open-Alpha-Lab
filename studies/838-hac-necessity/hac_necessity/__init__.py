"""Study 838 — HAC Necessity (Newey-West necessity).

Newey & West (1987); Hansen & Hodrick (1980): on a serially-correlated return series — the
signature of any daily strategy with an overlapping formation window — the naive i.i.d. OLS
*t*-statistic **overstates significance**, sometimes wildly. The HAC (Newey-West) *t*
corrects it. We build a mean-zero null with a tunable amount of injected autocorrelation and
show the naive false-positive rate blows past 5% while the Newey-West *t* stays calibrated,
quantifying the *t*-inflation against its closed form (``sqrt(window)`` for a ``window``-day
overlap, ``sqrt((1+rho)/(1-rho))`` for AR(1)).

* ``data``     — deterministic seeded null-world generators (overlapping-window MA and AR(1)),
                 a ``mean`` knob for the positive control, and the closed-form inflation
                 factors. No network, no market data — a simulation study.
* ``strategy`` — the inference primitives (naive one-sample / Welch / Newey-West HAC /
                 Wilson / autocorrelation), their vectorised Monte-Carlo forms, and the
                 false-positive / inflation-curve / placebo / power experiments.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
