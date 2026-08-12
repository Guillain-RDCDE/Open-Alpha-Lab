"""Study 880 — Aggregate Short Interest.

Rapach, Ringgenberg & Zhou (2016): the **market-wide** short-interest index is
"arguably the strongest known predictor" of the aggregate equity return — a high,
detrended aggregate short-interest reading forecasts **lower** forward market
returns (a negative predictive slope). This is the aggregate / time-series cousin
of the cross-sectional short-interest sort in study 262.

* ``data``     — the real aggregate short-interest tape (FINRA Consolidated Short
                 Interest, the bi-monthly settlement-date days-to-cover report,
                 pulled per-name from the public FINRA Query API and averaged
                 equal-weight across a fixed liquid panel), SPY total-return for
                 the return side, and a deterministic seeded synthetic positive
                 control (a planted negative SI->return relation, null at
                 ``edge=0``).
* ``strategy`` — the detrended-log short-interest index (the RRZ step), the
                 forward-return predictive regression with a Newey-West slope t,
                 the inference primitives (Welch / one-sample / Newey-West HAC /
                 Wilson / permutation placebo), the era cut, and the costed
                 market-timing overlay.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
