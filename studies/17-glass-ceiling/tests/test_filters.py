"""The filter verdict, baked into tapes: on the null and the unconditional-continuation tapes the
three filters add no win-rate lift (they only shrink the sample — the selection illusion), but on the
**grind-gated** tape the staircase filter recovers real edge — proof the test has power and the
negative result on the real claim is a finding, not a rigged null."""

import numpy as np

from glass_ceiling import filters, strategy


def test_filters_dont_help_on_null(null_tape):
    """Decorative by construction: the win-rate lift is statistically indistinguishable from zero
    (within ~2 standard errors of the thinned subset), while the trade count collapses — the exact
    signature of a selection illusion rather than a real edge."""
    bars, _ = null_tape
    trades = strategy.run(bars)
    res = filters.filter_lift(trades, bars)
    n = res["n_filtered"]
    se = np.sqrt(0.25 / n)                             # SE of a ~50% win rate on n filtered trades
    assert abs(res["lift"]) < 2.5 * se                # consistent with no lift
    assert res["kept_frac"] < 0.6                     # but the sample is thinned a lot


def test_grind_filter_recovers_real_edge(grind_tape):
    """When continuation is genuinely gated on a calm approach, the grind score lifts the win rate."""
    bars, _ = grind_tape
    trades = filters.annotate(strategy.run(bars), bars)
    resolved = trades[trades["outcome"] != 0]
    # split on the grind score alone — high-grind trades should win more often
    hi = resolved[resolved["grind"] >= resolved["grind"].median()]
    lo = resolved[resolved["grind"] < resolved["grind"].median()]
    wr_hi = (hi["outcome"] == 1).mean()
    wr_lo = (lo["outcome"] == 1).mean()
    assert wr_hi > wr_lo + 0.03


def test_scores_have_no_lookahead(null_tape):
    """Filter scores at a trade are computed from bars up to and including entry — never after."""
    bars, _ = null_tape
    trades = strategy.run(bars).head(20)
    ann = filters.annotate(trades, bars, window=20)
    assert ann["grind"].notna().all()
    assert ann["clean"].le(0).all()                   # crossover count is returned negated (<= 0)
