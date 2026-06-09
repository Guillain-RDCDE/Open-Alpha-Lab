"""The decomposition recovers the baked-in truth: OC-red lifts the close rate, IB-rejection
adds nothing on top of it, the headline is mostly a mechanical head-start, and selection over
many small samples inflates a modest edge into a headline."""

import math

import numpy as np
import pandas as pd

from crimson_hour import decompose, signals


def test_wilson_is_wide_at_small_n():
    """The first finding, as a unit test: 22/25 carries a CI floor well under the point."""
    lo, hi = decompose.wilson_ci(22, 25)
    assert 0.86 < 22 / 25 < 0.90
    assert lo < 0.72            # the honest lower bound is far below the quoted 88%
    assert hi < 1.0


def test_wilson_matches_known_value():
    lo, hi = decompose.wilson_ci(50, 100)
    assert math.isclose(lo, 0.4038, abs_tol=1e-3)
    assert math.isclose(hi, 0.5962, abs_tol=1e-3)


def test_beta_binomial_posterior_below_point():
    post = decompose.beta_binomial(22, 25, thresholds=(0.7,))
    assert post["posterior_mean"] < 22 / 25            # shrinks toward 0.5
    assert post["cred_low"] < 0.72
    assert 0.0 < post["P(rate>0.7)"] < 1.0


def test_oc_red_lifts_the_close_rate(feat):
    """Momentum is real and baked in: OC-red days close red more often than baseline."""
    m = signals.condition_masks(feat)
    tab = decompose.conditional_table(m, signals.session_red(feat))
    assert tab.loc["oc_red", "rate"] > tab.loc["baseline", "rate"] + 0.10
    assert tab.loc["oc_red", "lift_pp"] > 10


def test_ib_rejection_adds_nothing(feat, truth):
    """The baked-in null: IB-rejection carries no info beyond OC-red, so the increment ~ 0."""
    assert truth.ib_is_null
    inc = decompose.ib_increment(feat)
    assert abs(inc["increment_pp"]) < 8.0          # economically tiny
    assert inc["fisher_p_value"] > 0.05            # and statistically indistinguishable


def test_headline_is_mostly_mechanical(feat):
    """P(session red | OC-red) is far higher than the genuine continuation forecast."""
    split = decompose.mechanical_vs_predictive(feat)
    assert split["headline_rate"] > split["continuation_rate"] + 0.15
    assert split["mechanical_share"] > 0.5          # most of the lift is the head-start
    # the genuine continuation lift is the small, real momentum edge
    assert 0 < split["continuation_lift_pp"] < split["headline_lift_pp"]


def test_continuation_lift_tracks_momentum():
    """More baked-in momentum -> bigger genuine continuation lift. We're measuring the real thing."""
    from crimson_hour import data
    lo_feat, _ = data.synthetic_sessions(momentum=0.02, seed=5)
    hi_feat, _ = data.synthetic_sessions(momentum=0.30, seed=5)
    lo = decompose.mechanical_vs_predictive(lo_feat)["continuation_lift_pp"]
    hi = decompose.mechanical_vs_predictive(hi_feat)["continuation_lift_pp"]
    assert hi > lo


def test_continuation_is_detectable_and_tracks_momentum():
    """The genuine forecast leg is real on the synthetic and scales with the baked-in momentum."""
    from crimson_hour import data
    ct = decompose.continuation_test(data.synthetic_sessions(momentum=0.20, seed=7)[0])
    # red mornings -> more red afternoons and a more negative mean afternoon than green mornings
    assert ct["sign_diff_pp"] > 0
    assert ct["mean_rest_oc_red_bps"] < ct["mean_rest_oc_green_bps"]
    lo = decompose.continuation_test(data.synthetic_sessions(momentum=0.02, seed=7)[0])["mean_diff_bps"]
    hi = decompose.continuation_test(data.synthetic_sessions(momentum=0.40, seed=7)[0])["mean_diff_bps"]
    assert abs(hi) > abs(lo)            # stronger momentum -> larger afternoon contrast


def test_afternoon_short_backtest_breakeven_and_cost_monotone(feat):
    """The tradable leg: a break-even cost exists, and net Sharpe falls as costs rise."""
    bt = decompose.afternoon_short_backtest(feat)
    assert bt["n_trades"] > 50
    assert bt["break_even_cost_bps"] == bt["gross_mean_bps"]
    nets = [bt["net"][c]["net_sharpe"] for c in (0.0, 0.5, 1.0, 2.0, 5.0)]
    assert nets == sorted(nets, reverse=True)          # monotonically decreasing in cost
    # net mean at the break-even cost is ~0 by construction
    be = bt["break_even_cost_bps"]
    assert abs(bt["gross_mean_bps"] - be) < 1e-6


def test_welch_t_zero_for_identical():
    import numpy as np
    a = np.array([1.0, 2.0, 3.0, 4.0])
    out = decompose.welch_t(a, a)
    assert abs(out["t"]) < 1e-9 and out["p_value"] > 0.99


def test_mining_inflates_a_modest_edge():
    """A true 62% edge, mined across a dozen tiny confluences, routinely *looks* far bigger."""
    res = decompose.mining_inflation(p_true=0.62, n_cond=25, n_candidates=12,
                                     observed=0.88, seed=0)
    assert res["expected_best_rate"] > 0.62 + 0.10     # selection inflates well above the truth
    assert res["best_rate_p95"] > 0.80
    assert res["P(best>=observed)"] > 0.02             # 88% is not a freak under mining


def test_rate_drops_na(feat):
    """NA IB flags are excluded from both numerator and denominator, never counted as False."""
    full = decompose.rate(signals.ib_high_rejected(feat), signals.session_red(feat))
    f = feat.copy()
    f.loc[f.index[:10], "ib_high_rejected"] = pd.NA
    r = decompose.rate(signals.ib_high_rejected(f), signals.session_red(f))
    n_true_blanked = int(feat["ib_high_rejected"].iloc[:10].astype(bool).sum())
    assert r["n"] == full["n"] - n_true_blanked          # exactly the rejected-True NA rows drop
