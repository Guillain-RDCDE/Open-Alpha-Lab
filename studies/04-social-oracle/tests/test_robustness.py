"""Clustering bootstrap, fade curve, name jackknife, deflated Sharpe, split."""

import numpy as np

from social_oracle import robustness


def test_block_bootstrap_returns_ci(panel, events):
    out = robustness.block_bootstrap_excess(panel, events, horizon=5, n_iter=300)
    for k in ("mean", "ci_low", "ci_high", "p_excess_le_0"):
        assert k in out
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]
    assert 0.0 <= out["p_excess_le_0"] <= 1.0


def test_fade_curve_has_expected_horizons(panel, events):
    fc = robustness.fade_curve(panel, events, horizons=(1, 5, 21))
    assert list(fc.index) == [1, 5, 21]
    assert {"mean_car", "pct_positive", "tstat", "n"}.issubset(fc.columns)


def test_name_jackknife_first_row_is_full_set(panel, events):
    jk = robustness.name_jackknife(panel, events, horizon=5, top=2)
    assert jk.index[0] == "(none)"
    assert jk.loc["(none)", "n_events"] == len(events)
    # dropping a name never increases the event count
    assert (jk["n_events"].iloc[1:] <= len(events)).all()


def test_deflated_sharpe_in_unit_interval():
    p = robustness.deflated_sharpe(best_sharpe=2.0, n_trials=50, n_obs=500)
    assert 0.0 <= p <= 1.0
    # more trials -> a given Sharpe is less impressive
    p_more = robustness.deflated_sharpe(best_sharpe=2.0, n_trials=500, n_obs=500)
    assert p_more <= p


def test_split_sample_is_chronological(events):
    a, b = robustness.split_sample(events, frac=0.6)
    assert len(a) + len(b) == len(events)
    if len(a) and len(b):
        assert a["entry_date"].max() <= b["entry_date"].min()
