"""Strategy tests for Study 967 — window mechanics, out-of-sample discipline, known cases."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from window_choice import data, strategy as st  # noqa: E402


def _panel(planted):
    prices, cash, _ = planted
    r = st.to_returns(prices)
    r["MKT"] = r.mean(axis=1)
    return r


# --------------------------------------------------------------------------- #
# Mechanics
# --------------------------------------------------------------------------- #
def test_beta_of_recovers_a_planted_slope():
    rng = np.random.default_rng(967)
    idx = pd.bdate_range("2010-01-01", periods=2000)
    m = pd.Series(rng.normal(0, 0.01, 2000), index=idx)
    a = 1.4 * m + pd.Series(rng.normal(0, 0.004, 2000), index=idx)
    assert st.beta_of(a, m) == pytest.approx(1.4, abs=0.05)


def test_beta_of_is_nan_on_a_stub():
    s = pd.Series([0.01, -0.01, 0.02])
    assert np.isnan(st.beta_of(s, s))


def test_train_slice_respects_the_window_and_never_peeks(planted):
    r = _panel(planted)
    asof = r.index[1500]
    roll = st.train_slice(r, asof, 2)
    exp = st.train_slice(r, asof, st.EXPANDING)
    assert roll.index[-1] <= asof and exp.index[-1] <= asof
    assert len(roll) == 2 * st.TRADING_DAYS
    assert len(exp) == 1501


def test_year_ends_are_the_last_session_of_each_year(planted):
    r = _panel(planted)
    ends = st.year_ends(r.index)
    years = [d.year for d in ends]
    assert len(years) == len(set(years))
    for d in ends[:-1]:
        after = r.loc[d:].index
        assert after[1].year == d.year + 1


def test_min_variance_weights_sum_to_one_and_beat_equal_weight_in_sample():
    rng = np.random.default_rng(967)
    x = rng.normal(0, 1, (2000, 4)) @ np.diag([0.01, 0.02, 0.03, 0.04])
    cov = np.cov(x.T, ddof=1)
    w = st.min_variance_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    eq = np.full(4, 0.25)
    assert w @ cov @ w <= eq @ cov @ eq + 1e-15


def test_min_variance_long_only_has_no_shorts():
    cov = np.array([[4e-4, 3.5e-4], [3.5e-4, 4.5e-4]])
    w = st.min_variance_weights(cov, long_only=True)
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0)


def test_blume_shrinkage_pulls_toward_one():
    assert st.blume_shrunk(2.0) == pytest.approx(2 / 3 * 2 + 1 / 3)
    assert st.blume_shrunk(1.0) == pytest.approx(1.0)
    assert 1.0 < st.blume_shrunk(1.5) < 1.5


# --------------------------------------------------------------------------- #
# The experiments
# --------------------------------------------------------------------------- #
def test_beta_experiment_is_out_of_sample(planted):
    """Mangling a year's returns must not change any estimate made before that year."""
    r = _panel(planted)
    sectors = tuple(c for c in r.columns if c != "MKT")[:4]
    cut = r.index[-400]
    mangled = r.copy()
    mangled.loc[cut:] *= 5
    a = st.beta_experiment(r, sectors, "MKT", windows=(2,))
    b = st.beta_experiment(mangled, sectors, "MKT", windows=(2,))
    a = a[a["date"] < cut - pd.Timedelta(days=370)]
    b = b[b["date"] < cut - pd.Timedelta(days=370)]
    assert np.allclose(a["estimate"].to_numpy(), b["estimate"].to_numpy())


def test_expanding_window_wins_when_parameters_are_stable(planted):
    """With a stationary generator, more data must beat less for the mean."""
    r = _panel(planted)
    sectors = tuple(c for c in r.columns if c != "MKT")
    exp = st.mean_experiment(r, sectors, windows=(1, 3))
    sc = st.score(exp)
    assert sc.loc[st.EXPANDING, "mse"] <= sc.loc[1, "mse"]


def test_score_orders_windows_and_counts_observations(planted):
    r = _panel(planted)
    sectors = tuple(c for c in r.columns if c != "MKT")
    sc = st.score(st.beta_experiment(r, sectors, "MKT", windows=(1, 2)))
    assert list(sc.index) == [1, 2, st.EXPANDING]
    assert (sc["n"] > 0).all() and (sc["mse"] > 0).all()


def test_covariance_experiment_reports_a_realised_and_a_predicted_vol(planted):
    r = _panel(planted)
    sectors = tuple(c for c in r.columns if c != "MKT")
    ex = st.covariance_experiment(r, sectors, windows=(1, 3))
    assert {"realised_vol", "predicted_vol", "turnover", "n_assets"} <= set(ex.columns)
    assert (ex["realised_vol"] > 0).all()
    assert (ex["predicted_vol"] > 0).all()
    assert set(ex["window"]) == {1, 3, st.EXPANDING}


def test_optimism_appears_when_parameters_outnumber_observations():
    """The optimiser's in-sample promise beats its out-of-sample delivery — but only when
    the matrix is thinly estimated. On six assets and a year of data (42 rows per parameter)
    there is nothing to see; on twenty-four assets (under 4 rows per parameter) there is.
    Recording both is the point: 'optimisers are optimistic' is a statement about
    rows-per-parameter, not a law of nature."""
    wide, _, _ = data.synthetic_panel(n_assets=24, n_years=15, seed=967)
    narrow, _, _ = data.synthetic_panel(n_assets=6, n_years=15, seed=967)
    out = {}
    for name, panel in (("wide", wide), ("narrow", narrow)):
        r = st.to_returns(panel)
        ex = st.covariance_experiment(r, tuple(panel.columns), windows=(1,))
        ex = ex[ex["window"] == 1]
        out[name] = float((ex["predicted_vol"] / ex["realised_vol"] - 1).mean())
    assert out["wide"] < -0.05          # promises materially less risk than it delivers
    assert out["wide"] < out["narrow"]  # and the effect grows with the parameter count


def test_short_windows_produce_more_extreme_minimum_variance_weights(planted):
    r = _panel(planted)
    sectors = tuple(c for c in r.columns if c != "MKT")
    ex = st.covariance_experiment(r, sectors, windows=(1, 5))
    short = ex[ex["window"] == 1]["max_weight"].mean()
    long = ex[ex["window"] == 5]["max_weight"].mean()
    assert short > long


def test_grand_mean_benchmark_gives_every_sector_the_same_number(planted):
    r = _panel(planted)
    sectors = tuple(c for c in r.columns if c != "MKT")
    gm = st.grand_mean_benchmark(r, sectors)
    per_date = gm.groupby("date")["estimate"].nunique()
    assert (per_date == 1).all()


def test_pairwise_dm_is_symmetric_in_sign(planted):
    r = _panel(planted)
    sectors = tuple(c for c in r.columns if c != "MKT")
    ex = st.beta_experiment(r, sectors, "MKT", windows=(1, 5))
    ab = st.pairwise_dm(ex, 1, 5)["dm"]
    ba = st.pairwise_dm(ex, 5, 1)["dm"]
    assert ab == pytest.approx(-ba, abs=1e-9)


def test_diebold_mariano_null_behaviour():
    rng = np.random.default_rng(967)
    a = pd.Series(rng.normal(1, 0.3, 500))
    b = pd.Series(rng.normal(1, 0.3, 500))
    assert abs(st.diebold_mariano(a, b)["dm"]) < 3.0
    assert st.diebold_mariano(a + 0.5, b)["dm"] > 2.0


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"spread_beta": 0.4, "spread_mean": 0.3, "spread_cov": 0.2, "max_abs_dm": 3.0,
         "best_beta": 5, "best_mean": "expanding", "best_cov": 10,
         "grand_mean_ratio": 0.8, "blume_gain": 0.1, "n_sectors": 11,
         "obs_per_param_1y": 3.8}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(max_abs_dm=1.0))["signal"] == "Weak"
    assert st.verdict(_headline(spread_beta=0.05, spread_mean=0.05, spread_cov=0.05,
                                max_abs_dm=1.0))["signal"] == "None"


def test_verdict_usefulness_needs_one_winner_everywhere():
    assert st.verdict(_headline())["trad"] == "Fragile"
    same = _headline(best_beta="expanding", best_cov="expanding")
    assert st.verdict(same)["trad"] == "Useful"
    assert st.verdict(_headline(max_abs_dm=0.5))["trad"] == "Mirage"


def test_verdict_prose_quotes_the_numbers():
    v = st.verdict(_headline(spread_beta=0.37))
    assert "37%" in v["signal_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
