"""Strategy tests for Study 987 — is silver levered gold?"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from loudcousin import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Beta estimation
# --------------------------------------------------------------------------- #
def test_full_sample_beta_recovers_a_planted_loading():
    df = st.synthetic_world(n=6000, true_beta=1.8)
    b = st.full_sample_beta(df["silver"], df["gold"])
    assert b["beta"] == pytest.approx(1.8, abs=0.05)
    assert b["t_vs_zero"] > 20


def test_beta_of_a_series_on_itself_is_one():
    df = st.synthetic_world(n=2000)
    assert st.full_sample_beta(df["gold"], df["gold"])["beta"] == pytest.approx(1.0, abs=1e-9)


def test_full_sample_beta_declines_on_too_little_data():
    df = st.synthetic_world(n=50)
    assert "beta" not in st.full_sample_beta(df["silver"], df["gold"])


def test_r2_falls_when_silver_gains_an_independent_driver():
    pure = st.synthetic_world(n=6000, industrial_load=0.0)
    mixed = st.synthetic_world(n=6000, industrial_load=1.5)
    assert (st.full_sample_beta(mixed["silver"], mixed["gold"])["r2"]
            < st.full_sample_beta(pure["silver"], pure["gold"])["r2"])


def test_rolling_beta_never_looks_forward():
    df = st.synthetic_world(n=4000)
    bad = df.copy()
    bad.iloc[3000:, bad.columns.get_loc("gold")] *= 6
    a = st.rolling_beta(df["silver"], df["gold"]).iloc[:2700].dropna()
    b = st.rolling_beta(bad["silver"], bad["gold"]).iloc[:2700].dropna()
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_a_constant_beta_world_gives_a_narrow_rolling_range():
    df = st.synthetic_world(n=6000, true_beta=1.8, beta_drift=0.0)
    t = st.rolling_beta_table(df["silver"], df["gold"], windows=(252,))
    assert t.loc[252, "range_over_mean"] < 0.6


def test_a_drifting_beta_world_gives_a_wide_one():
    steady = st.synthetic_world(n=6000, beta_drift=0.0)
    drifting = st.synthetic_world(n=6000, beta_drift=0.8)
    a = st.rolling_beta_table(steady["silver"], steady["gold"], windows=(252,))
    b = st.rolling_beta_table(drifting["silver"], drifting["gold"], windows=(252,))
    assert b.loc[252, "sd"] > a.loc[252, "sd"] * 1.5


def test_shorter_windows_give_noisier_betas():
    df = st.synthetic_world(n=6000)
    t = st.rolling_beta_table(df["silver"], df["gold"], windows=(63, 756))
    assert t.loc[63, "sd"] > t.loc[756, "sd"]


def test_beta_by_regime_finds_no_asymmetry_when_none_is_planted():
    df = st.synthetic_world(n=8000, true_beta=1.8)
    r = st.beta_by_regime(df["silver"], df["gold"])
    up = r.loc["gold up days", "beta"]
    down = r.loc["gold down days", "beta"]
    assert abs(up - down) < 0.25


def test_beta_by_regime_covers_up_down_and_volatility_buckets():
    df = st.synthetic_world(n=5000)
    r = st.beta_by_regime(df["silver"], df["gold"], n_buckets=3)
    assert len(r) == 5
    assert "gold up days" in r.index and "gold vol Q3" in r.index


# --------------------------------------------------------------------------- #
# The residual
# --------------------------------------------------------------------------- #
def test_the_residual_is_orthogonal_to_gold():
    df = st.synthetic_world(n=6000)
    r = st.residuals(df["silver"], df["gold"])
    assert abs(r.corr(df["gold"].reindex(r.index))) < 0.15


def test_the_residual_is_pure_noise_when_silver_is_only_levered_gold():
    """The null: if the folklore is literally true, nothing should load on the leftover."""
    df = st.synthetic_world(n=8000, industrial_load=0.0)
    r = st.residuals(df["silver"], df["gold"])
    load = st.residual_loadings(r, df[["industrial"]])
    assert abs(load.loc["industrial", "t"]) < 3.0


def test_the_residual_finds_a_planted_second_driver():
    df = st.synthetic_world(n=8000, industrial_load=1.5)
    r = st.residuals(df["silver"], df["gold"])
    load = st.residual_loadings(r, df[["industrial"]])
    assert load.loc["industrial", "t"] > 10
    assert load.loc["industrial", "beta"] == pytest.approx(1.5, abs=0.3)


def test_residual_loadings_reports_univariate_and_joint():
    df = st.synthetic_world(n=6000, industrial_load=1.0)
    r = st.residuals(df["silver"], df["gold"])
    load = st.residual_loadings(r, df[["industrial", "gold"]])
    assert len(load) == 4
    assert "industrial (joint)" in load.index


def test_residual_diagnostics_reports_the_shape_of_the_leftover():
    df = st.synthetic_world(n=6000)
    d = st.residual_diagnostics(st.residuals(df["silver"], df["gold"]))
    assert d["vol_ann"] > 0.05
    assert abs(d["autocorr_1"]) < 0.2
    assert set(("mean_ann", "sharpe", "skew", "kurtosis")) <= set(d)


def test_residual_diagnostics_declines_on_too_little_data():
    assert "vol_ann" not in st.residual_diagnostics(pd.Series([0.01] * 20))


# --------------------------------------------------------------------------- #
# The arithmetic of leverage
# --------------------------------------------------------------------------- #
def test_leverage_drag_is_zero_at_beta_one():
    assert st.leverage_drag(1.0, 0.16) == pytest.approx(0.0)


def test_leverage_drag_grows_with_beta_and_volatility():
    assert st.leverage_drag(2.0, 0.16) > st.leverage_drag(1.5, 0.16) > 0
    assert st.leverage_drag(2.0, 0.30) > st.leverage_drag(2.0, 0.16)


def test_leverage_drag_matches_its_closed_form():
    assert st.leverage_drag(2.0, 0.16) == pytest.approx(2 * 1 * 0.16 ** 2 / 2)


def test_deleveraging_below_one_is_a_gain_not_a_drag():
    """Beta below 1 gives a negative 'drag' — holding less than the asset is a free lunch here."""
    assert st.leverage_drag(0.5, 0.20) < 0


def test_a_levered_position_scales_and_pays_financing():
    df = st.synthetic_world(n=2000)
    free = st.levered_position(df["gold"], 2.0, df["cash"], financing_spread=0.0, cost_bps=0.0)
    paid = st.levered_position(df["gold"], 2.0, df["cash"], financing_spread=0.05, cost_bps=0.0)
    assert paid.mean() < free.mean()
    assert free.std() == pytest.approx(2 * df["gold"].std(), rel=0.01)


def test_no_financing_is_charged_below_full_investment():
    df = st.synthetic_world(n=1000)
    a = st.levered_position(df["gold"], 0.5, df["cash"], financing_spread=0.05, cost_bps=0.0)
    b = st.levered_position(df["gold"], 0.5, df["cash"], financing_spread=0.00, cost_bps=0.0)
    assert np.allclose(a.to_numpy(), b.to_numpy())


# --------------------------------------------------------------------------- #
# Replication
# --------------------------------------------------------------------------- #
def test_replication_is_near_perfect_when_silver_really_is_levered_gold():
    df = st.synthetic_world(n=8000, true_beta=1.8, industrial_load=0.0)
    r = st.replication_backtest(df["silver"], df["gold"], df["cash"],
                                financing_spread=0.0, cost_bps=0.0)
    assert r["correlation"] > 0.75
    assert r["beta_used"] == pytest.approx(1.8, abs=0.05)


def test_replication_degrades_when_silver_has_its_own_driver():
    pure = st.synthetic_world(n=8000, industrial_load=0.0)
    mixed = st.synthetic_world(n=8000, industrial_load=2.0)
    a = st.replication_backtest(pure["silver"], pure["gold"], pure["cash"])
    b = st.replication_backtest(mixed["silver"], mixed["gold"], mixed["cash"])
    assert b["tracking_error_ann"] > a["tracking_error_ann"]
    assert b["correlation"] < a["correlation"]


def test_replication_reports_the_predicted_drag():
    df = st.synthetic_world(n=5000, true_beta=2.0)
    r = st.replication_backtest(df["silver"], df["gold"], df["cash"])
    assert r["predicted_drag"] > 0


def test_an_explicit_beta_overrides_the_fitted_one():
    df = st.synthetic_world(n=3000, true_beta=1.8)
    assert st.replication_backtest(df["silver"], df["gold"], df["cash"],
                                   beta=3.0)["beta_used"] == 3.0


# --------------------------------------------------------------------------- #
# The ratio trade
# --------------------------------------------------------------------------- #
def test_the_ratio_starts_at_one_by_construction():
    idx = pd.bdate_range("2010-01-01", periods=500)
    g = pd.Series(np.linspace(100, 150, 500), index=idx)
    s = pd.Series(np.linspace(20, 40, 500), index=idx)
    r = st.gold_silver_ratio(g, s)
    assert r.iloc[0] == pytest.approx(1.0)
    assert r.iloc[-1] < 1.0        # silver outran gold


def test_ratio_mean_reversion_is_flat_on_a_random_walk():
    """With HAC errors a random walk must not look like a mean-reverting trade."""
    rng = np.random.default_rng(987)
    idx = pd.bdate_range("2006-05-01", periods=5000)
    r = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.012, 5000))), index=idx)
    out = st.ratio_mean_reversion(r)
    assert (out["t"].abs() < 3).all()


def test_overlapping_windows_need_hac_and_hc1_would_lie():
    """The trap, made explicit: the same random walk, the same regression, two error models."""
    rng = np.random.default_rng(2)
    n = 5000
    lr = np.cumsum(rng.normal(0, 0.012, n))
    ratio = pd.Series(np.exp(lr), index=pd.bdate_range("2006-05-01", periods=n))
    z = ((np.log(ratio) - np.log(ratio).rolling(252).mean())
         / np.log(ratio).rolling(252).std())
    fwd = np.log(ratio).shift(-252) - np.log(ratio)
    df = pd.concat([z.rename("z"), fwd.rename("f")], axis=1, sort=False).dropna()
    naive = abs(st.full_sample_beta(df["f"], df["z"], hac_lags=0)["t_vs_zero"])
    hac = abs(st.full_sample_beta(df["f"], df["z"], hac_lags=252)["t_vs_zero"])
    assert hac < naive / 2
    assert naive > 2          # past the conventional bar: HC1 would call this a finding
    assert hac < 2            # HAC correctly calls it nothing


def test_ratio_mean_reversion_finds_a_planted_reverting_ratio():
    rng = np.random.default_rng(987)
    n = 6000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.995 * x[t - 1] + rng.normal(0, 0.01)
    r = pd.Series(np.exp(x), index=pd.bdate_range("2006-05-01", periods=n))
    out = st.ratio_mean_reversion(r, lookback=252, horizons=(126,))
    assert out.loc[126, "slope"] < 0        # stretched high -> falls back


def test_ratio_mean_reversion_is_empty_on_a_short_series():
    r = pd.Series(np.ones(100), index=pd.bdate_range("2020-01-01", periods=100))
    assert len(st.ratio_mean_reversion(r)) == 0


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_days": 5000, "beta": 1.72, "beta_se": 0.03, "r2": 0.62, "resid_vol": 0.19,
         "max_abs_residual_t": 4.1, "strongest_factor": "XLI", "beta_min": 0.95,
         "beta_max": 2.55, "beta_range_over_mean": 0.93, "beta_up": 1.68, "beta_down": 1.77,
         "financing_spread": 0.005, "correlation": 0.79, "tracking_error": 0.20,
         "years": 20.0, "silver_cagr": 0.041, "replica_cagr": 0.019,
         "silver_sharpe": 0.13, "replica_sharpe": 0.00, "gold_vol": 0.16,
         "predicted_drag": 0.016}
    h.update(over)
    return h


def test_verdict_signal_needs_a_clean_residual_and_a_stable_beta():
    assert st.verdict(_headline())["signal"] == "Busted"
    assert st.verdict(_headline(max_abs_residual_t=1.2))["signal"] == "Partial"
    assert st.verdict(_headline(beta_range_over_mean=0.3))["signal"] == "Partial"
    assert st.verdict(_headline(max_abs_residual_t=1.2,
                                beta_range_over_mean=0.3))["signal"] == "Confirmed"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Mirage"
    assert st.verdict(_headline(replica_sharpe=0.10))["trad"] == "Partial"
    assert st.verdict(_headline(replica_sharpe=0.30))["trad"] == "Useful"


def test_verdict_prose_quotes_the_drag_and_the_beta_range():
    v = st.verdict(_headline())
    assert "volatility drag" in v["trad_why"]
    assert "beta" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
