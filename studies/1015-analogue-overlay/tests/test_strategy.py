"""Strategy tests for Study 1015 — the analogue forecaster, its inference, its null, its verdict.

Synthetic-first: the detector must fire on a tape where a template genuinely recurs and stay
quiet on the matched random walk. Real-tape checks are gated on the bundled tapes.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from overlay import data, strategy as st  # noqa: E402

OOS = "1950-01-31"


# --------------------------------------------------------------------------- #
# The machinery
# --------------------------------------------------------------------------- #
def test_window_dot_product_is_pearson_correlation():
    rng = np.random.default_rng(0)
    v = np.cumsum(rng.normal(0, 1, 200))
    W, valid = st._normalised_windows(v, 24)
    assert valid.all()
    c = W[10] @ W[150]
    assert c == pytest.approx(np.corrcoef(v[10:34], v[150:174])[0, 1], abs=1e-12)


def test_search_finds_a_planted_exact_copy_first():
    rng = np.random.default_rng(1)
    lr = rng.normal(0.005, 0.05, 600)
    lr[100:124] = lr[476:500]            # the window ending at 499 has an exact past twin
    logp = np.cumsum(lr)
    s = st.analogue_search(logp, 24, [499], k_max=3, metric="zret")
    assert s["ends"][0, 0] == 123
    assert s["corr"][0, 0] == pytest.approx(1.0, abs=1e-9)


def test_search_respects_the_past_and_never_overlaps():
    rng = np.random.default_rng(2)
    logp = np.cumsum(rng.normal(0, 1, 800))
    idx = np.arange(400, 800, 7)
    for gap in (24, 36):
        s = st.analogue_search(logp, 24, idx, k_max=10, gap=gap)
        E = s["ends"]
        for row, t in zip(E, idx):
            f = row[row >= 0]
            assert (f <= t - gap).all()                 # strictly in the past, outcome known
            d = np.abs(f[:, None] - f[None, :]) + np.eye(f.size, dtype=int) * 10 ** 6
            assert (d >= 24).all()                      # analogues never overlap each other
        assert np.all(np.diff(s["corr"], axis=1)[np.isfinite(np.diff(s["corr"], axis=1))]
                      <= 1e-12)                         # greedy order: best first


def test_forecast_has_no_look_ahead():
    df, _ = data.synthetic_tape(n_periods=500, seed=4)
    logp = df["logp"].to_numpy()
    t = 400
    s1 = st.analogue_search(logp, 24, [t], k_max=5, gap=24)
    f1 = st.analogue_forecast(logp, s1, 12, 5)
    scrambled = logp.copy()
    scrambled[t + 1:] = scrambled[t + 1:] + np.random.default_rng(9).normal(0, 5, 500 - t - 1)
    s2 = st.analogue_search(scrambled, 24, [t], k_max=5, gap=24)
    f2 = st.analogue_forecast(scrambled, s2, 12, 5)
    assert np.allclose(f1, f2)
    assert np.array_equal(s1["ends"], s2["ends"])


def test_forecast_refuses_unobserved_follow_ons():
    logp = np.cumsum(np.ones(200))
    s = st.analogue_search(logp + np.sin(np.arange(200)), 24, [150], k_max=2, gap=24)
    with pytest.raises(ValueError):
        st.analogue_forecast(logp, s, 36, 2)


def test_realised_and_drift_benchmark():
    logp = np.arange(100, dtype=float) * 0.01
    r = st.realised_forward(logp, [10, 95], 6)
    assert r[0] == pytest.approx(0.06) and np.isnan(r[1])
    b = st.historical_drift_forecast(logp, [50], 12)
    assert b[0] == pytest.approx(0.12)


def test_ols_hac_recovers_slope_and_widens_under_overlap():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 3000)
    y = 0.5 * x + rng.normal(0, 1, 3000)
    r = st.ols_hac(y, x, 0)
    assert r["b"] == pytest.approx(0.5, abs=0.05)
    e = np.convolve(rng.normal(0, 1, 3011), np.ones(12), mode="valid")  # 12-period overlap
    z = np.convolve(rng.normal(0, 1, 3011), np.ones(12), mode="valid")
    assert st.ols_hac(e, z, 12)["se_b"] > 1.5 * st.ols_hac(e, z, 0)["se_b"]


def test_hac_mean_t_and_holm():
    assert abs(st.hac_mean_t(np.random.default_rng(0).normal(0, 1, 2000), 5)) < 3
    adj = st.holm([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])


def test_evaluate_forecast_perfect_and_drift_aware_hits():
    rng = np.random.default_rng(5)
    y = rng.normal(0.01, 0.05, 500)
    bench = np.full(500, 0.01)
    ev = st.evaluate_forecast(y, y, bench, lags=1)
    assert ev["corr"] == pytest.approx(1.0) and ev["hit"] == pytest.approx(1.0)
    assert ev["oos_r2"] == pytest.approx(1.0)
    # An always-positive "forecast" scores a raw sign hit rate equal to the up-share,
    # and nothing against the drift: exactly the confusion the drift-aware hit removes.
    ev2 = st.evaluate_forecast(np.full(500, 0.02) + rng.normal(0, 1e-4, 500), y, bench, 1)
    assert ev2["hit_raw"] == pytest.approx(ev2["up_share"])
    assert abs(ev2["hit"] - 0.5) < 0.1


def test_timing_rule_lag_costs_and_always_long():
    df, _ = data.synthetic_tape(n_periods=400, signal_strength=0.0, seed=6)
    idx = np.arange(100, 400)
    long_ = st.timing_rule(df, idx, np.ones(idx.size), 12, cost_bps=10)
    assert long_["time_in_market"] == 1.0 and long_["turnover_per_year"] == 0.0
    assert long_["sharpe_net"] == pytest.approx(long_["sharpe_bh"])
    assert np.allclose(long_["diff"], 0.0)
    assert np.allclose(long_["diff_volmatched"], 0.0)
    flip = np.where(np.arange(idx.size) % 2 == 0, 1.0, -1.0)
    g = st.timing_rule(df, idx, flip, 12, cost_bps=0)
    n = st.timing_rule(df, idx, flip, 12, cost_bps=50)
    assert n["sharpe_net"] < g["sharpe_net"]
    # one-period execution lag: the signal at t first earns period t + 2
    fc = np.full(idx.size, -1.0)
    fc[50] = 1.0
    r = st.timing_rule(df, idx, fc, 12, lag=1, cost_bps=0)
    eq = r["equity"]
    rf = df["rf"].iloc[idx[0] + 2:]
    ret = np.log(eq).diff()
    ret.iloc[0] = np.log(eq.iloc[0])
    invested = ~np.isclose(ret.to_numpy(), np.log1p(rf.to_numpy()))
    assert invested.sum() == 1
    assert ret.index[invested][0] == df.index[idx[50] + 2]


# --------------------------------------------------------------------------- #
# The detector: fires on the plant, quiet on the null
# --------------------------------------------------------------------------- #
def _headline_t(tape):
    logp = tape["logp"].to_numpy()
    idx = st.eval_positions(tape.index, OOS)
    H = st.HEADLINE_MONTHLY
    s = st.analogue_search(logp, H["L"], idx, k_max=H["k"], gap=max(H["L"], H["h"]))
    fc = st.analogue_forecast(logp, s, H["h"], H["k"])
    return st.evaluate_forecast(fc, st.realised_forward(logp, idx, H["h"]),
                                st.historical_drift_forecast(logp, idx, H["h"]), H["h"])


def test_detector_fires_on_the_planted_template(planted):
    ev = _headline_t(planted[0])
    assert ev["t"] > 2.0 and ev["corr"] > 0.2 and ev["hit"] > 0.55


def test_detector_quiet_on_the_null(null_tape):
    ev = _headline_t(null_tape[0])
    assert ev["t"] < 2.0 and abs(ev["corr"]) < 0.15


def test_null_size_and_planted_power_across_seeds():
    H = st.HEADLINE_MONTHLY
    seeds = range(1015, 1023)
    null = st.null_sweep_power(lambda s: data.synthetic_tape(n_periods=800, seed=s,
                                                             signal_strength=0.0)[0],
                               None, 12, OOS, seeds, H)
    plant = st.null_sweep_power(lambda s: data.synthetic_tape(n_periods=800, seed=s,
                                                              signal_strength=1.0)[0],
                                None, 12, OOS, seeds, H)
    assert (null["t"] >= 2).mean() <= 0.25
    assert (plant["t"] >= 2).mean() >= 0.75
    # And the null STILL prints spectacular best matches — the teaching point.
    assert null["match_corr_median"].median() > 0.8


def test_sweep_reports_every_combination(planted):
    grid = {"L": (12, 24), "h": (1, 6), "k": (1, 5)}
    out = st.sweep(planted[0], grid, 12, OOS, keep=(("logprice", 24, 6, 5),))
    assert len(out["table"]) == 2 * 2 * 2 * 2
    assert out["diffs"].shape[1] == 16 and out["diffs_vm"].shape == out["diffs"].shape
    assert ("logprice", 24, 6, 5) in out["kept"]
    rc = st.reality_check(out["diffs"], 12, n_boot=200)
    assert 0.0 <= rc["reality_check_pvalue"] <= 1.0


# --------------------------------------------------------------------------- #
# The null: why 0.9 is cheap
# --------------------------------------------------------------------------- #
def test_independent_price_paths_correlate_returns_do_not():
    p = st.pair_correlation_null(24, 0.008, 0.053, n_sims=5000)
    assert p["p_price_gt_07"] > 0.10            # trending paths: a 0.7 is routine
    assert p["p_returns_gt_05"] < 0.02          # the returns that made them: almost never
    assert np.std(p["price"]) > 2 * np.std(p["returns"])


def test_searching_a_random_walk_finds_a_great_match():
    b = st.best_match_null(1000, 24, 0.008, 0.053, n_sims=60)
    assert np.median(b) > 0.85
    z = st.best_match_null(1000, 24, 0.008, 0.053, metric="zret", n_sims=60)
    assert np.median(z) < np.median(b) - 0.15


def test_richer_null_generators_shape_and_determinism():
    lr = np.random.default_rng(0).normal(0.01, 0.05, 300)
    a = st.vol_path_walks(lr, 3, seed=1)
    b = st.block_bootstrap_walks(lr, 3, mean_block=6, seed=1)
    assert a.shape == b.shape == (3, 300)
    assert np.allclose(a, st.vol_path_walks(lr, 3, seed=1))
    # block bootstrap reuses the real returns, so the increments are drawn from them
    inc = np.diff(b[0], prepend=0.0)
    assert np.isin(np.round(inc, 12), np.round(lr, 12)).all()


def test_template_overlay_and_episodes():
    rng = np.random.default_rng(8)
    lr = rng.normal(0.008, 0.05, 600)
    lr[400:424] = lr[100:124]
    logp = pd.Series(np.cumsum(lr), index=pd.date_range("1926-07-31", periods=600,
                                                         freq="ME"))
    end = logp.index[123]
    ov = st.template_overlay(logp, str(end.date()), 24, 12)
    assert ov.index[0] > logp.index[123 + 23]                # template's own window excluded
    assert ov.loc[logp.index[423], "corr"] == pytest.approx(1.0, abs=1e-6)
    e = st.overlay_episodes(ov, 0.9)
    assert e["n_match"] >= 1 and 0 <= e["crash_rate_all"] <= 1
    tn = st.template_null_share(logp.to_numpy()[100:124], 0.008, 0.05, n_sims=2000)
    assert 0 <= tn[0.9] <= tn[0.8] <= 1


# --------------------------------------------------------------------------- #
# The verdict, both directions
# --------------------------------------------------------------------------- #
def _h(**kw):
    h = {"m_head_t": 0.5, "d_head_t": -0.3, "holm_min_p": 0.9, "share_raw_sig": 0.04,
         "rc_p_monthly": 0.9, "rc_p_daily": 0.9, "m_head_alpha_t": 0.1, "d_head_alpha_t": -0.5,
         "m_head_sharpe_net": 0.40, "m_bh_sharpe": 0.50, "d_head_sharpe_net": 0.20,
         "d_bh_sharpe": 0.30, "m_head_corr": 0.05, "m_head_r2": -0.1, "n_combos": 126,
         "crash_rate_match": 0.05, "crash_rate_all": 0.04,
         "best_match_null": {"monthly logprice L24": {"share_gt_09": 0.5}}}
    h.update(kw)
    return h


def test_verdict_none_and_mirage_on_a_null_result():
    v = st.verdict(_h())
    assert v["signal"] == "None" and v["trad"] == "Mirage"
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
    assert "126 combinations" in v["signal_why"]


def test_verdict_real_needs_both_tapes_and_the_correction():
    assert st.verdict(_h(m_head_t=2.5, d_head_t=2.2, holm_min_p=0.01))["signal"] == "Real"
    assert st.verdict(_h(m_head_t=2.5, d_head_t=2.2, holm_min_p=0.20))["signal"] == "Weak"
    assert st.verdict(_h(m_head_t=2.5, d_head_t=1.0, holm_min_p=0.01))["signal"] == "Weak"


def test_verdict_mixed_weak_paths():
    assert st.verdict(_h(m_head_t=3.0, d_head_t=-1.0, holm_min_p=0.01))["signal"] == "Mixed"
    assert st.verdict(_h(share_raw_sig=0.2))["signal"] == "Weak"
    assert st.verdict(_h(holm_min_p=0.03))["signal"] == "Weak"


def test_verdict_tradability_ladder():
    real = dict(m_head_t=2.5, d_head_t=2.2, holm_min_p=0.01)
    inv = st.verdict(_h(**real, rc_p_monthly=0.01, rc_p_daily=0.02, m_head_sharpe_net=0.7,
                        d_head_sharpe_net=0.5))
    assert inv["trad"] == "Investable"
    # beating B&H on one tape only is not enough for Investable
    assert st.verdict(_h(**real, rc_p_monthly=0.01, rc_p_daily=0.02, m_head_sharpe_net=0.7)
                      )["trad"] == "Fragile"
    assert st.verdict(_h(m_head_alpha_t=2.3))["trad"] == "Fragile"
    assert st.verdict(_h(rc_p_daily=0.02))["trad"] == "Fragile"
    assert st.verdict(_h(rc_p_daily=0.04))["trad"] == "Mirage"


def test_myth_check_both_directions():
    base = {"real_median": 0.94, "real_share_gt_09": 0.7, "null_median": 0.90,
            "null_median_lo": 0.88, "null_median_hi": 0.92}
    assert st.myth_check({"like_for_like": {"monthly logprice L24": base}})["stamp"] \
        == "Confirmed"
    inside = dict(base, real_median=0.91)
    assert st.myth_check({"like_for_like": {"monthly logprice L24": inside}})["stamp"] \
        == "Busted"


# --------------------------------------------------------------------------- #
# Real tapes (gated)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_monthly(), reason="arch Fama-French tape not installed")
def test_real_1929_template_matches_later_rallies():
    m = data.load_monthly()
    ov = st.template_overlay(m["logp"], "1929-08-31", 24, 12)
    assert ov.index[0] > pd.Timestamp("1931-07-01")
    assert ov.loc["2014-01-31", "corr"] > 0.9          # the early-2014 overlay is real ...
    assert ov.loc["2014-01-31", "fwd"] > 0.0           # ... and no crash followed


@pytest.mark.skipif(not data.have_monthly(), reason="arch Fama-French tape not installed")
def test_real_monthly_headline_runs_and_is_finite():
    m = data.load_monthly()
    H = st.HEADLINE_MONTHLY
    logp = m["logp"].to_numpy()
    idx = st.eval_positions(m.index, OOS)
    s = st.analogue_search(logp, H["L"], idx, k_max=H["k"], gap=H["L"])
    fc = st.analogue_forecast(logp, s, H["h"], H["k"])
    ev = st.evaluate_forecast(fc, st.realised_forward(logp, idx, H["h"]),
                              st.historical_drift_forecast(logp, idx, H["h"]), H["h"])
    assert ev["n"] > 700 and np.isfinite(ev["t"])
    assert np.nanmedian(s["corr"][:, 0]) > 0.85        # spectacular matches, as advertised
