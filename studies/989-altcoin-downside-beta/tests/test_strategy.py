"""Strategy tests for Study 989 — the asymmetry, and the artefact that mimics it."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from onewaybeta import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Plain beta
# --------------------------------------------------------------------------- #
def test_ols_beta_recovers_a_planted_loading():
    df = st.synthetic_world(n=6000, beta_up=1.5, beta_down=1.5)
    assert st.ols_beta(df["alt"], df["bench"])["beta"] == pytest.approx(1.5, abs=0.08)


def test_beta_of_a_series_on_itself_is_one():
    df = st.synthetic_world(n=2000)
    assert st.ols_beta(df["bench"], df["bench"])["beta"] == pytest.approx(1.0, abs=1e-9)


def test_ols_beta_declines_on_too_little_data():
    df = st.synthetic_world(n=30)
    assert np.isnan(st.ols_beta(df["alt"], df["bench"])["beta"])


def test_align_intersects_the_two_calendars():
    df = st.synthetic_world(n=1000)
    y = df["alt"].iloc[100:]
    assert len(st.align(y, df["bench"])) == 900


# --------------------------------------------------------------------------- #
# Conditional betas
# --------------------------------------------------------------------------- #
def test_conditional_betas_recover_a_planted_asymmetry():
    df = st.synthetic_world(n=8000, beta_up=1.0, beta_down=2.0)
    cb = st.conditional_betas(df["alt"], df["bench"])
    assert cb["beta_up"] == pytest.approx(1.0, abs=0.15)
    assert cb["beta_down"] == pytest.approx(2.0, abs=0.15)
    assert cb["difference"] > 0.5


def test_conditional_betas_find_no_asymmetry_on_average_when_none_exists():
    diffs = []
    for s in range(12):
        df = st.synthetic_world(n=4000, beta_up=1.5, beta_down=1.5, seed=989 + s)
        diffs.append(st.conditional_betas(df["alt"], df["bench"])["difference"])
    assert abs(np.mean(diffs)) < 0.12


def test_but_a_single_symmetric_run_often_looks_asymmetric():
    """The study's central failure mode, pinned as a test."""
    hits = 0
    for s in range(20):
        df = st.synthetic_world(n=2000, beta_up=1.5, beta_down=1.5, seed=989 + s)
        cb = st.conditional_betas(df["alt"], df["bench"])
        hits += abs(st.naive_two_sample_t(cb)) >= 2
    assert hits >= 1        # the naive test fires on a symmetric world


def test_a_higher_threshold_uses_fewer_days():
    df = st.synthetic_world(n=4000)
    loose = st.conditional_betas(df["alt"], df["bench"], 0.0)
    tight = st.conditional_betas(df["alt"], df["bench"], 0.03)
    assert tight["n_down"] < loose["n_down"]
    assert tight["n_up"] < loose["n_up"]


def test_threshold_sweep_covers_every_definition():
    df = st.synthetic_world(n=5000)
    s = st.threshold_sweep(df["alt"], df["bench"])
    assert len(s) == 5
    assert s["n_down"].is_monotonic_decreasing


# --------------------------------------------------------------------------- #
# The named downside betas
# --------------------------------------------------------------------------- #
def test_bawa_lindenberg_matches_the_conditional_beta_when_asymmetry_is_planted():
    df = st.synthetic_world(n=8000, beta_up=1.0, beta_down=2.0)
    assert st.bawa_lindenberg_beta(df["alt"], df["bench"]) == pytest.approx(2.0, abs=0.2)


def test_hogan_warren_agrees_in_direction():
    sym = st.synthetic_world(n=8000, beta_up=1.5, beta_down=1.5)
    asym = st.synthetic_world(n=8000, beta_up=1.0, beta_down=2.0)
    assert (st.hogan_warren_beta(asym["alt"], asym["bench"])
            > st.hogan_warren_beta(sym["alt"], sym["bench"]))


def test_the_two_downside_betas_agree_with_each_other():
    df = st.synthetic_world(n=8000, beta_up=1.0, beta_down=2.2)
    bl = st.bawa_lindenberg_beta(df["alt"], df["bench"])
    hw = st.hogan_warren_beta(df["alt"], df["bench"])
    assert abs(bl - hw) < 0.4


def test_downside_betas_return_nan_on_too_few_down_days():
    df = st.synthetic_world(n=200)
    assert np.isnan(st.bawa_lindenberg_beta(df["alt"], df["bench"], target=-10.0))


def test_coskewness_is_zero_in_a_symmetric_world():
    cs = [st.coskewness(*[st.synthetic_world(n=6000, seed=989 + s)[c]
                          for c in ("alt", "bench")]) for s in range(8)]
    assert abs(np.mean(cs)) < 0.15


def test_coskewness_turns_negative_when_the_downside_beta_is_higher():
    asym = st.synthetic_world(n=12000, beta_up=0.8, beta_down=2.4)
    sym = st.synthetic_world(n=12000, beta_up=1.6, beta_down=1.6)
    assert st.coskewness(asym["alt"], asym["bench"]) < st.coskewness(sym["alt"], sym["bench"])


# --------------------------------------------------------------------------- #
# Tail correlation and its normal benchmark
# --------------------------------------------------------------------------- #
def test_tail_correlation_reports_the_normal_benchmark():
    df = st.synthetic_world(n=4000, beta_up=1.5, beta_down=1.5)
    tc = st.tail_correlation(df["alt"], df["bench"])
    assert set(("down_tail", "normal_down_tail", "excess_down")) <= set(tc)


def test_a_normal_world_shows_lower_tail_correlation_than_overall():
    """Longin-Solnik: conditioning on the tail LOWERS measured correlation under normality."""
    df = st.synthetic_world(n=8000, beta_up=1.5, beta_down=1.5)
    tc = st.tail_correlation(df["alt"], df["bench"])
    assert tc["normal_down_tail"] < tc["overall"]


def test_the_excess_tail_correlation_is_near_zero_under_symmetry():
    df = st.synthetic_world(n=8000, beta_up=1.5, beta_down=1.5)
    assert abs(st.tail_correlation(df["alt"], df["bench"])["excess_down"]) < 0.2


def test_tail_correlation_declines_on_a_short_series():
    df = st.synthetic_world(n=100)
    assert "down_tail" not in st.tail_correlation(df["alt"], df["bench"])


# --------------------------------------------------------------------------- #
# Testing the difference honestly
# --------------------------------------------------------------------------- #
def test_the_bootstrap_is_wider_than_the_naive_two_sample_test():
    """The single most important assertion in this module."""
    df = st.synthetic_world(n=6000, beta_up=1.2, beta_down=1.8)
    cb = st.conditional_betas(df["alt"], df["bench"])
    at = st.asymmetry_test(df["alt"], df["bench"], n_boot=300)
    assert abs(at["t"]) < abs(st.naive_two_sample_t(cb))


def test_the_bootstrap_still_finds_a_large_planted_asymmetry():
    df = st.synthetic_world(n=8000, beta_up=0.8, beta_down=2.4)
    assert st.asymmetry_test(df["alt"], df["bench"], n_boot=300)["t"] > 2


def test_the_bootstrap_does_not_find_asymmetry_that_is_not_there():
    ts = []
    for s in range(8):
        df = st.synthetic_world(n=4000, beta_up=1.5, beta_down=1.5, seed=989 + s)
        ts.append(st.asymmetry_test(df["alt"], df["bench"], n_boot=200)["t"])
    ts = np.abs(np.array(ts))
    assert np.nanmean(ts) < 2.0
    assert np.nanmean(ts >= 2) <= 0.35


def test_asymmetry_test_keeps_the_point_estimate():
    df = st.synthetic_world(n=4000, beta_up=1.0, beta_down=2.0)
    at = st.asymmetry_test(df["alt"], df["bench"], n_boot=200)
    cb = st.conditional_betas(df["alt"], df["bench"])
    assert at["difference"] == pytest.approx(cb["difference"], abs=1e-9)


def test_asymmetry_test_declines_on_a_short_series():
    df = st.synthetic_world(n=200)
    assert "t" not in st.asymmetry_test(df["alt"], df["bench"])


# --------------------------------------------------------------------------- #
# Controls and outputs
# --------------------------------------------------------------------------- #
def test_time_varying_control_splits_into_eras():
    df = st.synthetic_world(n=4000, beta_up=1.2, beta_down=1.8)
    e = st.time_varying_control(df["alt"], df["bench"], n_eras=4)
    assert len(e) == 4
    assert (e["difference"] > 0).sum() >= 2


def test_panel_summary_runs_every_measurement():
    alts = {f"ALT{k}": st.synthetic_world(n=3000, seed=989 + k)["alt"] for k in range(3)}
    bench = st.synthetic_world(n=3000)["bench"]
    p = st.panel_summary(alts, bench)
    assert len(p) == 3
    for c in ("beta_up", "beta_down", "bawa_lindenberg", "hogan_warren", "coskewness"):
        assert c in p.columns


def test_capture_ratios_are_one_for_the_benchmark_against_itself():
    df = st.synthetic_world(n=3000)
    c = st.capture_ratios(df["bench"], df["bench"])
    assert c["up_capture"] == pytest.approx(1.0, abs=1e-9)
    assert c["down_capture"] == pytest.approx(1.0, abs=1e-9)


def test_capture_ratios_reflect_a_planted_asymmetry():
    df = st.synthetic_world(n=8000, beta_up=1.0, beta_down=2.0, idio_vol=0.005)
    c = st.capture_ratios(df["alt"], df["bench"])
    assert c["down_capture"] > c["up_capture"]


def test_capture_ratios_do_not_saturate_on_a_long_sample():
    """The textbook compounded definition would return ~1.0 here whatever the truth."""
    short = st.synthetic_world(n=1000, beta_up=1.0, beta_down=2.0, idio_vol=0.005)
    long = st.synthetic_world(n=12000, beta_up=1.0, beta_down=2.0, idio_vol=0.005)
    cs = st.capture_ratios(short["alt"], short["bench"])
    cl = st.capture_ratios(long["alt"], long["bench"])
    assert cl["down_capture"] > 1.5
    assert abs(cl["down_capture"] - cs["down_capture"]) < 0.5


def test_capture_ratios_decline_on_a_short_series():
    df = st.synthetic_world(n=40)
    assert "up_capture" not in st.capture_ratios(df["alt"], df["bench"])


def test_drawdown_comparison_is_neutral_for_a_series_against_itself():
    df = st.synthetic_world(n=3000)
    px = (1 + df["bench"]).cumprod()
    d = st.drawdown_comparison(px, px)
    assert d["dd_ratio"] == pytest.approx(1.0)


def test_a_higher_beta_asset_draws_down_further():
    df = st.synthetic_world(n=6000, beta_up=2.0, beta_down=2.0, idio_vol=0.005)
    d = st.drawdown_comparison((1 + df["alt"]).cumprod(), (1 + df["bench"]).cumprod())
    assert d["dd_ratio"] > 1.2


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_alts": 6, "years": 8.6, "median_beta_up": 1.02, "median_beta_down": 1.19,
         "median_difference": 0.17, "median_naive_t": 2.9, "median_boot_t": 1.3,
         "share_negative_coskew": 0.67, "null_false_positive": 0.28,
         "median_up_capture": 0.88, "median_down_capture": 1.05,
         "median_max_dd": -0.93, "bench_max_dd": -0.77, "median_dd_ratio": 1.21}
    h.update(over)
    return h


def test_verdict_signal_needs_bootstrap_significance_and_corroboration():
    assert st.verdict(_headline())["signal"] == "Weak"
    assert st.verdict(_headline(median_boot_t=2.5))["signal"] == "Real"
    assert st.verdict(_headline(median_boot_t=2.5,
                                share_negative_coskew=0.2))["signal"] == "Weak"
    assert st.verdict(_headline(median_difference=-0.05))["signal"] == "None"


def test_a_signal_that_is_not_real_cannot_be_tradable():
    for over in ({}, {"median_difference": -0.05}):
        assert st.verdict(_headline(**over))["trad"] == "Mirage"


def test_verdict_tradability_ladder_when_the_signal_is_real():
    real = {"median_boot_t": 2.5}
    assert st.verdict(_headline(**real))["trad"] == "Partial"
    assert st.verdict(_headline(median_down_capture=1.30, **real))["trad"] == "Useful"


def test_verdict_prose_names_the_false_positive_rate():
    v = st.verdict(_headline())
    assert "symmetric" in v["one_sentence"]
    assert "naive" in v["signal_why"] and "bootstrap" in v["signal_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
