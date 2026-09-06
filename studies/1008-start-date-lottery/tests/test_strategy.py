"""Strategy tests for Study 1008 — sequence risk, isolated and priced."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from startdate import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
def test_a_lump_sum_does_not_care_about_order():
    """Multiplication commutes. The foundation of the whole study."""
    rng = np.random.default_rng(1008)
    r = rng.normal(0.0004, 0.011, 5000)
    a = st.terminal_lump_sum(r)
    b = st.terminal_lump_sum(rng.permutation(r))
    assert a == pytest.approx(b, rel=1e-12)


def test_a_contributor_cares_enormously_about_order():
    rng = np.random.default_rng(1008)
    r = rng.normal(0.0004, 0.011, 5000)
    outs = [st.terminal_with_contributions(rng.permutation(r), 1.0)
            for _ in range(50)]
    assert np.std(outs) / np.mean(outs) > 0.05


def test_rising_then_falling_beats_nothing_for_a_contributor():
    """The mechanism in miniature: money added late has no time to compound."""
    up = np.concatenate([np.full(500, 0.002), np.full(500, -0.001)])
    down = up[::-1].copy()
    assert st.terminal_lump_sum(up) == pytest.approx(st.terminal_lump_sum(down),
                                                     rel=1e-12)
    assert st.terminal_with_contributions(down, 1.0) > \
        st.terminal_with_contributions(up, 1.0)


def test_contributions_accumulate_without_returns():
    assert st.terminal_with_contributions(np.zeros(100), 1.0) == pytest.approx(100.0)


def test_an_initial_balance_compounds():
    r = np.full(252, 0.0004)
    a = st.terminal_with_contributions(r, 0.0, initial=1.0)
    assert a == pytest.approx(1.0004 ** 252, rel=1e-9)


def test_withdrawals_can_exhaust_a_portfolio():
    out = st.terminal_with_withdrawals(np.full(1000, -0.001), 0.002, 1.0)
    assert out["ruined"]
    assert out["terminal"] == 0.0
    assert 0 <= out["ruined_at"] < 1000


def test_a_portfolio_that_survives_is_not_flagged_ruined():
    out = st.terminal_with_withdrawals(np.full(1000, 0.001), 0.0002, 1.0)
    assert not out["ruined"]
    assert out["terminal"] > 1.0


def test_withdrawals_punish_a_bad_START_not_a_bad_end():
    """The mirror image of the contributor case, and the reason retirees are told to de-risk."""
    bad_first = np.concatenate([np.full(500, -0.0015), np.full(500, 0.002)])
    bad_last = bad_first[::-1].copy()
    a = st.terminal_with_withdrawals(bad_first, 0.0005, 1.0)
    b = st.terminal_with_withdrawals(bad_last, 0.0005, 1.0)
    assert b["terminal"] > a["terminal"]


# --------------------------------------------------------------------------- #
# The shuffle test
# --------------------------------------------------------------------------- #
def test_the_shuffle_leaves_a_lump_sum_untouched():
    r = st.synthetic_path(n_days=5000).to_numpy()
    s = st.shuffle_invariance(r, n_shuffles=60)
    assert s["lump_cv"] < 1e-12


def test_the_shuffle_scatters_a_contributor():
    r = st.synthetic_path(n_days=5000).to_numpy()
    s = st.shuffle_invariance(r, n_shuffles=200)
    assert s["contrib_cv"] > 0.02
    assert s["sequence_spread"] > 1.05


def test_pure_sequence_risk_grows_with_volatility():
    lo = st.shuffle_invariance(st.synthetic_path(n_days=5000, vol=0.08).to_numpy(),
                               n_shuffles=200)
    hi = st.shuffle_invariance(st.synthetic_path(n_days=5000, vol=0.32).to_numpy(),
                               n_shuffles=200)
    assert hi["contrib_cv"] > lo["contrib_cv"]


def test_a_constant_return_path_has_no_sequence_risk():
    """The control: no variation in returns, nothing for the order to rearrange."""
    s = st.shuffle_invariance(np.full(2000, 0.0004), n_shuffles=40)
    assert s["contrib_cv"] < 1e-12


# --------------------------------------------------------------------------- #
# Real paths
# --------------------------------------------------------------------------- #
def test_accumulation_paths_cover_every_start_date():
    px = data.load_prices()
    r = _eq(px)
    p = st.accumulation_paths(r, years=20)
    assert len(p) > 20
    assert p["start"].is_monotonic_increasing
    assert (p["terminal"] > 0).all()


def test_the_start_date_matters_a_great_deal():
    px = data.load_prices()
    r = _eq(px)
    d = st.path_dispersion(st.accumulation_paths(r, years=20), "multiple")
    assert d["ratio_max_min"] > 1.5


def test_the_effective_sample_is_reported_and_small():
    px = data.load_prices()
    r = _eq(px)
    d = st.path_dispersion(st.accumulation_paths(r, years=20), "multiple")
    assert d["n_paths"] > 50
    assert d["effective_n"] < 3.0


def test_accumulation_is_empty_when_the_horizon_exceeds_the_data():
    px = data.load_prices()
    assert st.accumulation_paths(_eq(px), years=200).empty
    assert st.path_dispersion(pd.DataFrame()) == {}


def test_decumulation_paths_report_ruin():
    px = data.load_prices()
    r = _eq(px)
    p = st.decumulation_paths(r, years=20, withdrawal_rate=0.04)
    assert len(p) > 20
    assert set(p.columns) >= {"terminal", "ruined", "ruined_at", "first_5y_cagr"}


def test_a_higher_withdrawal_rate_ruins_more_often():
    px = data.load_prices()
    r = _eq(px)
    lo = st.decumulation_paths(r, years=20, withdrawal_rate=0.03)["ruined"].mean()
    hi = st.decumulation_paths(r, years=20, withdrawal_rate=0.12)["ruined"].mean()
    assert hi > lo


def test_a_bad_first_five_years_predicts_a_worse_retirement():
    """Sequence risk for retirees, on the real tape."""
    px = data.load_prices()
    r = _eq(px)
    p = st.decumulation_paths(r, years=20, withdrawal_rate=0.05)
    assert p["first_5y_cagr"].corr(p["terminal"]) > 0.3


# --------------------------------------------------------------------------- #
# Where the risk lives
# --------------------------------------------------------------------------- #
def test_a_lump_sum_weights_every_period_equally():
    """Which is what makes the contributor's profile meaningful by comparison."""
    px = data.load_prices()
    r = _eq(px)
    m = st.sequence_risk_metrics(r, years=20, n_buckets=4)
    spread = m["corr_lump_sum"].max() - m["corr_lump_sum"].min()
    assert spread < 0.45


def test_a_contributor_is_dominated_by_the_final_years():
    px = data.load_prices()
    r = _eq(px)
    m = st.sequence_risk_metrics(r, years=20, n_buckets=4)
    assert m["corr_contributor"].iloc[-1] > m["corr_contributor"].iloc[0]


def test_the_contributor_profile_is_more_tilted_than_the_lump_sum_profile():
    px = data.load_prices()
    r = _eq(px)
    m = st.sequence_risk_metrics(r, years=20, n_buckets=4)
    c = m["corr_contributor"].iloc[-1] - m["corr_contributor"].iloc[0]
    l = m["corr_lump_sum"].iloc[-1] - m["corr_lump_sum"].iloc[0]
    assert c > l


def test_sequence_metrics_decline_on_too_few_paths():
    px = data.load_prices()
    assert st.sequence_risk_metrics(_eq(px), years=200).empty


# --------------------------------------------------------------------------- #
# Glide paths
# --------------------------------------------------------------------------- #
def test_glide_paths_start_and_end_where_they_say():
    for shape in ("linear", "late", "early"):
        w = st.glide_path(1000, 1.0, 0.3, shape)
        assert w[0] == pytest.approx(1.0)
        assert w[-1] == pytest.approx(0.3)


def test_a_constant_glide_is_constant():
    w = st.glide_path(500, 0.7, 0.7, "constant")
    assert np.allclose(w, 0.7)


def test_the_late_glide_holds_equity_longer_than_the_early_one():
    late = st.glide_path(1000, 1.0, 0.3, "late")
    early = st.glide_path(1000, 1.0, 0.3, "early")
    assert late.sum() > early.sum()
    assert late[400] > early[400]


def test_an_unknown_glide_shape_is_rejected():
    with pytest.raises(ValueError):
        st.glide_path(100, 1.0, 0.3, "sigmoid")


def test_glided_accumulation_matches_pure_equity_when_the_weight_is_one():
    px = data.load_prices()
    e, b = _pair(px)
    g = st.glided_accumulation(e, b, years=15, shape="constant",
                               start_weight=1.0, end_weight=1.0)
    direct = st.accumulation_paths(e.reindex(
        e.index.intersection(b.dropna().index)).dropna(), years=15)
    assert len(g) == len(direct)
    assert g["terminal"].iloc[0] == pytest.approx(direct["terminal"].iloc[0], rel=1e-9)


def test_de_risking_reduces_dispersion():
    px = data.load_prices()
    e, b = _pair(px)
    full = st.path_dispersion(st.glided_accumulation(
        e, b, years=15, shape="constant", start_weight=1.0, end_weight=1.0), "multiple")
    glided = st.path_dispersion(st.glided_accumulation(
        e, b, years=15, shape="linear", end_weight=0.3), "multiple")
    assert glided["cv"] < full["cv"]


def test_de_risking_also_costs_expected_wealth():
    """A remedy has two columns, and studies that print only one are selling something."""
    px = data.load_prices()
    e, b = _pair(px)
    full = st.path_dispersion(st.glided_accumulation(
        e, b, years=15, shape="constant", start_weight=1.0, end_weight=1.0), "multiple")
    glided = st.path_dispersion(st.glided_accumulation(
        e, b, years=15, shape="linear", end_weight=0.3), "multiple")
    assert glided["median"] < full["median"]


def test_glided_accumulation_is_empty_when_the_horizon_exceeds_the_data():
    px = data.load_prices()
    e, b = _pair(px)
    assert st.glided_accumulation(e, b, years=200).empty


# --------------------------------------------------------------------------- #
# The remedy table
# --------------------------------------------------------------------------- #
def test_the_remedy_table_scores_every_variant():
    px = data.load_prices()
    e, b = _pair(px)
    t = st.remedy_comparison(e, b, years=15)
    assert "100% equity throughout" in t.index
    assert len(t) >= 5
    assert t.loc["100% equity throughout", "cv_reduction"] == 0.0


def test_every_remedy_reduces_dispersion_and_costs_something():
    px = data.load_prices()
    e, b = _pair(px)
    t = st.remedy_comparison(e, b, years=15).drop(index="100% equity throughout")
    assert (t["cv_reduction"] > 0).all()


def test_late_de_risking_is_more_efficient_than_early():
    """The study's practical claim, on the real tape."""
    px = data.load_prices()
    e, b = _pair(px)
    t = st.remedy_comparison(e, b, years=15)
    assert t.loc["late glide to 30%", "efficiency"] > \
        t.loc["early glide to 30%", "efficiency"]


def test_remedy_comparison_is_empty_without_enough_data():
    px = data.load_prices()
    e, b = _pair(px)
    assert st.remedy_comparison(e, b, years=200).empty


# --------------------------------------------------------------------------- #
# The synthetic control
# --------------------------------------------------------------------------- #
def test_the_lottery_grows_with_volatility_on_independent_paths():
    d = st.lottery_size_by_volatility(vols=(0.08, 0.16, 0.32), n_paths=120, years=20)
    assert d["contrib_cv"].is_monotonic_increasing
    assert d["lump_cv"].is_monotonic_increasing


def test_a_contributor_faces_less_dispersion_than_a_lump_sum():
    """Averaging in is a genuine benefit, and it is worth stating the direction correctly."""
    d = st.lottery_size_by_volatility(vols=(0.16,), n_paths=200, years=20)
    assert d.loc[0.16, "contrib_cv"] < d.loc[0.16, "lump_cv"]


def test_the_synthetic_path_has_the_volatility_it_claims():
    r = st.synthetic_path(n_days=40000, vol=0.16)
    assert r.std(ddof=1) * np.sqrt(252) == pytest.approx(0.16, rel=0.05)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _eq(px):
    return px[data.EQUITY].dropna().pct_change().dropna()


def _pair(px):
    e = px[data.EQUITY].dropna().pct_change()
    b = px[data.BONDS].dropna().pct_change()
    idx = e.dropna().index.intersection(b.dropna().index)
    return e.reindex(idx), b.reindex(idx)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"asset": "SPY", "years": 20.0, "best_multiple": 3.41, "worst_multiple": 1.62,
         "ratio_max_min": 2.10, "ratio_95_05": 1.94, "best_start": "2003-03-31",
         "worst_start": "1999-08-31", "effective_n": 1.6,
         "lump_ratio_95_05": 2.85, "shuffle_lump_cv": 1e-15,
         "shuffle_sequence_spread": 1.31, "last_bucket_corr": 0.71,
         "first_bucket_corr": 0.11, "best_remedy": "late glide to 30%",
         "best_cv_reduction": 0.34, "best_median_cost": 0.11,
         "best_efficiency": 3.09, "linear_efficiency": 1.42,
         "sixty_forty_efficiency": 1.05, "best_ratio_95_05": 1.55}
    h.update(over)
    return h


def test_verdict_signal_keys_off_the_spread():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(ratio_95_05=1.5))["signal"] == "Weak"
    assert st.verdict(_headline(ratio_95_05=1.1))["signal"] == "None"


def test_verdict_tradability_keys_off_efficiency_not_dispersion_alone():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(best_efficiency=1.0))["trad"] == "Partial"
    assert st.verdict(_headline(best_efficiency=0.5))["trad"] == "Mirage"


def test_verdict_prose_separates_sequence_from_distribution():
    v = st.verdict(_headline())
    assert "shuffle test" in v["signal_why"]
    assert "pure sequence risk" in v["signal_why"]
    assert "effective_n" not in v["signal_why"]
    assert "not on this table at all" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
