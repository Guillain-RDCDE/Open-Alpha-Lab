"""Strategy tests for Study 996 — the machinery of finding nothing, verified."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from palindrome import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The predicates themselves
# --------------------------------------------------------------------------- #
def test_palindrome_detection_by_hand():
    assert st._is_palindrome("22022022")
    assert st._is_palindrome("")
    assert not st._is_palindrome("20240101")


def test_a_known_palindromic_date_is_flagged():
    """2022-02-22 is 22022022 in DDMMYYYY — a genuine palindrome."""
    preds = st.date_predicates()
    d = pd.Timestamp("2022-02-22")
    assert preds["palindrome DDMMYYYY"](d)
    assert not preds["palindrome DDMMYYYY"](pd.Timestamp("2022-02-23"))


def test_primality_by_hand():
    for n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        assert st._is_prime(n)
    for n in (0, 1, 4, 6, 8, 9, 15, 21, 25, 27):
        assert not st._is_prime(n)


def test_there_are_enough_rules_for_the_point_to_land():
    preds = st.date_predicates()
    assert len(preds) >= 90
    assert len(preds) == len(set(preds))


def test_every_predicate_returns_a_bool_for_every_date():
    preds = st.date_predicates()
    idx = pd.bdate_range("2019-01-01", periods=400)
    for name, pred in preds.items():
        vals = [pred(d) for d in idx[:40]]
        assert all(isinstance(v, (bool, np.bool_)) for v in vals), name


def test_predicates_depend_only_on_the_date():
    """No rule may touch the returns — the search space is the calendar alone."""
    preds = st.date_predicates()
    idx = pd.bdate_range("2015-01-01", periods=300)
    a = {n: st.apply_predicate(idx, p).to_numpy().tolist() for n, p in preds.items()}
    b = {n: st.apply_predicate(idx, p).to_numpy().tolist() for n, p in preds.items()}
    assert a == b


def test_the_day_of_week_rules_partition_the_week():
    preds = st.date_predicates()
    idx = pd.bdate_range("2020-01-01", periods=500)
    total = sum(st.apply_predicate(idx, preds[d]).sum()
                for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"))
    assert total == len(idx)


def test_the_month_rules_partition_the_year():
    preds = st.date_predicates()
    idx = pd.bdate_range("2015-01-01", periods=2000)
    total = sum(st.apply_predicate(idx, preds[f"month {m}"]).sum() for m in range(1, 13))
    assert total == len(idx)


# --------------------------------------------------------------------------- #
# Testing one rule
# --------------------------------------------------------------------------- #
def test_a_planted_effect_is_detected():
    r = st.synthetic_returns(n=6000)
    mask = st.apply_predicate(r.index, lambda d: d.day == 1)
    r = r + mask * 0.01
    out = st.test_rule(r, mask)
    assert out["t"] > 5
    assert out["difference"] == pytest.approx(0.01, abs=0.003)


def test_no_effect_is_found_when_none_exists():
    ts = []
    for k in range(20):
        r = st.synthetic_returns(n=6000, seed=996 + k)
        mask = st.apply_predicate(r.index, lambda d: d.day == 1)
        ts.append(st.test_rule(r, mask)["t"])
    assert abs(np.mean(ts)) < 0.6
    assert (np.abs(ts) >= 2).mean() < 0.25


def test_test_rule_declines_on_too_few_hits():
    r = st.synthetic_returns(n=1000)
    mask = st.apply_predicate(r.index, lambda d: d.year == 1800)
    assert np.isnan(st.test_rule(r, mask)["t"])


# --------------------------------------------------------------------------- #
# The scan
# --------------------------------------------------------------------------- #
def test_the_scan_covers_every_usable_rule():
    r = st.synthetic_returns(n=8000)
    d = st.scan(r)
    assert len(d) > 80
    assert d["t"].abs().is_monotonic_decreasing


def test_the_scan_finds_something_significant_on_pure_noise():
    """The entire point of the study, as a single assertion."""
    r = st.synthetic_returns(n=8000)
    d = st.scan(r)
    assert (d["t"].abs() > 2).sum() >= 1


def test_the_number_of_false_positives_matches_the_test_count():
    r = st.synthetic_returns(n=8000)
    d = st.scan(r)
    s = st.multiple_testing_summary(d["t"])
    assert s["n_tests"] == len(d)
    # rules overlap heavily, so the count is noisy — but it should be in the right region
    assert 0 <= s["n_significant"] <= 6 * s["expected_by_luck"]


def test_scan_panel_stacks_every_asset():
    assets = {f"A{k}": st.synthetic_returns(n=5000, seed=996 + k) for k in range(3)}
    d = st.scan_panel(assets)
    assert set(d["asset"]) == {"A0", "A1", "A2"}
    assert len(d) > 200


# --------------------------------------------------------------------------- #
# Multiple-testing arithmetic
# --------------------------------------------------------------------------- #
def test_bonferroni_is_stricter_than_the_nominal_level():
    r = st.synthetic_returns(n=8000)
    s = st.multiple_testing_summary(st.scan(r)["t"])
    assert s["bonferroni_p"] < 0.05
    assert s["bonferroni_t"] > 1.96
    assert s["n_surviving_bonferroni"] <= s["n_significant"]


def test_benjamini_hochberg_is_between_nominal_and_bonferroni():
    r = st.synthetic_returns(n=8000)
    s = st.multiple_testing_summary(st.scan(r)["t"])
    assert s["n_surviving_bh"] <= s["n_significant"]
    assert s["n_surviving_bh"] >= s["n_surviving_bonferroni"]


def test_multiple_testing_summary_declines_on_a_tiny_scan():
    assert "bonferroni_t" not in st.multiple_testing_summary(pd.Series([1.0, 2.0]))


def test_the_expected_maximum_t_grows_with_the_number_of_tries():
    a = st.expected_max_t(10, n_sims=2000)
    b = st.expected_max_t(200, n_sims=2000)
    c = st.expected_max_t(5000, n_sims=2000)
    assert a["median"] < b["median"] < c["median"]


def test_two_hundred_tries_makes_a_t_of_two_unremarkable():
    """The number this whole study exists to publicise."""
    e = st.expected_max_t(200, n_sims=4000)
    assert e["median"] > 2.5
    assert e["share_above_2"] > 0.95


def test_a_single_test_has_the_normal_median():
    e = st.expected_max_t(1, n_sims=8000)
    assert e["median"] == pytest.approx(0.674, abs=0.05)


def test_deflated_t_is_about_one_for_a_typical_search_winner():
    assert st.deflated_t(2.8, 200) == pytest.approx(1.0, abs=0.25)
    assert st.deflated_t(6.0, 200) > 1.8
    assert st.deflated_t(2.0, 200) < 1.0


# --------------------------------------------------------------------------- #
# The shuffle test
# --------------------------------------------------------------------------- #
def test_the_shuffle_test_does_not_reject_on_pure_noise():
    """Across seeds, not on one: a single shuffle p-value is itself a random draw.

    At the 5% level one run in twenty rejects by construction, so asserting on a single seed
    would give a test that fails a twentieth of the time for the most boring possible reason.
    """
    ps = []
    for k in range(3):
        out = st.best_rule_distribution(st.synthetic_returns(n=6000, seed=996 + k),
                                        n_shuffles=60, seed=996 + k)
        ps.append(out["p_value"])
        assert out["null_median_max_t"] > 2.0
    assert float(np.median(ps)) > 0.05


def test_the_shuffle_test_does_reject_a_planted_effect():
    r = st.synthetic_returns(n=8000)
    preds = st.date_predicates()
    mask = st.apply_predicate(r.index, preds["Monday"])
    r = r + mask * 0.004
    out = st.best_rule_distribution(r, preds, n_shuffles=40)
    assert out["p_value"] < 0.10
    assert out["observed_max_t"] > out["null_median_max_t"]


def test_the_shuffle_null_exceeds_the_naive_threshold_by_a_lot():
    """Shuffling preserves the overlap between rules, which the theory ignores."""
    r = st.synthetic_returns(n=6000)
    out = st.best_rule_distribution(r, n_shuffles=40)
    assert out["null_median_max_t"] > 1.96


def test_best_rule_distribution_handles_an_empty_scan():
    r = st.synthetic_returns(n=30)
    assert st.best_rule_distribution(r, n_shuffles=3) == {}


# --------------------------------------------------------------------------- #
# Out of sample
# --------------------------------------------------------------------------- #
def test_the_best_rules_do_not_survive_the_split():
    r = st.synthetic_returns(n=10000)
    d = st.split_sample_check(r)
    assert len(d) == 10
    assert d["t_in_sample"].abs().mean() > d["t_out_of_sample"].abs().mean()


def test_out_of_sample_t_values_are_centred_on_zero():
    outs = []
    for k in range(4):
        r = st.synthetic_returns(n=8000, seed=996 + k)
        outs.extend(st.split_sample_check(r)["t_out_of_sample"].dropna().tolist())
    assert abs(np.mean(outs)) < 0.8


def test_split_sample_check_is_empty_on_a_short_series():
    assert st.split_sample_check(st.synthetic_returns(n=40)).empty


def test_the_traded_rule_underperforms_buy_and_hold():
    r = st.synthetic_returns(n=8000)
    d = st.scan(r)
    preds = st.date_predicates()
    mask = st.apply_predicate(r.index, preds[d.index[0]])
    out = st.tradable_check(r, mask, cost_bps=5.0)
    assert out["strategy"]["cagr"] < out["buy_hold"]["cagr"]


def test_tradable_check_charges_switching():
    r = st.synthetic_returns(n=5000)
    mask = st.apply_predicate(r.index, lambda d: d.dayofweek == 0)
    free = st.tradable_check(r, mask, cost_bps=0.0)
    paid = st.tradable_check(r, mask, cost_bps=50.0)
    assert paid["strategy"]["cagr"] < free["strategy"]["cagr"]
    assert free["switches_per_year"] > 50


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_rules": 118, "n_assets": 6, "n_tests": 708, "n_significant": 41,
         "expected_by_luck": 35.4, "best_rule": "digit sum == 19", "best_asset": "IWM",
         "best_t": 3.41, "best_ann": 0.062, "best_naive_odds": 1500.0,
         "null_median_max_t": 3.52, "null_p95_max_t": 4.10, "shuffle_p": 0.58,
         "bonferroni_t": 3.79, "n_surviving_bonferroni": 0, "expected_max": 3.2,
         "mean_is_t": 2.9, "mean_oos_t": 0.15, "median_oos_t": 0.2,
         "n_oos_survive": 5, "best_traded_gap": -0.031}
    h.update(over)
    return h


def test_verdict_is_busted_when_the_search_finds_nothing_real():
    assert st.verdict(_headline())["signal"] == "Busted"


def test_verdict_escalates_if_a_rule_somehow_survives():
    assert st.verdict(_headline(shuffle_p=0.01))["signal"] == "Partial"
    assert st.verdict(_headline(shuffle_p=0.01,
                                n_surviving_bonferroni=1))["signal"] == "Confirmed"


def test_verdict_tradability_is_a_mirage():
    assert st.verdict(_headline())["trad"] == "Mirage"
    assert st.verdict(_headline(median_oos_t=3.0))["trad"] == "Fragile"


def test_verdict_prose_states_the_maximum_distribution_point():
    v = st.verdict(_headline())
    assert "maximum" in v["signal_why"]
    assert "shuffl" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
