"""Strategy tests for Study 994 — the arithmetic checked by hand first."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from roundingtax import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The arithmetic, computable by hand
# --------------------------------------------------------------------------- #
def test_a_clean_case_allocates_exactly():
    """$1000, 50/50, $100 and $50 shares: 5 and 10 shares, no cash left."""
    prices = pd.Series({"A": 100.0, "B": 50.0})
    a = st.whole_share_allocation({"A": 0.5, "B": 0.5}, prices, 1000.0)
    assert a["shares"] == {"A": 5.0, "B": 10.0}
    assert a["cash"] == pytest.approx(0.0)
    assert a["weights"]["A"] == pytest.approx(0.5)


def test_an_expensive_share_forces_a_bad_fit():
    """$1000, 50/50, one share costs $600: the $500 sleeve cannot be filled at all.

    Flooring gives A:0, B:10 and leaves $500 idle. Largest remainder then hands one extra
    share to the position with the biggest fractional shortfall it can afford — B — so the
    account ends up 55% in B, 0% in A, and $450 in cash against a 50/50 plan.
    """
    prices = pd.Series({"A": 600.0, "B": 50.0})
    a = st.whole_share_allocation({"A": 0.5, "B": 0.5}, prices, 1000.0)
    assert a["shares"]["A"] == 0.0            # a $500 target cannot buy a $600 share
    assert a["weights"]["A"] == 0.0
    assert a["shares"]["B"] == 11.0
    assert a["cash"] == pytest.approx(450.0)
    # L1 = |0 - 0.50| + |0.55 - 0.50|
    assert st.allocation_error(a)["l1"] == pytest.approx(0.55)


def test_fractional_shares_hit_the_target_exactly():
    prices = pd.Series({"A": 613.37, "B": 47.11, "C": 92.5})
    tgt = {"A": 0.5, "B": 0.3, "C": 0.2}
    a = st.whole_share_allocation(tgt, prices, 7777.0, allow_fractional=True)
    for t, w in tgt.items():
        assert a["weights"][t] == pytest.approx(w)
    assert a["cash"] == pytest.approx(0.0, abs=1e-9)


def test_largest_remainder_beats_naive_flooring():
    """The allocator must not leave cash on the table that a real investor would spend."""
    prices = pd.Series({"A": 99.0, "B": 98.0, "C": 97.0})
    tgt = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    a = st.whole_share_allocation(tgt, prices, 1000.0)
    naive_cash = 1000.0 - sum(np.floor(1000 / 3 / prices[t]) * prices[t] for t in tgt)
    assert a["cash"] < naive_cash
    assert a["cash"] >= 0


def test_largest_remainder_adds_at_most_one_share_per_position():
    """The apportionment rule, not a greedy spend-down of the residue."""
    prices = pd.Series({"A": 600.0, "B": 50.0})
    a = st.whole_share_allocation({"A": 0.5, "B": 0.5}, prices, 1000.0)
    # flooring gives A:0 B:10; the leftover $500 may buy ONE more of each it can afford
    assert a["shares"]["B"] <= 11.0
    assert a["shares"]["A"] == 0.0


def test_the_allocator_never_overspends():
    rng = np.random.default_rng(994)
    for _ in range(50):
        prices = pd.Series({f"F{k}": float(rng.uniform(5, 700)) for k in range(5)})
        w = rng.dirichlet(np.ones(5))
        tgt = {f"F{k}": float(w[k]) for k in range(5)}
        cap = float(rng.uniform(200, 50000))
        a = st.whole_share_allocation(tgt, prices, cap)
        assert a["cash"] >= -1e-9
        assert a["invested"] <= cap + 1e-9


def test_share_counts_are_whole_numbers():
    rng = np.random.default_rng(994)
    prices = pd.Series({f"F{k}": float(rng.uniform(20, 400)) for k in range(6)})
    tgt = {f"F{k}": 1 / 6 for k in range(6)}
    a = st.whole_share_allocation(tgt, prices, 12345.0)
    for v in a["shares"].values():
        assert v == int(v)


def test_a_zero_capital_account_holds_nothing():
    prices = pd.Series({"A": 100.0})
    a = st.whole_share_allocation({"A": 1.0}, prices, 0.0)
    assert a["invested"] == 0.0 and a["shares"] == {}


def test_missing_or_zero_prices_are_skipped():
    prices = pd.Series({"A": 100.0, "B": np.nan, "C": 0.0})
    a = st.whole_share_allocation({"A": 0.5, "B": 0.3, "C": 0.2}, prices, 1000.0)
    assert set(a["shares"]) == {"A"}


# --------------------------------------------------------------------------- #
# Error measurement
# --------------------------------------------------------------------------- #
def test_allocation_error_is_zero_on_a_perfect_fit():
    prices = pd.Series({"A": 100.0, "B": 50.0})
    e = st.allocation_error(st.whole_share_allocation({"A": 0.5, "B": 0.5}, prices, 1000.0))
    assert e["l1"] == pytest.approx(0.0)
    assert e["cash_share"] == pytest.approx(0.0)


def test_allocation_error_shrinks_with_account_size():
    prices = pd.Series({"A": 600.0, "B": 47.0, "C": 91.0})
    tgt = {"A": 0.5, "B": 0.3, "C": 0.2}
    small = st.allocation_error(st.whole_share_allocation(tgt, prices, 2000.0))["l1"]
    large = st.allocation_error(st.whole_share_allocation(tgt, prices, 500_000.0))["l1"]
    assert large < small / 10


def test_error_vs_capital_is_broadly_decreasing():
    prices = pd.Series({"A": 600.0, "B": 47.0, "C": 91.0, "D": 210.0})
    tgt = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    d = st.error_vs_capital(tgt, prices)
    assert d["l1_error"].iloc[0] > d["l1_error"].iloc[-1]
    assert d["l1_error"].iloc[-1] < 0.01


def test_the_one_share_cost_understates_what_is_needed():
    """The figure everyone quotes against the figure that matters."""
    prices = pd.Series({"A": 600.0, "B": 47.0, "C": 91.0, "D": 210.0})
    tgt = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    assert st.min_viable_capital(tgt, prices) > 5 * st.one_share_cost(tgt, prices)


def test_min_viable_capital_actually_achieves_the_tolerance():
    prices = pd.Series({"A": 600.0, "B": 47.0, "C": 91.0})
    tgt = {"A": 0.5, "B": 0.3, "C": 0.2}
    cap = st.min_viable_capital(tgt, prices, tol=0.01)
    assert st.allocation_error(st.whole_share_allocation(tgt, prices, cap))["l1"] <= 0.011


def test_cheaper_shares_lower_the_capital_requirement():
    tgt = {"A": 0.5, "B": 0.5}
    dear = pd.Series({"A": 600.0, "B": 500.0})
    cheap = pd.Series({"A": 30.0, "B": 25.0})
    assert st.min_viable_capital(tgt, cheap) < st.min_viable_capital(tgt, dear) / 5


# --------------------------------------------------------------------------- #
# Forward simulation
# --------------------------------------------------------------------------- #
def test_a_large_account_matches_the_fractional_ideal():
    px = st.synthetic_prices(n=1260)
    tgt = {f"F{k}": 1 / 6 for k in range(6)}
    c = st.compare_to_ideal(px, tgt, 5_000_000.0)
    assert abs(c["cagr_gap"]) < 0.001
    assert c["tracking_error"] < 0.002


def test_a_small_account_does_not():
    px = st.synthetic_prices(n=1260)
    tgt = {f"F{k}": 1 / 6 for k in range(6)}
    c = st.compare_to_ideal(px, tgt, 3000.0)
    assert c["tracking_error"] > 0.002
    assert c["mean_l1_error"] > 0.01


def test_tracking_error_falls_as_the_account_grows():
    px = st.synthetic_prices(n=1260)
    tgt = {f"F{k}": 1 / 6 for k in range(6)}
    tes = [st.compare_to_ideal(px, tgt, c)["tracking_error"]
           for c in (2000, 20_000, 200_000)]
    assert tes[0] > tes[1] > tes[2]


def test_expensive_shares_hurt_more_than_cheap_ones():
    """The mechanism, isolated: same returns, different share prices."""
    dear = st.synthetic_prices(n=1260, price_levels=(600.0,) * 6)
    cheap = st.synthetic_prices(n=1260, price_levels=(15.0,) * 6)
    tgt = {f"F{k}": 1 / 6 for k in range(6)}
    assert (st.compare_to_ideal(dear, tgt, 5000.0)["mean_l1_error"]
            > st.compare_to_ideal(cheap, tgt, 5000.0)["mean_l1_error"] * 3)


def test_a_small_account_carries_more_uninvested_cash():
    px = st.synthetic_prices(n=1260)
    tgt = {f"F{k}": 1 / 6 for k in range(6)}
    small = st.rebalance_simulation(px, tgt, 3000.0)
    large = st.rebalance_simulation(px, tgt, 1_000_000.0)
    assert small["mean_cash_share"] > large["mean_cash_share"]


def test_a_wider_no_trade_band_means_fewer_rebalances():
    px = st.synthetic_prices(n=2520)
    tgt = {f"F{k}": 1 / 6 for k in range(6)}
    tight = st.rebalance_simulation(px, tgt, 20_000.0, band=0.0)
    wide = st.rebalance_simulation(px, tgt, 20_000.0, band=0.20)
    assert wide["n_rebalances"] <= tight["n_rebalances"]
    assert wide["n_skipped"] >= tight["n_skipped"]


def test_costs_reduce_the_final_value():
    px = st.synthetic_prices(n=2520)
    tgt = {f"F{k}": 1 / 6 for k in range(6)}
    free = st.rebalance_simulation(px, tgt, 20_000.0, cost_bps=0.0)
    paid = st.rebalance_simulation(px, tgt, 20_000.0, cost_bps=100.0)
    assert paid["final"] < free["final"]
    assert paid["total_costs"] > 0


def test_the_simulation_declines_on_a_short_panel():
    px = st.synthetic_prices(n=30)
    assert "cagr" not in st.rebalance_simulation(px, {"F0": 1.0}, 10_000.0)


def test_cash_earns_the_cash_rate():
    px = st.synthetic_prices(n=1260)
    tgt = {"F0": 1.0}
    idx = px.index
    zero = st.rebalance_simulation(px, tgt, 1000.0,
                                   cash_rate=pd.Series(0.0, index=idx))
    paid = st.rebalance_simulation(px, tgt, 1000.0,
                                   cash_rate=pd.Series(0.05 / 252, index=idx))
    assert paid["final"] >= zero["final"]


# --------------------------------------------------------------------------- #
# The decomposition — noise versus drag
# --------------------------------------------------------------------------- #
def test_the_decomposition_adds_up():
    px = st.synthetic_prices(n=2520)
    tgt = {f"F{k}": 1 / 6 for k in range(6)}
    c = st.compare_to_ideal(px, tgt, 4000.0)
    d = st.decompose_shortfall(c, 0.02, 0.06)
    assert (d["cash_drag"] + d["trading_costs"] + d["unexplained_noise"]
            == pytest.approx(d["total_gap"], abs=1e-9))


def test_cash_drag_is_always_a_cost():
    px = st.synthetic_prices(n=1260)
    tgt = {f"F{k}": 1 / 6 for k in range(6)}
    d = st.decompose_shortfall(st.compare_to_ideal(px, tgt, 3000.0), 0.02, 0.06)
    assert d["cash_drag"] <= 0
    assert d["trading_costs"] <= 0


def test_the_noise_term_can_go_either_way():
    """Allocation error is mean-zero, so across seeds the residual must change sign."""
    signs = set()
    for seed in range(6):
        px = st.synthetic_prices(n=1260, seed=994 + seed)
        tgt = {f"F{k}": 1 / 6 for k in range(6)}
        d = st.decompose_shortfall(st.compare_to_ideal(px, tgt, 4000.0), 0.02, 0.06)
        signs.add(np.sign(d["unexplained_noise"]))
    assert len(signs) > 1


def test_decompose_handles_a_failed_comparison():
    assert st.decompose_shortfall({"capital": 100.0}, 0.02, 0.06) == {}


# --------------------------------------------------------------------------- #
# The escapes
# --------------------------------------------------------------------------- #
def test_fewer_funds_renormalises_to_one():
    tgt = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    f = st.fewer_funds_variant(tgt, 2)
    assert set(f) == {"A", "B"}
    assert sum(f.values()) == pytest.approx(1.0)


def test_fewer_funds_reduces_the_allocation_error():
    px = st.synthetic_prices(n=1260)
    tgt = {f"F{k}": 1 / 6 for k in range(6)}
    full = st.rebalance_simulation(px, tgt, 3000.0)["mean_l1_error"]
    three = st.rebalance_simulation(px, st.fewer_funds_variant(tgt, 3), 3000.0)["mean_l1_error"]
    assert three < full


def test_cheaper_share_swap_merges_duplicate_targets():
    tgt = {"A": 0.5, "B": 0.3, "C": 0.2}
    out = st.cheaper_share_variant(tgt, {"A": "Z", "B": "Z"})
    assert out["Z"] == pytest.approx(0.8)
    assert sum(out.values()) == pytest.approx(1.0)


def test_escape_table_prices_every_variant():
    px = st.synthetic_prices(n=1260)
    tgt = {f"F{k}": 1 / 6 for k in range(6)}
    e = st.escape_table(px, tgt, 3000.0)
    assert "fractional shares" in e.index
    assert "three funds instead" in e.index
    # Not zero: the reported error is averaged over every day, and weights drift between the
    # annual rebalances even with perfect fractional execution. What must hold is that the
    # fractional portfolio is closer to target than the whole-share one.
    assert (e.loc["fractional shares", "mean_l1_error"]
            < e.loc["whole shares, as specified", "mean_l1_error"])
    assert e.loc["fractional shares", "mean_cash_share"] < 1e-6


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"capital": 3000.0, "n_funds": 6, "mean_l1_error": 0.086, "max_abs_error": 0.041,
         "mean_cash_share": 0.021, "tracking_error": 0.0112, "years": 18.0,
         "one_share_cost": 1180.0, "min_viable": 42_000.0,
         "cagr_whole": 0.0712, "cagr_fractional": 0.0738, "cagr_gap": -0.0026,
         "cash_drag": -0.0013, "trading_costs": -0.0004, "unexplained_noise": -0.0009,
         "best_escape_error": 0.031}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(mean_l1_error=0.005))["signal"] == "Weak"
    assert st.verdict(_headline(mean_l1_error=0.005,
                                tracking_error=0.001))["signal"] == "None"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(best_escape_error=0.075))["trad"] == "Partial"
    assert st.verdict(_headline(best_escape_error=0.086))["trad"] == "Mirage"


def test_verdict_prose_separates_noise_from_drag():
    v = st.verdict(_headline())
    assert "noise, not drag" in v["trad_why"]
    assert "mean-zero" in v["trad_why"]
    assert "three funds" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
