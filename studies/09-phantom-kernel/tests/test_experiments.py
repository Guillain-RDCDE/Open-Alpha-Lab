"""The teardown functions must return the right shapes and the signs the verdict rests on."""

from phantom_kernel import experiments as ex, sim


def test_estimator_recovery_is_accurate():
    out = ex.estimator_recovery(n_orders=200_000, seed=0)
    assert abs(out["rel_error_pct"]) < 3.0
    assert out["r2"] > 0.99


def test_kernel_gof_table_winners():
    tab = ex.kernel_gof_table(n_orders=200_000, seed=0)
    assert tab.loc[sim.WORLD_A.name, "winner"] == "exponential"
    assert tab.loc[sim.WORLD_B.name, "winner"] == "power-law"
    # The labelled stress row (alpha=1.2) must agree, and the MLE must see World B's exponent.
    assert tab.loc[sim.WORLD_B_STRESS.name, "winner"] == "power-law"
    assert abs(float(tab.loc[sim.WORLD_B.name, "alpha_mle"]) - sim.WORLD_B.pareto_alpha) < 0.05


def test_k_instability_reports_material_spread_error_at_both_horizons():
    out = ex.k_instability(seed=0)
    assert out["k_true_ratio"] == 4.0
    # Headline: the tournament's own horizon. Bound: T=1, the k-term-only worst case.
    assert out["max_abs_spread_pct_error"] > 20.0
    assert out["bound_T1"]["max_abs_spread_pct_error"] > 100.0
    assert out["max_abs_spread_pct_error"] < out["bound_T1"]["max_abs_spread_pct_error"]


def test_k_ablation_fitted_phantom_buys_nothing():
    """The MISATTRIBUTED ablation, at the study's own session length: swapping the phantom k
    fitted on World B's fills for the uncalibrated textbook 0.6 must not cost AS anything —
    the calibration machinery contributes nothing to the World-B win."""
    ab = ex.k_ablation(sim.WORLD_B, seeds=(0,), n_steps=60_000)
    s_cols = [c for c in ab.columns if c.startswith("sharpe")]
    assert len(s_cols) == 2
    textbook = float(ab.loc["mean", [c for c in s_cols if "textbook" in c][0]])
    fitted = float(ab.loc["mean", [c for c in s_cols if "fitted" in c][0]])
    assert textbook > 0 and fitted > 0
    # The fitted phantom never beats the textbook value materially, and at this horizon the
    # two are indistinguishable (the docs' 5-seed means differ by < 1%).
    assert fitted < textbook * 1.10
    assert abs(textbook - fitted) < 0.25 * max(textbook, fitted)


def test_tournament_shape_and_inventory_control():
    t = ex.tournament(sim.WORLD_A, n_steps=20_000, seed=0)
    assert list(t.index) == ["AS (fixed)", "AS (adaptive vol)", "Symmetric (no skew)", "Inventory clamp"]
    # AS holds far less inventory than the skew-free quoter (its whole purpose).
    assert t.loc["AS (fixed)", "inv_std"] < t.loc["Symmetric (no skew)", "inv_std"]


def test_tournament_as_wins_in_friction_world():
    """When inventory is dangerous (jumps + informed flow), AS's tight control pays."""
    t = ex.tournament(sim.WORLD_B, n_steps=20_000, seed=0)
    assert t.loc["AS (fixed)", "pnl_sharpe"] > t.loc["Inventory clamp", "pnl_sharpe"]
    assert t.loc["AS (fixed)", "n_adverse"] > 0          # adverse selection really bites
