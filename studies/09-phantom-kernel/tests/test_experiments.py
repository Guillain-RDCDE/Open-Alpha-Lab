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


def test_k_instability_reports_large_spread_error():
    out = ex.k_instability(seed=0)
    assert out["k_true_ratio"] == 4.0
    assert out["max_abs_spread_pct_error"] > 50.0


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
