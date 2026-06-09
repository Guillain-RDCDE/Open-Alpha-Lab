"""Beat-7 worked complement — the jump-robust vol estimator and the rescue experiment.

The base study found the article's rolling-vol "production fix" collapses under jumps. The
complement asks whether a jump-robust (bipower) vol estimate rescues it. These tests pin the
two facts the write-up rests on: (1) bipower variation really does ignore a jump the naive
estimator inhales, and (2) it *mitigates but does not rescue* — better than naive, still far
below plain fixed-sigma AS.
"""

import numpy as np

from phantom_kernel import experiments as ex, sim
from phantom_kernel.strategies import AdaptiveASQuoter, JumpRobustASQuoter


def test_bipower_vol_ignores_a_jump_the_naive_estimator_inhales():
    rng = np.random.default_rng(0)
    calm = 100.0 + np.cumsum(0.01 * rng.standard_normal(300))
    naive = AdaptiveASQuoter(0.1, 0.4, 0.6, 1.0, vol_window=300, dt=0.01)
    robust = JumpRobustASQuoter(0.1, 0.4, 0.6, 1.0, vol_window=300, dt=0.01)
    for p in calm:
        naive.observe(float(p))
        robust.observe(float(p))
    # Inject one big jump and compare the resulting vol estimates.
    jump_price = float(calm[-1]) + 5.0
    naive.observe(jump_price)
    robust.observe(jump_price)
    assert robust.sigma < 0.5 * naive.sigma          # the jump barely moves the robust estimate


def test_rescue_is_partial_not_a_rescue():
    """Jump-robust beats naive (mitigation) but stays well below fixed-sigma AS (no rescue)."""
    t = ex.extension_rescue(sim.WORLD_B, n_steps=30_000, seed=0)
    assert list(t.index) == ["AS (fixed sigma)", "AS (naive RV)", "AS (jump-robust BV)"]
    naive = t.loc["AS (naive RV)", "pnl_sharpe"]
    robust = t.loc["AS (jump-robust BV)", "pnl_sharpe"]
    fixed = t.loc["AS (fixed sigma)", "pnl_sharpe"]
    assert robust > naive                            # mitigation: it helps
    assert robust < 0.6 * fixed                      # but the fix is still broken


def test_estimator_choice_is_a_no_op_without_jumps():
    """In the calm World A (no jumps) naive and jump-robust must land in the same place."""
    t = ex.extension_rescue(sim.WORLD_A, n_steps=30_000, seed=0)
    naive = t.loc["AS (naive RV)", "pnl_sharpe"]
    robust = t.loc["AS (jump-robust BV)", "pnl_sharpe"]
    assert abs(naive - robust) < 0.15
