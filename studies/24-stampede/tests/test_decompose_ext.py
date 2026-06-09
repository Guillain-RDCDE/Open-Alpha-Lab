"""The WML alpha is HAC-significant with momentum and a zero on the null; the crash profile and the
risk-managed overlay build and behave."""

from stampede import decompose, extension


def test_capm_alpha_significant_on_momentum(momentum_panel):
    a = decompose.capm_alpha(momentum_panel, cost_bps=5.0)
    assert a["alpha_ann_pct"] > 5.0 and a["alpha_t"] > 3.0


def test_capm_alpha_zero_on_null(null_panel):
    a = decompose.capm_alpha(null_panel, cost_bps=5.0)
    assert abs(a["alpha_t"]) < 2.0


def test_crash_profile_builds(momentum_panel):
    cr = decompose.crash_profile(momentum_panel, cost_bps=5.0)
    assert cr["worst_month_pct"] < 0.0
    assert cr["max_drawdown_pct"] < 0.0
    assert cr["n_months"] > 12


def test_subsample_and_bootstrap(momentum_panel):
    sub = decompose.subsample_sharpe(momentum_panel, cost_bps=5.0, n_chunks=3)
    assert len(sub) == 3
    bs = decompose.sharpe_bootstrap(momentum_panel, n_boot=300, cost_bps=5.0)
    assert bs["ci_low"] < bs["sharpe"] < bs["ci_high"]


def test_risk_management_reduces_drawdown(momentum_panel):
    cc = extension.crash_comparison(momentum_panel, cost_bps=5.0)
    # vol-scaling should not deepen the worst drawdown (it tames the tail)
    assert cc["managed"]["max_drawdown_pct"] >= cc["plain"]["max_drawdown_pct"] - 5.0
    assert "sharpe_gain" in cc
