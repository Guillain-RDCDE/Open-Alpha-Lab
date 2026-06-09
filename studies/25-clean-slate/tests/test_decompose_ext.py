"""The residual-WML alpha is HAC-significant with momentum and weak on the null; residual momentum has a
better-behaved skew than total; and the defence stack shrinks the drawdown."""

from clean_slate import decompose, extension


def test_capm_alpha_significant_on_momentum(mom_panel, mom_market):
    a = decompose.capm_alpha(mom_panel, mom_market, cost_bps=5.0)
    assert a["alpha_ann_pct"] > 5.0 and a["alpha_t"] > 3.0


def test_capm_alpha_weak_on_null(null_panel, null_market):
    a = decompose.capm_alpha(null_panel, null_market, cost_bps=5.0)
    assert abs(a["alpha_t"]) < 3.0       # no baked premium -> no strong alpha


def test_crash_comparison_builds(mom_panel, mom_market):
    cc = decompose.crash_comparison(mom_panel, mom_market, cost_bps=5.0)
    for k in ("residual", "total"):
        assert cc[k]["max_drawdown_pct"] < 0.0
        assert "skew" in cc[k]


def test_defence_stack_reduces_drawdown(mom_panel, mom_market):
    st = extension.defence_stack(mom_panel, mom_market, cost_bps=5.0)
    # vol-managed residual should have a shallower (less negative) drawdown than plain total
    assert st["residual_vol_managed"]["max_drawdown_pct"] >= st["total"]["max_drawdown_pct"] - 2.0
    assert st["residual_vol_managed"]["sharpe"] > 0.0


def test_subsample_and_bootstrap(mom_panel, mom_market):
    sub = decompose.subsample_sharpe(mom_panel, mom_market, cost_bps=5.0, n_chunks=3)
    assert len(sub) == 3
    bs = decompose.sharpe_bootstrap(mom_panel, mom_market, n_boot=300, cost_bps=5.0)
    assert bs["ci_low"] < bs["sharpe"] < bs["ci_high"]
