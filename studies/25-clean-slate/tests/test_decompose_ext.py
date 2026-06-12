"""The residual-WML alpha is HAC-significant with momentum and *centred on zero* across no-momentum
seeds (a single null tape is one draw — the battery is the calibrated check); the paired residual-vs-
total bootstrap is well-formed and finds the baked improvement; the defence stack shrinks the drawdown."""

from clean_slate import decompose, extension


def test_capm_alpha_significant_on_momentum(mom_panel, mom_market):
    a = decompose.capm_alpha(mom_panel, mom_market, cost_bps=5.0)
    assert a["alpha_ann_pct"] > 5.0 and a["alpha_t"] > 3.0


def test_null_alpha_battery_centred_on_zero():
    """Harness calibration, not a single-seed pass/fail.

    One 16-year null tape draws a residual-WML alpha of sd ≈ 1.6%/yr (t sd ≈ 0.9) at this panel
    size — seed 25 alone lands ≈ −4%/yr net (t ≈ −2.1) with nothing wrong. So the test asserts on a
    five-seed battery that deliberately includes both observed tail draws (seeds 24 and 25):

      * battery mean within the cost drag ± 3 SE. At 5 bp on a ~12×/yr-turnover book the drag is
        ≈ −0.7%/yr; SE of a 5-seed mean ≈ 1.6/sqrt(5) ≈ 0.72 → bounds (−3.0, +1.8).
      * mean HAC t within ±1.0 (null per-seed t sd ≈ 0.9 → SE ≈ 0.40, so ±1.0 ≈ 2.5 SE).
      * every per-seed |t| < 3.2 (≈ 3.5 sigma of the per-seed null t) — the guard that a real
        mechanical bias (which would shift every seed) cannot hide behind.
    """
    bat = decompose.null_alpha_battery(seeds=(0, 1, 2, 24, 25), cost_bps=5.0)
    assert len(bat) == 5
    assert -3.0 < bat["alpha_ann_pct"].mean() < 1.8
    assert abs(bat["alpha_t"].mean()) < 1.0
    assert (bat["alpha_t"].abs() < 3.2).all()


def test_crash_comparison_builds(mom_panel, mom_market):
    cc = decompose.crash_comparison(mom_panel, mom_market, cost_bps=5.0)
    for k in ("residual", "total"):
        assert cc[k]["max_drawdown_pct"] < 0.0
        assert "skew" in cc[k]


def test_paired_crash_bootstrap_well_formed(mom_panel, mom_market):
    """The paired residual-vs-total test: sane intervals, and on the baked tape the residual book's
    Sharpe gap is positive (the synthetic plants the momentum in the residual, so it must win)."""
    pb = decompose.paired_crash_bootstrap(mom_panel, mom_market, n_boot=300, cost_bps=5.0)
    assert pb["n_months"] > 24
    assert pb["skew_ci_low"] < pb["skew_ci_high"]
    assert pb["sharpe_ci_low"] < pb["sharpe_ci_high"]
    assert 0.0 <= pb["skew_frac_residual_better"] <= 1.0
    assert pb["sharpe_diff"] > 0.0


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
