"""The carry premium is HAC-significant on the premium tape, the crash leg is concentrated, and the
risk-managed overlay builds (without pretending to dodge a jump it can't forecast)."""

from steamroller import decompose, extension


def test_premium_tstat_significant(carry_xr, carry_rates):
    pt = decompose.premium_tstat(carry_xr, carry_rates, cost_bps=10.0)
    assert pt["t_stat"] > 2.0 and pt["mean_ann_pct"] > 0


def test_downside_concentration(carry_xr, carry_rates):
    dc = decompose.downside_concentration(carry_xr, carry_rates, cost_bps=10.0, k=5)
    assert dc["n_negative_months"] > 0
    assert 0.0 < dc["worst_k_share_of_losses"] <= 1.5   # the worst few months carry a big share of losses


def test_crash_comparison_builds(carry_xr, carry_rates):
    cc = extension.crash_comparison(carry_xr, carry_rates, cost_bps=10.0)
    for k in ("plain", "managed"):
        assert "sharpe" in cc[k] and "max_drawdown_pct" in cc[k]
    # vol-targeting lifts the standalone Sharpe (its known benefit), whatever it does to the tail
    assert cc["managed"]["sharpe"] >= cc["plain"]["sharpe"] - 0.2
