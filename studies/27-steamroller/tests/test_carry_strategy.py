"""The carry premium and its negative-skew crash are recovered on the premium tape and absent on the
full-UIRP null; the FRED reader is cache-only offline."""

import numpy as np

from steamroller import data, carry, strategy, decompose


def test_deterministic_and_premium(carry):
    xr, rates, truth = carry
    xr2, _, _ = data.synthetic_carry(carry_strength=0.9, seed=27)
    assert np.allclose(xr.to_numpy(), xr2.to_numpy())
    assert truth.has_premium


def test_premium_by_bucket_recovers(carry_xr, carry_rates):
    pb = carry.carry_premium_by_bucket(carry_xr, carry_rates)
    assert pb["hml_ann_pct"] > 0          # high-rate currencies out-earn low-rate ones
    assert pb["premium_present"]


def test_carry_pays_with_premium(carry_xr, carry_rates):
    cmp = strategy.compare(carry_xr, carry_rates, cost_bps=10.0)
    assert cmp["sharpe"] > 0.2
    assert cmp["turnover_ann"] < 6.0      # rates move slowly -> low turnover


def test_carry_flat_on_null(null_xr, null_rates):
    pt = decompose.premium_tstat(null_xr, null_rates, cost_bps=10.0)
    assert abs(pt["t_stat"]) < 2.0        # full UIRP -> no premium


def test_negative_skew_steamroller(carry_xr, carry_rates):
    cr = decompose.crash_profile(carry_xr, carry_rates, cost_bps=10.0)
    assert cr["skew"] < -0.4              # the carry crash gives a fat negative tail
    assert cr["max_drawdown_pct"] < -10.0


def test_fred_reader_cache_only_offline():
    # offline: no cached file and no fetch -> empty (never a silent network stall)
    out = data.fetch_carry(cache_dir="/nonexistent_cache_dir_xyz", fetch=False)
    assert out == {}
