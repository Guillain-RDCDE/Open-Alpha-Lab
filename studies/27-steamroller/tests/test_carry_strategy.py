"""The carry premium and its negative-skew crash are recovered on the premium tape and absent on the
full-UIRP null; the bucket diagnostic sorts on the lagged rate; the real-tape reader is cache-only
offline (never a silent network call)."""

import numpy as np
import pandas as pd

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


def test_bucket_sort_uses_lagged_rate():
    """The bucket diagnostic must rank on the PRIOR month-end's rate (the book's information set).

    Build a panel where the month-t rate spikes together with the month-t return: an unlagged sort
    would 'discover' a huge premium from that contemporaneous association; the lagged sort must not."""
    idx = pd.period_range("2000-01", periods=120, freq="M").to_timestamp(how="end")
    rng = np.random.default_rng(7)
    n = 6
    rates = pd.DataFrame(rng.uniform(0.0, 8.0, size=(120, n)), index=idx,
                         columns=[f"C{i}" for i in range(n)])
    # the return of month t is driven by the rate REVEALED at the end of month t (not investable)
    xret = pd.DataFrame((rates.to_numpy() - rates.to_numpy().mean(axis=1, keepdims=True)) * 1e-3
                        + 1e-4 * rng.standard_normal((120, n)), index=idx, columns=rates.columns)
    pb = carry.carry_premium_by_bucket(xret, rates)
    # rates are i.i.d. across months, so the lagged sort sees no spread; unlagged would see ~+4.8%/yr
    assert abs(pb["hml_ann_pct"]) < 1.0


def test_carry_flat_on_null(null_xr, null_rates):
    pt = decompose.premium_tstat(null_xr, null_rates, cost_bps=10.0)
    assert abs(pt["t_stat"]) < 2.0        # full UIRP -> no premium


def test_negative_skew_steamroller(carry_xr, carry_rates):
    cr = decompose.crash_profile(carry_xr, carry_rates, cost_bps=10.0)
    assert cr["skew"] < -0.4              # the carry crash gives a fat negative tail
    assert cr["max_drawdown_pct"] < -10.0


def test_real_tape_reader_cache_only_offline():
    # offline: no cached parquets and no fetch -> empty (never a silent network call)
    out = data.fetch_carry(cache_dir="/nonexistent_cache_dir_xyz", fetch=False)
    assert out == {}
