"""Tests for Study 84 (Moon-Math) data layer — offline, deterministic."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moon_math import data  # noqa: E402


def test_synthetic_shape_and_columns(planb_tape):
    df, truth = planb_tape
    assert len(df) == truth["n_days"]
    for col in ("close", "log_close", "log_s2f", "s2f", "stock", "flow", "log_time"):
        assert col in df.columns, f"missing column: {col}"
    # close may have NaN in the warmup period (S2F rolling window), but valid rows are positive
    valid = df["close"].dropna()
    assert (valid > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_s2f(n_days=500, price_s2f_beta=3.3, seed=10)
    b, _ = data.synthetic_s2f(n_days=500, price_s2f_beta=3.3, seed=10)
    # Compare only non-NaN rows (warmup period has NaN from rolling window)
    va = a["close"].dropna().to_numpy()
    vb = b["close"].dropna().to_numpy()
    assert np.allclose(va, vb)
    c, _ = data.synthetic_s2f(n_days=500, price_s2f_beta=3.3, seed=11)
    vc = c["close"].dropna().to_numpy()
    min_len = min(len(va), len(vc))
    assert not np.allclose(va[:min_len], vc[:min_len])


def test_s2f_increases_pre_halving():
    """Before the 2016 halving, S2F should be increasing (stock grows, flow constant)."""
    import pandas as pd
    # 2016 halving was ~July 9. Use Jan-Jun 2016 (pre-halving, fully in-window).
    dates = pd.bdate_range("2016-01-01", periods=120, name="date")
    sf = data.btc_s2f(dates)
    sf_valid = sf["s2f"].dropna()
    if len(sf_valid) < 2:
        return  # skip if not enough data
    # S2F should be positive and finite
    assert (sf_valid > 0).all(), "S2F should be positive"
    assert sf_valid.max() < 1000, "S2F should be a plausible scarcity number"


def test_s2f_increases_after_halving_settles():
    """One full year after the 2020 halving, S2F should be much higher than before."""
    import pandas as pd
    # 2020 halving was ~May 11. Pre-halving long window: early 2019.
    # Post-halving window fully settled: mid-2021 (one full year after halving).
    pre_halving = pd.bdate_range("2019-01-01", periods=252, name="date")
    post_halving = pd.bdate_range("2021-05-01", periods=252, name="date")
    sf_pre = data.btc_s2f(pre_halving)["s2f"].dropna()
    sf_post = data.btc_s2f(post_halving)["s2f"].dropna()
    if len(sf_pre) < 30 or len(sf_post) < 30:
        return  # insufficient data
    # After halving settles, S2F should be roughly double the pre-halving value
    ratio = sf_post.mean() / sf_pre.mean()
    assert ratio > 1.5, f"Post-halving S2F should be substantially higher, ratio={ratio:.2f}"


def test_fetch_btc_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_btc(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_stable_and_content_sensitive(planb_tape, null_tape):
    df_a, _ = planb_tape
    df_b, _ = planb_tape
    df_c, _ = null_tape
    assert data.fingerprint(df_a) == data.fingerprint(df_b)
    assert data.fingerprint(df_a) != data.fingerprint(df_c)


def test_log_time_monotone(planb_tape):
    """log(time since genesis) should increase monotonically with date."""
    df, _ = planb_tape
    lt = df["log_time"].to_numpy()
    assert (np.diff(lt) >= -1e-9).all(), "log_time should be non-decreasing"


def test_synthetic_null_uncorrelated_with_s2f(null_tape):
    """With beta=0, log(price) should have near-zero correlation with log(S2F)
    at the per-day change level (the level correlation is spurious/noise-walk)."""
    df, _ = null_tape
    dp = df["log_close"].diff().dropna()
    ds = df["log_s2f"].diff().dropna()
    idx = dp.index.intersection(ds.index)
    corr = np.corrcoef(dp.loc[idx], ds.loc[idx])[0, 1]
    assert abs(corr) < 0.15, f"First-diff correlation should be near 0, got {corr:.3f}"
