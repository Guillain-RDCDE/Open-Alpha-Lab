"""Strategy invariants and the synthetic positive control for both effect legs.

The spine: with a planted vol premium the range detector finds it; with a planted
return premium the return test finds it; on the null tape neither does (or at most
weakly, at noise level).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from witching_hour import data, strategy as st  # noqa: E402


# ---- measure helpers -------------------------------------------------------

def test_daily_range_positive(null_tape):
    bars, _ = null_tape
    rng = st.daily_range_pct(bars)
    assert (rng > 0).all()


def test_log_volume_finite(null_tape):
    bars, _ = null_tape
    lvol = st.log_volume(bars)
    assert np.isfinite(lvol.to_numpy()).all()


def test_daily_return_length(null_tape):
    bars, _ = null_tape
    ret = st.daily_return(bars)
    # pct_change drops the first row
    assert len(ret.dropna()) == len(bars) - 1


# ---- compare_witching_vs_baseline ------------------------------------------

def test_compare_returns_correct_keys(null_tape):
    bars, _ = null_tape
    import pandas as pd
    rng = st.daily_range_pct(bars)
    idx = pd.DatetimeIndex(bars.index)
    mask = data.witching_day_mask(idx)
    result = st.compare_witching_vs_baseline(rng, mask)
    expected = {"label", "n_witching", "n_baseline", "mean_witching", "mean_baseline",
                "std_witching", "std_baseline", "diff", "tstat"}
    assert expected == set(result.keys())


def test_compare_counts_sum_to_total(null_tape):
    bars, _ = null_tape
    import pandas as pd
    rng = st.daily_range_pct(bars)
    idx = pd.DatetimeIndex(bars.index)
    mask = data.witching_day_mask(idx)
    result = st.compare_witching_vs_baseline(rng, mask)
    assert result["n_witching"] + result["n_baseline"] == len(rng)


# ---- vol/volume leg: planted effect is detectable -------------------------

def test_vol_premium_detectable(vol_tape, null_tape):
    """With a planted range premium, the HAC t-stat on range should be larger."""
    import pandas as pd

    def range_tstat(bars):
        rng = st.daily_range_pct(bars)
        idx = pd.DatetimeIndex(bars.index)
        mask = data.witching_day_mask(idx)
        return st.compare_witching_vs_baseline(rng, mask)["tstat"]

    t_vol = range_tstat(vol_tape[0])
    t_null = range_tstat(null_tape[0])
    # The planted-vol tape should produce a larger (positive) t-stat
    assert t_vol > t_null
    assert t_vol > 2.0, f"planted vol premium not detected: t={t_vol:.2f}"


def test_null_range_tstat_small(null_tape):
    """On the null tape the range t-stat should be small (vol premium absent)."""
    import pandas as pd
    bars, _ = null_tape
    rng = st.daily_range_pct(bars)
    idx = pd.DatetimeIndex(bars.index)
    mask = data.witching_day_mask(idx)
    t = st.compare_witching_vs_baseline(rng, mask)["tstat"]
    assert abs(t) < 2.5, f"null tape has unexpected range t={t:.2f}"


# ---- return leg: planted return premium is detectable ----------------------

def test_return_premium_detectable(null_tape):
    """With a planted return premium, the HAC t-stat on returns should be larger than null."""
    # Use a larger premium and more years to make it clearly detectable
    bars_ret, _ = data.synthetic_daily(n_years=30, vol_premium=0.0, ret_premium_bps=20.0, seed=82)

    def ret_tstat(bars):
        res = st.witching_return_test(bars)
        return res["day"]["tstat"]

    t_ret = ret_tstat(bars_ret)
    t_null = ret_tstat(null_tape[0])
    # The planted premium should produce a clearly larger t-stat
    assert t_ret > t_null
    assert t_ret > 1.5, f"planted return premium not detected: t={t_ret:.2f}"


def test_null_return_tstat_small(null_tape):
    bars, _ = null_tape
    res = st.witching_return_test(bars)
    t = res["day"]["tstat"]
    assert abs(t) < 2.5, f"null tape has unexpected return t={t:.2f}"


# ---- summarize -------------------------------------------------------------

def test_summarize_keys(null_tape):
    bars, _ = null_tape
    ret = st.daily_return(bars).dropna()
    out = st.summarize(ret)
    assert set(out.keys()) == {"n", "mean_bps", "sharpe_ann", "cagr", "tstat"}
    assert out["n"] == len(ret)


def test_summarize_zero_mean():
    """A zero-mean series should produce a t-stat near zero."""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 500))
    out = st.summarize(r)
    assert abs(out["tstat"]) < 3.0
