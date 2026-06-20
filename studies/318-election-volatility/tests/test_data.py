"""The election table is well-formed, the synthetic tape is deterministic and plants what
it claims, and the real loaders are cache-safe (skip when the shared cache is absent)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from election_volatility import data  # noqa: E402

# Cache-gating: the real loaders resolve the shared repo-root _cache. Guard any test that
# touches it so offline CI skips, while the synthetic core always runs.
GSPC_CACHE = os.path.join(data.SHARED_CACHE, "^GSPC_split_only.parquet")
VIX_CACHE = os.path.join(data.SHARED_CACHE, "^VIX_raw.parquet")


# ---- the election table ----------------------------------------------------
def test_election_table_well_formed():
    tbl = data.election_table()
    assert list(tbl.columns) == ["election_date", "label"]
    assert len(tbl) >= 24
    assert tbl["election_date"].is_monotonic_increasing
    assert tbl["election_date"].is_unique
    assert tbl["label"].str.len().gt(0).all()
    assert pd.api.types.is_datetime64_any_dtype(tbl["election_date"])


def test_election_dates_are_november_and_every_four_years():
    tbl = data.election_table()
    assert (tbl["election_date"].dt.month == 11).all()
    years = tbl["election_date"].dt.year.to_numpy()
    assert (np.diff(years) == 4).all()           # every fourth year
    assert (years % 4 == 0).all()                # election years divisible by 4


def test_election_dates_are_weekdays():
    tbl = data.election_table()
    assert (tbl["election_date"].dt.dayofweek < 5).all()


def test_min_year_filter():
    vix_era = data.election_table(min_year=1992)
    assert (vix_era["election_date"].dt.year >= 1992).all()
    assert len(vix_era) == 9


# ---- the synthetic tape ----------------------------------------------------
def test_synthetic_shape_and_ohlc(null_tape):
    bars, vix, el, truth = null_tape
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()
    assert list(el.columns) == ["election_date", "label"]
    assert len(el) == truth["n_elections"]
    assert (vix > 0).all()
    assert len(vix) == len(bars)


def test_synthetic_is_deterministic():
    a, va, ea, _ = data.synthetic_daily(vol_bump=0.5, vrp=5.0, seed=7)
    b, vb, eb, _ = data.synthetic_daily(vol_bump=0.5, vrp=5.0, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert np.allclose(va.to_numpy(), vb.to_numpy())
    assert (ea["election_date"].to_numpy() == eb["election_date"].to_numpy()).all()
    c, _, _, _ = data.synthetic_daily(vol_bump=0.5, vrp=5.0, seed=8)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_vol_bump_actually_raises_pre_election_vol():
    """A positive vol_bump must lift average pre-election realized vol above the null."""
    null, _, e0, _ = data.synthetic_daily(vol_bump=0.0, vrp=0.0, seed=318)
    spike, _, e1, t1 = data.synthetic_daily(vol_bump=0.8, vrp=0.0, seed=318)
    pre = t1["pre_window"]

    def pre_sd(bars, el):
        r = np.log(bars["close"]).diff().to_numpy()
        idx = bars.index
        sds = []
        for d in el["election_date"]:
            p = idx.searchsorted(d)
            sds.append(np.nanstd(r[p - pre:p], ddof=1))
        return np.nanmean(sds)

    assert pre_sd(spike, e1) > pre_sd(null, e0) * 1.3


def test_vrp_knob_adds_pre_election_implied_vol():
    """A positive vrp must lift the implied-vol proxy in the pre-election window."""
    _, v0, e0, t = data.synthetic_daily(vol_bump=0.0, vrp=0.0, seed=318)
    _, v1, e1, _ = data.synthetic_daily(vol_bump=0.0, vrp=8.0, seed=318)
    pre = t["pre_window"]
    idx = v0.index

    def pre_implied(v, el):
        out = []
        for d in el["election_date"]:
            p = idx.searchsorted(d)
            out.append(v.to_numpy()[p - pre:p].mean())
        return np.nanmean(out)

    assert pre_implied(v1, e1) > pre_implied(v0, e0) + 5.0


def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    bars, vix, _, _ = null_tape
    assert data.fingerprint(bars) == data.fingerprint(bars)
    assert data.fingerprint(vix) == data.fingerprint(vix)
    other, _, _, _ = data.synthetic_daily(vol_bump=0.0, vrp=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


# ---- real tapes: cache-gated, never skip the synthetic core -----------------
@pytest.mark.skipif(not os.path.exists(GSPC_CACHE), reason="offline CI: no ^GSPC cache")
def test_real_gspc_loads_and_is_clean():
    bars = data.load_real("^GSPC")
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert bars.index.is_monotonic_increasing
    assert bars.index.tz is None
    assert (bars["close"] > 0).all()


@pytest.mark.skipif(not os.path.exists(VIX_CACHE), reason="offline CI: no ^VIX cache")
def test_real_vix_loads_as_series():
    vix = data.load_vix()
    assert isinstance(vix, pd.Series)
    assert (vix > 0).all()
    assert vix.index.tz is None
