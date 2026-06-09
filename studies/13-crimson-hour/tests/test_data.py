"""The synthetic tape is well-formed, deterministic, and offline; the real reducer is exercised
on a hand-built intraday frame so we never touch the network in CI."""

import numpy as np
import pandas as pd

from crimson_hour import data
from crimson_hour.data import FEATURE_COLUMNS


def test_panel_shape_and_columns(feat):
    assert list(feat.columns) == FEATURE_COLUMNS
    assert len(feat) == 520
    assert feat.index.name == "date"


def test_identities_hold(feat):
    """day_ret = oc_ret + rest_ret (in log space), and the red flags match the signs."""
    lhs = np.log1p(feat["day_ret"].to_numpy())
    rhs = np.log1p(feat["oc_ret"].to_numpy()) + np.log1p(feat["rest_ret"].to_numpy())
    assert np.allclose(lhs, rhs, atol=1e-9)
    assert (feat["oc_red"] == (feat["oc_ret"] < 0)).all()
    assert (feat["session_red"] == (feat["day_ret"] < 0)).all()
    assert (feat["rest_red"] == (feat["rest_ret"] < 0)).all()


def test_deterministic_given_seed():
    a, _ = data.synthetic_sessions(seed=13)
    b, _ = data.synthetic_sessions(seed=13)
    pd.testing.assert_frame_equal(a, b)


def test_ib_flag_is_boolean_and_mixed(feat):
    vals = set(feat["ib_high_rejected"].astype(bool).unique().tolist())
    assert vals <= {True, False}
    assert len(vals) == 2          # both rejection outcomes occur


def test_offline_fetch_is_cache_only(tmp_path):
    """Without fetch=True and with no cache, the reader returns empty — never hits the network."""
    out = data.fetch_intraday("ES=F", interval="5m", cache_dir=str(tmp_path), fetch=False)
    assert out.empty


def _toy_session(date, open_, prices_first_hour, close_):
    """Build one RTH day of 5-min bars from a list of first-hour closes + an afternoon close."""
    ts = pd.date_range(f"{date} 09:30", f"{date} 10:25", freq="5min", tz=data.NY_TZ)
    fh = pd.DataFrame({"Open": open_, "High": np.maximum(prices_first_hour, open_),
                       "Low": np.minimum(prices_first_hour, open_),
                       "Close": prices_first_hour}, index=ts)
    aft = pd.DataFrame({"Open": close_, "High": close_, "Low": close_, "Close": close_},
                       index=pd.DatetimeIndex([pd.Timestamp(f"{date} 15:55", tz=data.NY_TZ)]))
    return pd.concat([fh, aft])


def test_daily_features_reduces_intraday():
    """A red, high-before-low morning that then closes green is graded correctly."""
    # first hour: rises to a high early (bar 1), falls to a low late (bar 11), nets DOWN
    fh = np.array([100, 101.5, 101, 100.5, 100.2, 100.0, 99.8, 99.6, 99.4, 99.2, 99.0, 99.5])
    bars = _toy_session("2026-04-06", 100.0, fh, close_=100.8)
    out = data.daily_features(bars)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["session_open"] == 100.0
    assert row["oc_close"] == 99.5              # last first-hour close
    assert row["oc_red"]                         # 99.5 < 100.0
    assert row["ib_high_rejected"]               # high (bar 1) printed before low (bar 10)
    assert not row["session_red"]                # closes 100.8 > 100.0 (green day...)
    assert not row["rest_red"]                   # ...because the rest-of-day rallied
