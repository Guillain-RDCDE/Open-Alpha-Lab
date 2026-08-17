"""Data-layer tests — the event list, the synthetic panel, and a skipif cache smoke test.

Everything here is offline. The real-cache test is skipped cleanly when the shared
``studies/_cache`` is absent, so a fresh checkout with no parquet files is still green.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rights_offering import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The event list is well-formed
# --------------------------------------------------------------------------- #
def test_event_list_shape_and_fields():
    assert len(data.RIGHTS_EVENTS) >= 30
    bands = {"deep", "moderate", "shallow"}
    kinds = {"cef", "bdc", "reit", "opco"}
    for tk, ann, prec, band, kind in data.RIGHTS_EVENTS:
        assert tk.isupper() and 1 <= len(tk) <= 5
        assert pd.Timestamp(ann) < pd.Timestamp(data.AS_OF)
        assert prec in {"day", "month"}
        assert band in bands
        assert kind in kinds


def test_event_list_has_no_exact_duplicates():
    keys = [(tk, ann) for tk, ann, _, _, _ in data.RIGHTS_EVENTS]
    assert len(keys) == len(set(keys))


def test_events_frame_timetable_is_ordered():
    ef = data.events_frame()
    assert (ef["announce"] < ef["ex_rights"]).all()
    assert (ef["ex_rights"] < ef["expiry"]).all()
    # a different assumed timetable moves the modelled dates, as it must
    ef2 = data.events_frame(timetable={"ex_rights_days": 5, "expiry_days": 60})
    assert (ef2["expiry"] > ef["expiry"]).all()


def test_tickers_cover_the_event_list():
    listed = {e[0] for e in data.RIGHTS_EVENTS}
    assert listed.issubset(set(data.TICKERS))
    assert data.MARKET in data.TICKERS and data.CASH in data.TICKERS


def test_survivorship_dropouts_are_named():
    """The deals with no retrievable tape must stay documented, not silently vanish."""
    assert len(data.DROPPED_NO_TAPE) >= 3
    listed = {e[0] for e in data.RIGHTS_EVENTS}
    assert not (set(data.DROPPED_NO_TAPE) & listed)


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, ea, _ = data.synthetic_panel(seed=929)
    b, eb, _ = data.synthetic_panel(seed=929)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert ea == eb


def test_synthetic_shape_and_columns():
    px, ev, truth = data.synthetic_panel(n_names=6, n_events_per_name=2, n_years=8, seed=929)
    assert data.MARKET in px.columns and data.CASH in px.columns
    assert isinstance(px.index, pd.DatetimeIndex)
    assert len(px) == truth["n_days"] == 8 * data.TRADING_DAYS_PER_YEAR
    assert 1 <= truth["n_events"] <= 12
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert px.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_daily_is_single_name():
    px, ev, truth = data.synthetic_daily(seed=929)
    assert truth["n_names"] == 1
    assert {e[0] for e in ev} == {"N00"}


def test_synthetic_cash_is_monotone_growing():
    px, _, _ = data.synthetic_panel(seed=929)
    assert (px[data.CASH].diff().dropna() > 0).all()
    assert px[data.CASH].iloc[-1] > px[data.CASH].iloc[0]


def test_signal_strength_zero_removes_the_effect():
    """At ss=0 the panel is the same noise as ss=1 minus the planted event moves."""
    pa, ea, _ = data.synthetic_panel(signal_strength=1.0, seed=929)
    pb, eb, _ = data.synthetic_panel(signal_strength=0.0, seed=929)
    assert ea == eb                       # same anchors, so the null is matched
    assert not np.allclose(pa["N00"].to_numpy(), pb["N00"].to_numpy())


def test_fingerprint_stable_and_sensitive():
    a, _, _ = data.synthetic_panel(seed=929)
    b, _, _ = data.synthetic_panel(seed=930)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_is_false_on_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared studies/_cache present (offline / CI) — "
                           "the synthetic tests cover the logic")
def test_real_cache_event_study_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    panel = st.event_panel(px, data.RIGHTS_EVENTS)
    assert len(panel) >= 25
    for w in st.WINDOWS:
        assert np.isfinite(panel[f"car_{w}"]).all()
