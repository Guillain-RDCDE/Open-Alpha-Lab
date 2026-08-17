"""Data-layer tests — the event list, synthetic determinism, and a skipif cache smoke test.

Everything here runs offline. The real-cache test is skipped entirely when the shared
``studies/_cache`` is absent, so a fresh checkout with no parquet files is green.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creation_halt import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The hardcoded event list
# --------------------------------------------------------------------------- #
REQUIRED_FIELDS = {
    "key", "fund", "proxy", "halt", "resume", "direction", "confidence",
    "announcement", "ruler", "kind", "proxy_kind", "note",
}


def test_events_have_every_field():
    assert len(data.EVENTS) >= 5
    for ev in data.EVENTS:
        assert REQUIRED_FIELDS.issubset(ev), f"{ev['key']} is missing fields"


def test_event_keys_unique_and_lookup_works():
    keys = [e["key"] for e in data.EVENTS]
    assert len(keys) == len(set(keys))
    assert data.event(keys[0])["key"] == keys[0]
    with pytest.raises(KeyError):
        data.event("NO-SUCH-EVENT")


def test_event_dates_are_ordered_and_not_in_the_future():
    asof = pd.Timestamp(data.AS_OF)
    for ev in data.EVENTS:
        halt, resume = pd.Timestamp(ev["halt"]), pd.Timestamp(ev["resume"])
        assert halt < resume, f"{ev['key']}: suspension must precede resumption"
        assert resume <= asof, f"{ev['key']}: resumption must not be after the as-of"


def test_event_direction_and_tags_are_legal():
    for ev in data.EVENTS:
        assert ev["direction"] in (+1, -1)
        assert ev["confidence"] in ("FIRM", "APPROX", "SOFT")
        assert ev["ruler"] in ("exact", "curve-mismatched")
        assert isinstance(ev["announcement"], bool)


def test_redemption_halts_are_signed_negative():
    """A suspension of *redemptions* must carry direction −1 (the price can only cheapen)."""
    for ev in data.EVENTS:
        if "redemption suspended" in ev["kind"]:
            assert ev["direction"] == -1


# --------------------------------------------------------------------------- #
# Synthetic generators
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=918)
    b, _ = data.synthetic_daily(seed=918)
    assert np.allclose(a["fund"].to_numpy(), b["fund"].to_numpy())
    assert np.allclose(a["proxy"].to_numpy(), b["proxy"].to_numpy())


def test_synthetic_shape_and_index_is_oob_safe():
    frame, truth = data.synthetic_daily(n_days=1200, seed=918)
    assert {"fund", "proxy", "premium"}.issubset(frame.columns)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert len(frame) == truth["n_days"] == 1200
    assert frame.index[-1] < pd.Timestamp("2262-01-01")


def test_planted_premium_accretes_then_fades():
    frame, truth = data.synthetic_daily(seed=918)
    prem = frame["premium"]
    assert truth["planted_total_premium"] > 0
    assert prem.max() == pytest.approx(truth["planted_total_premium"], rel=1e-9)
    assert prem.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_signal_strength_zero_plants_nothing():
    frame, truth = data.synthetic_daily(signal_strength=0.0, seed=918)
    assert truth["planted_total_premium"] == 0.0
    assert float(frame["premium"].abs().max()) == 0.0


def test_synthetic_panel_shapes_and_event_schema():
    frames, evs = data.synthetic_panel(n_events=4, seed=918)
    assert len(frames) == len(evs) == 4
    for ev in evs:
        assert {"key", "fund", "proxy", "halt", "resume", "direction"}.issubset(ev)
        assert pd.Timestamp(ev["halt"]) < pd.Timestamp(ev["resume"])
        assert ev["key"] in frames


def test_synthetic_panel_events_do_not_all_share_a_halt_date():
    _, evs = data.synthetic_panel(n_events=4, seed=918)
    assert len({e["halt"] for e in evs}) == 4


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=918)
    b, _ = data.synthetic_daily(seed=919)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_spreads_build_and_cover_every_event():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    spreads = st.build_spreads(px)
    for ev in data.EVENTS:
        s = spreads[ev["key"]]
        assert len(s) > 250
        assert s.index[0] <= pd.Timestamp(ev["resume"])
        assert np.isfinite(s.to_numpy()).all()
