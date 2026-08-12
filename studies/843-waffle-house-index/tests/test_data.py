"""The disaster table is sane; the synthetic world is well-formed, deterministic and
carries the planted drift only when asked; the real loader is cache-gated (skips
cleanly with no cache present)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waffle_index import data  # noqa: E402


def test_disaster_table_is_sane():
    ev = data.disaster_table()
    assert list(ev.columns) == ["date", "label", "loss_tier"]
    assert len(ev) >= 12
    assert ev["date"].is_monotonic_increasing
    assert ev["date"].dt.tz is None
    labels = " ".join(ev["label"])
    for name in ("Katrina", "Sandy", "Harvey", "Irma", "Ian", "Helene"):
        assert name in labels
    assert set(ev["loss_tier"]).issubset({1, 2, 3})


def test_disaster_table_asof_truncates():
    ev = data.disaster_table(asof="2018-01-01")
    assert (ev["date"] <= pd.Timestamp("2018-01-01")).all()
    assert not ev["label"].str.contains("Ian").any()      # 2022 dropped
    assert ev["label"].str.contains("Katrina").any()       # 2005 kept


def test_ticker_universe_shape():
    assert data.BENCHMARK == "SPY"
    assert set(data.INSURERS) == {"ALL", "TRV", "PGR"}
    assert set(data.REBUILDERS) == {"HD", "LOW"}
    assert data.TICKERS[0] == "SPY" and len(data.TICKERS) == 6


def test_synthetic_shape_and_schema(null_world):
    closes, events = null_world
    assert set(closes) == set(data.TICKERS)
    for t, s in closes.items():
        assert isinstance(s, pd.Series)
        assert (s > 0).all()
        assert s.index.is_monotonic_increasing
        assert s.index.tz is None
    assert len(events) == 16
    assert events.isin(closes["SPY"].index).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_world(edge=0.001, seed=7)
    b, _ = data.synthetic_world(edge=0.001, seed=7)
    assert np.allclose(a["ALL"].to_numpy(), b["ALL"].to_numpy())
    c, _ = data.synthetic_world(edge=0.001, seed=8)
    assert not np.allclose(a["ALL"].to_numpy(), c["ALL"].to_numpy())


def test_edge_pushes_insurers_down_rebuilders_up():
    """The edge knob really depresses insurers' and lifts rebuilders' market-adjusted
    CAR around planted events; the null leaves both near zero."""
    from waffle_index import strategy as st
    flat, ef = data.synthetic_world(edge=0.0, seed=843)
    up, eu = data.synthetic_world(edge=0.0015, seed=843)
    for world, ev, expect_ins_down in ((flat, ef, False), (up, eu, True)):
        spy = world["SPY"]
        ins = st.car_stats(st.basket_ar(world, data.INSURERS, spy), ev, 10, 20, 0, 20)["mean"]
        reb = st.car_stats(st.basket_ar(world, data.REBUILDERS, spy), ev, 10, 20, 0, 20)["mean"]
        if expect_ins_down:
            assert ins < -0.005 and reb > 0.005    # clearly separated
        else:
            assert abs(ins) < 0.01 and abs(reb) < 0.01


def test_fingerprint_stable_and_content_sensitive(null_world):
    closes, _ = null_world
    s = closes["SPY"]
    assert data.fingerprint(s) == data.fingerprint(s)
    other, _ = data.synthetic_world(edge=0.0, seed=99)
    assert data.fingerprint(s) != data.fingerprint(other["SPY"])


def test_load_real_raises_without_cache(tmp_path, monkeypatch):
    """With the cache pointed at an empty dir, the loader raises (offline-safe)."""
    monkeypatch.setattr(data, "CACHE",
                        {t: str(tmp_path / f"whi_{t.lower()}.csv") for t in data.TICKERS})
    with pytest.raises(FileNotFoundError):
        data.load_real()


# --- cache-gated real-tape smoke test (skips offline) ----------------------
@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_load_real_cache_hit_well_formed():
    closes = data.load_real()
    assert set(closes) == set(data.TICKERS)
    for t, s in closes.items():
        assert isinstance(s, pd.Series)
        assert s.index.tz is None
        assert (s > 0).all()
        assert s.index.max() <= pd.Timestamp(data.AS_OF)
