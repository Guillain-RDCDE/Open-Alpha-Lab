"""The synthetic tapes are well-formed and deterministic; the real fetch is cache-safe;
and crucially the ``chase`` knob really plants the performance-chasing flow pattern."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ark_innovation import data  # noqa: E402

CACHE = data.DEFAULT_CACHE
ARKK_CACHE = data._cache_path("ARKK", CACHE)


def test_hype_cycle_shape(steady_grower):
    frame, truth = steady_grower
    assert len(frame) == truth["n_months"]
    assert list(frame.columns) == ["price", "ret", "flow"]
    assert (frame["price"] > 0).all()
    assert frame.index.is_monotonic_increasing


def test_hype_cycle_deterministic():
    a, _ = data.synthetic_hype_cycle(n_months=120, bust=0.04, chase=3.0, seed=7)
    b, _ = data.synthetic_hype_cycle(n_months=120, bust=0.04, chase=3.0, seed=7)
    assert np.allclose(a["price"].to_numpy(), b["price"].to_numpy())
    assert np.allclose(a["flow"].to_numpy(), b["flow"].to_numpy())
    c, _ = data.synthetic_hype_cycle(n_months=120, bust=0.04, chase=3.0, seed=8)
    assert not np.allclose(a["price"].to_numpy(), c["price"].to_numpy())


def test_chase_knob_makes_flows_chase_returns():
    """With chase>0, this period's flow correlates positively with the prior trailing
    12-month return; with chase=0 the flow is a flat baseline (no correlation). This is
    the flow engine that concentrates money near the peak."""
    def corr(frame):
        ret = frame["ret"].to_numpy()
        trail12 = np.convolve(ret, np.ones(12), mode="valid")  # rolling 12-sum
        f = frame["flow"].to_numpy()[12:]
        t = trail12[:-1]
        m = min(f.size, t.size)
        return np.corrcoef(f[:m], t[:m])[0, 1]
    chase_on, _ = data.synthetic_hype_cycle(n_months=180, bust=0.04, chase=15.0, seed=334)
    chase_off, _ = data.synthetic_hype_cycle(n_months=180, bust=0.04, chase=0.0, seed=334)
    assert corr(chase_on) > 0.3        # flows clearly chase performance
    assert abs(corr(chase_off)) < 0.25  # flat baseline → roughly return-independent


def test_bust_knob_creates_round_trip():
    """bust>0 plants a boom-bust hump: the price peaks well above its end value; bust=0 is
    a steady grower that ends near its high."""
    bb, truth = data.synthetic_hype_cycle(n_months=180, bust=0.04, chase=0.0, seed=334)
    sg, _ = data.synthetic_hype_cycle(n_months=180, bust=0.0, chase=0.0, seed=334)
    assert bb["price"].max() / bb["price"].iloc[-1] > 2.0   # round-tripped from the peak
    assert sg["price"].max() / sg["price"].iloc[-1] < 1.5   # ends near its high


def test_synthetic_daily_momentum_sign():
    """The momentum knob induces positive autocorrelation; 0 induces ~none."""
    flat = data.synthetic_daily(n_days=3000, momentum=0.0, seed=334)
    mom = data.synthetic_daily(n_days=3000, momentum=0.20, seed=334)
    ac = lambda s: np.corrcoef(s[:-1], s[1:])[0, 1]
    r_flat = np.diff(np.log(flat["close"].to_numpy()))
    r_mom = np.diff(np.log(mom["close"].to_numpy()))
    assert abs(ac(r_flat)) < 0.08
    assert ac(r_mom) > 0.10


def test_fetch_prices_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_prices("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fetch_flows_returns_none_without_cache(tmp_path):
    assert data.fetch_flows(cache_dir=str(tmp_path)) is None


def test_fingerprint_stable_and_sensitive(steady_grower):
    frame, _ = steady_grower
    assert data.fingerprint(frame, "price") == data.fingerprint(frame, "price")
    other, _ = data.synthetic_hype_cycle(n_months=180, bust=0.0, chase=0.0, seed=99)
    assert data.fingerprint(frame, "price") != data.fingerprint(other, "price")


@pytest.mark.skipif(not os.path.exists(ARKK_CACHE), reason="offline CI — no real cache")
def test_real_arkk_cache_loads_clean():
    px = data.fetch_prices("ARKK", fetch=False)
    assert "close" in px.columns
    assert px.index.tz is None
    assert (px["close"] > 0).all()
