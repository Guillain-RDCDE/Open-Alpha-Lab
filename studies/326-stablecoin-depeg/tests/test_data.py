"""The synthetic tape and the depeg event table are well-formed and deterministic;
the real loader is cache-safe and skips offline."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stablecoin_depeg import data  # noqa: E402


def test_synthetic_shape_and_columns(shocked):
    bars, events, truth = shocked
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["close", "ret"]
    assert (bars["close"] > 0).all()
    assert bars.index.is_monotonic_increasing
    assert len(events) == truth["n_events"]


def test_synthetic_is_deterministic():
    a, ea, _ = data.synthetic_tape(n_days=400, n_events=4, shock=-0.1, seed=7)
    b, eb, _ = data.synthetic_tape(n_days=400, n_events=4, shock=-0.1, seed=7)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())
    assert list(ea) == list(eb)
    c, _, _ = data.synthetic_tape(n_days=400, n_events=4, shock=-0.1, seed=8)
    assert not np.allclose(a["ret"].to_numpy(), c["ret"].to_numpy())


def test_shock_makes_event_days_negative(shocked, null_tape):
    """Planted shock drives event-day returns sharply negative; the null does not."""
    sb, sev, _ = shocked
    nb, nev, _ = null_tape
    s_event = sb.loc[sev, "ret"].to_numpy()
    n_event = nb.loc[nev, "ret"].to_numpy()
    # Shocked event days average a large negative return; null event days ~0.
    assert s_event.mean() < -0.05
    assert abs(n_event.mean()) < 0.03


def test_null_event_days_not_special():
    """With shock=0 the event days are statistically ordinary (no abnormal return)."""
    bars, events, _ = data.synthetic_tape(n_events=10, shock=0.0, seed=326)
    ev_ret = bars.loc[events, "ret"].mean()
    all_ret = bars["ret"].mean()
    assert abs(ev_ret - all_ret) < 0.02


def test_depeg_events_major_count_is_two():
    """The headline real universe is exactly the two market-wide depegs."""
    major = data.depeg_events(major_only=True)
    allev = data.depeg_events(major_only=False)
    assert len(major) == 2
    assert len(allev) == 4
    assert pd.Timestamp("2022-05-09") in major
    assert pd.Timestamp("2023-03-11") in major


def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real("NOPE-USD", fetch=False, cache_dir=str(tmp_path))


def test_have_real_cache_false_on_empty_dir(tmp_path):
    assert data.have_real_cache(cache_dir=str(tmp_path)) is False


def test_fingerprint_stable_and_content_sensitive(shocked):
    bars, _, _ = shocked
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _, _ = data.synthetic_tape(n_events=10, shock=-0.10, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


# ---- cache-gated real-tape smoke test (skips offline) ----------------------
@pytest.mark.skipif(
    not data.have_real_cache(), reason="offline CI — no real BTC/ETH cache present"
)
def test_real_basket_well_formed():
    df = data.real_basket()
    assert "ret" in df.columns and "close" in df.columns
    assert (df["close"] > 0).all()
