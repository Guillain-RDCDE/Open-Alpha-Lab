"""The synthetic world is deterministic; the leveraged series is L times the underlying daily; the
volatility decay matches theory when vol is present and vanishes when it isn't. All offline."""
import numpy as np
from slow_burn import data, strategy as st


def test_world_deterministic(volatile_world):
    s, truth = volatile_world
    s2, _ = data.synthetic_underlying(vol_ann=0.20, seed=61)
    assert np.allclose(s.to_numpy(), s2.to_numpy())
    assert truth.has_vol


def test_lever_daily_is_L_times(volatile_world):
    s, _ = volatile_world
    assert np.allclose(st.lever_daily(s, 3.0).to_numpy(), 3.0 * s.to_numpy())


def test_decay_matches_theory_when_volatile(volatile_world):
    s, _ = volatile_world
    g = st.decay_gap(s, L=3.0)
    assert g["decay"] > 0.0                                   # leverage decays vs naive 3x
    assert abs(g["decay"] - g["drag_theory"]) < 0.05          # and it's close to 0.5*L(L-1)*vol^2


def test_no_decay_without_vol(calm_world):
    s, _ = calm_world
    g = st.decay_gap(s, L=3.0)
    assert abs(g["decay"]) < 0.01                             # no vol → no drag
    assert g["drag_theory"] < 0.01


def test_summary_runs(volatile_world):
    s, _ = volatile_world
    assert "sharpe" in st.summary(st.lever_daily(s, 3.0))
