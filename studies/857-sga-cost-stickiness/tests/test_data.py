"""Data-layer invariants: the ABJ stickiness estimator (asymmetric SG&A response, sign of β₂,
identification guards, no look-ahead), the YoY change construction, the synthetic panel's shape,
and the optional real-cache gate — all offline, fixed seeds."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sga_stickiness import data  # noqa: E402


def _quarter_series(vals, start="2010-03-31"):
    """A quarterly flow series (end/filed/val) with filing ~40 days after each quarter end."""
    ends = pd.period_range(start, periods=len(vals), freq="Q")
    end_ts = pd.to_datetime([p.end_time.normalize() for p in ends])
    filed = end_ts + pd.Timedelta(days=40)
    return pd.DataFrame({"end": end_ts, "filed": filed, "val": [float(v) for v in vals]})


# --------------------------------------------------------------------------- #
# YoY change construction
# --------------------------------------------------------------------------- #
def test_yoy_changes_basic_math():
    # revenue +10% YoY at q5, SG&A +5% YoY at q5
    rev = _quarter_series([100, 100, 100, 100, 110, 110, 110, 110])
    sga = _quarter_series([50, 50, 50, 50, 52.5, 52.5, 52.5, 52.5])
    yc = data.yoy_changes(sga, rev)
    row = yc.iloc[0]
    assert row["dlog_rev"] == pytest.approx(np.log(1.10), abs=1e-9)
    assert row["dlog_sga"] == pytest.approx(np.log(1.05), abs=1e-9)
    assert row["dec"] == 0                       # sales rose -> not a decrease


def test_yoy_changes_decrease_flag():
    rev = _quarter_series([100, 100, 100, 100, 90, 90, 90, 90])
    sga = _quarter_series([50, 50, 50, 50, 48, 48, 48, 48])
    yc = data.yoy_changes(sga, rev)
    assert yc.iloc[0]["dec"] == 1                # sales fell YoY -> decrease


def test_yoy_changes_filed_after_end_no_lookahead():
    rev = _quarter_series([100, 101, 102, 103, 104])
    sga = _quarter_series([50, 50, 50, 50, 51])
    yc = data.yoy_changes(sga, rev)
    assert (yc["filed"] > yc["end"]).all()       # signal stamped with the filing date


# --------------------------------------------------------------------------- #
# ABJ stickiness estimator — the spine of the study
# --------------------------------------------------------------------------- #
def _make_yoy_window(n, beta1, beta2, seed=0, dec_frac=0.4):
    """Build a YoY-observation window whose SG&A obeys the ABJ model with known (β₁, β₂)."""
    rng = np.random.default_rng(seed)
    g = rng.normal(0.0, 0.08, n)
    dec = (g < np.quantile(g, dec_frac)).astype(float)   # decreases are the low-growth tail
    # force the decrease rows to actually have negative sales growth
    g = np.where(dec == 1, -np.abs(g) - 0.02, np.abs(g) + 0.02)
    dec = (g < 0).astype(float)
    y = beta1 * g + beta2 * dec * g + rng.normal(0.0, 0.002, n)
    return pd.DataFrame({"dlog_rev": g, "dlog_sga": y, "dec": dec.astype(int)})


def test_estimate_stickiness_recovers_planted_betas():
    win = _make_yoy_window(120, beta1=0.60, beta2=-0.25, seed=1)
    b1, b2, n, n_dec = data.estimate_stickiness(win)
    assert b1 == pytest.approx(0.60, abs=0.05)
    assert b2 == pytest.approx(-0.25, abs=0.06)
    assert n == 120 and n_dec > data.MIN_DEC


def test_stickiness_is_negative_beta2():
    # a sticky firm (costs cling on the way down) has β₂ < 0 -> stickiness = -β₂ > 0
    win = _make_yoy_window(120, beta1=0.60, beta2=-0.30, seed=2)
    b1, b2, n, n_dec = data.estimate_stickiness(win)
    assert b2 < 0
    assert -b2 > 0                               # stickiness positive for a sticky firm


def test_estimate_stickiness_unidentified_when_too_few_obs():
    win = _make_yoy_window(data.MIN_OBS - 1, beta1=0.6, beta2=-0.2, seed=3)
    b1, b2, n, n_dec = data.estimate_stickiness(win)
    assert not np.isfinite(b2)                   # below the minimum-obs bar -> NaN


def test_estimate_stickiness_unidentified_without_declines():
    # a firm that never had a YoY sales decline cannot identify β₂
    rng = np.random.default_rng(4)
    g = np.abs(rng.normal(0.05, 0.02, 60)) + 0.01
    win = pd.DataFrame({"dlog_rev": g, "dlog_sga": 0.6 * g, "dec": np.zeros(60, int)})
    b1, b2, n, n_dec = data.estimate_stickiness(win)
    assert n_dec == 0
    assert not np.isfinite(b2)


def test_build_events_expanding_window_is_point_in_time():
    # SG&A rises 0.6% per 1% up-move, falls only 0.3% per 1% down-move -> sticky (β₂≈-0.3)
    rng = np.random.default_rng(5)
    n = 60
    g = rng.normal(0.0, 0.06, n)
    g[::5] = -np.abs(g[::5]) - 0.02               # sprinkle genuine declines
    dec = (g < 0).astype(float)
    y = 0.6 * g - 0.3 * dec * g + rng.normal(0, 0.002, n)
    # reconstruct SG&A/Rev levels from the YoY changes
    rev_lvl = np.exp(np.concatenate([[6.0] * 4, np.zeros(n)]))
    # build simple quarterly level series that reproduce these YoY changes
    r = np.zeros(n + 4); s = np.zeros(n + 4)
    r[:4] = 6.0; s[:4] = 5.0
    for t in range(4, n + 4):
        r[t] = r[t - 4] + g[t - 4]
        s[t] = s[t - 4] + y[t - 4]
    rev = _quarter_series(np.exp(r) * 1e6)
    sga = _quarter_series(np.exp(s) * 1e6)
    ev = data.build_stickiness_events(sga, rev)
    assert (ev["filed"] > ev["end"]).all()        # every event dated by its filing
    assert (ev["stickiness"] == -ev["beta2"]).all()
    # the mature (last) estimate should recover a sticky firm
    assert ev.iloc[-1]["stickiness"] > 0


# --------------------------------------------------------------------------- #
# synthetic panel shape
# --------------------------------------------------------------------------- #
def test_synthetic_panel_shapes():
    prices, ev = data.synthetic_panel(n_names=12, n_quarters=44, edge=0.0, seed=1)
    assert prices.shape[1] == 12
    assert {"ticker", "end", "filed", "disc", "stickiness", "beta2"}.issubset(ev.columns)
    assert (ev["filed"] > ev["end"]).all()
    assert len(ev) > 50
    assert (ev["disc"] == -ev["stickiness"]).all()


def test_synthetic_panel_deterministic():
    p1, e1 = data.synthetic_panel(n_names=8, n_quarters=44, edge=0.1, seed=42)
    p2, e2 = data.synthetic_panel(n_names=8, n_quarters=44, edge=0.1, seed=42)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
    assert np.allclose(e1["disc"].to_numpy(), e2["disc"].to_numpy())


def test_synthetic_no_timestamp_overflow():
    # PeriodIndex-based synthetic dates must stay well inside the Timestamp horizon
    _, ev = data.synthetic_panel(n_names=6, n_quarters=52, edge=0.0, seed=7)
    assert ev["end"].max() < pd.Timestamp("2100-01-01")


# --------------------------------------------------------------------------- #
# real-cache gate (skips cleanly when the cache is absent, e.g. offline CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_cache_wellformed():
    px, ev = data.load_real()
    assert px.shape[0] > 200 and px.shape[1] >= 10
    assert {"ticker", "filed", "disc", "stickiness"}.issubset(ev.columns)
    assert (ev["filed"] <= pd.Timestamp(data.AS_OF)).all()
    assert ev["stickiness"].notna().all()
