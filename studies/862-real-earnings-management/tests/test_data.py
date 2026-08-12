"""Data-layer invariants: the per-firm quarterly build (no look-ahead, lag alignment), the
Roychowdhury normal-model residuals (mean-zero, planted-abnormal recovery), and the synthetic
panel's shape — all offline, fixed seeds. A real-cache smoke test is skipped when the cache is
absent (offline CI)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from real_earn_mgmt import data  # noqa: E402


def _flow(vals, start="2016-03-31"):
    ends = pd.date_range(start, periods=len(vals), freq="QE")
    filed = ends + pd.Timedelta(days=35)
    return pd.DataFrame({"end": ends, "filed": filed, "val": [float(v) for v in vals]})


def _instant(vals, start="2016-03-31"):
    ends = pd.date_range(start, periods=len(vals), freq="QE")
    filed = ends + pd.Timedelta(days=35)
    return pd.DataFrame({"end": ends, "filed": filed, "val": [float(v) for v in vals]})


# --------------------------------------------------------------------------- #
# per-firm quarterly build
# --------------------------------------------------------------------------- #
def test_build_firm_quarters_shapes_and_lags():
    n = 8
    rev = _flow(np.linspace(100, 170, n))
    cogs = _flow(np.linspace(60, 100, n))
    sga = _flow([10] * n)
    rnd = _flow([5] * n)
    inv = _instant(np.linspace(40, 75, n))
    assets = _instant([1000] * n)
    fp = data.build_firm_quarters(rev, cogs, sga, rnd, inv, assets)
    # first quarter has no prior -> dropped; remaining rows carry a proper lag
    assert len(fp) == n - 1
    assert (fp["assets_lag"] == 1000).all()
    # inv_lag of row k equals inv of the previous quarter end
    row = fp.iloc[0]
    assert row["inv_lag"] == pytest.approx(inv["val"].iloc[0], abs=1e-6)


def test_build_firm_quarters_filed_after_end():
    n = 6
    rev = _flow([100] * n); cogs = _flow([60] * n); sga = _flow([10] * n)
    rnd = _flow([5] * n); inv = _instant([40] * n); assets = _instant([1000] * n)
    fp = data.build_firm_quarters(rev, cogs, sga, rnd, inv, assets)
    assert (fp["filed"] > fp["end"]).all()          # signal stamped with the FILING date


def test_build_firm_quarters_missing_rnd_is_zero():
    n = 6
    rev = _flow([100] * n); cogs = _flow([60] * n); sga = _flow([10] * n)
    rnd = pd.DataFrame(columns=["end", "filed", "val"])       # firm reports no R&D
    inv = _instant([40] * n); assets = _instant([1000] * n)
    fp = data.build_firm_quarters(rev, cogs, sga, rnd, inv, assets)
    assert (fp["rnd"] == 0.0).all()


# --------------------------------------------------------------------------- #
# Roychowdhury normal-model residuals
# --------------------------------------------------------------------------- #
def _panel_from_firms(firms):
    frames = []
    for tk, fp in firms.items():
        q = fp.copy(); q.insert(0, "ticker", tk); frames.append(q)
    return pd.concat(frames, ignore_index=True)


def test_normal_model_residuals_mean_zero():
    rng = np.random.default_rng(0)
    firms = {}
    for j in range(12):
        n = 20
        rev = _flow(rng.uniform(80, 200, n))
        cogs = _flow(rng.uniform(40, 120, n))
        sga = _flow(rng.uniform(5, 25, n))
        rnd = _flow(rng.uniform(0, 15, n))
        inv = _instant(rng.uniform(20, 90, n))
        assets = _instant(rng.uniform(800, 1200, n))
        firms[f"N{j}"] = data.build_firm_quarters(rev, cogs, sga, rnd, inv, assets)
    panel = data.normal_model_residuals(_panel_from_firms(firms))
    # OLS residuals about a fitted plane -> essentially mean zero
    assert abs(np.nanmean(panel["ab_disx"])) < 1e-6
    assert abs(np.nanmean(panel["ab_prod"])) < 1e-6
    # rem is the aggregate ab_prod - ab_disx
    assert np.allclose(panel["rem"], panel["ab_prod"] - panel["ab_disx"], equal_nan=True)


def test_normal_model_recovers_planted_overproduction():
    # Two cohorts sharing the same normal relations; one cohort overproduces (inflated PROD) and
    # cuts discretionary (deflated DISX) -> it must show high ab_prod and low ab_disx (=> high REM).
    rng = np.random.default_rng(1)
    firms = {}
    for j in range(20):
        n = 24
        manage = (j % 2 == 0)
        a = 1000.0
        sales = rng.uniform(150, 250, n)
        inv = np.cumsum(rng.normal(0, 3, n)) + 60
        # normal COGS ~ 0.6*sales ; managers overproduce -> +delta inventory buildup already in inv
        cogs = 0.6 * sales + rng.normal(0, 2, n)
        # normal discretionary ~ 0.12*sales ; managers cut it hard
        disc = (0.12 - (0.05 if manage else 0.0)) * sales + rng.normal(0, 0.5, n)
        rnd = 0.4 * disc
        sga = 0.6 * disc
        if manage:
            inv = inv + 25.0        # overproduction: inventory (hence PROD) abnormally high
        rev = _flow(sales); cogs_s = _flow(cogs); sga_s = _flow(sga); rnd_s = _flow(rnd)
        inv_s = _instant(inv); assets = _instant([a] * n)
        firms[f"N{j:02d}"] = data.build_firm_quarters(rev, cogs_s, sga_s, rnd_s, inv_s, assets)
    panel = data.normal_model_residuals(_panel_from_firms(firms))
    panel = panel.assign(mng=[int(t[1:]) % 2 == 0 for t in panel["ticker"]])
    managed = panel[panel["mng"]]["rem"].mean()
    clean = panel[~panel["mng"]]["rem"].mean()
    assert managed > clean                       # the manipulators score higher REM


# --------------------------------------------------------------------------- #
# build_events end to end (synthetic firm panels)
# --------------------------------------------------------------------------- #
def test_build_events_columns_and_next_gm():
    rng = np.random.default_rng(2)
    firms = {}
    for j in range(6):
        n = 16
        rev = _flow(rng.uniform(100, 200, n))
        cogs = _flow(rng.uniform(50, 120, n))
        sga = _flow(rng.uniform(5, 20, n))
        rnd = _flow(rng.uniform(0, 10, n))
        inv = _instant(rng.uniform(30, 80, n))
        assets = _instant(rng.uniform(900, 1100, n))
        firms[f"N{j}"] = data.build_firm_quarters(rev, cogs, sga, rnd, inv, assets)
    ev = data.build_events(firms)
    for c in ("ticker", "end", "filed", "rem", "ab_prod", "ab_disx", "gm", "next_gm"):
        assert c in ev.columns
    assert ev["rem"].notna().all()
    assert (ev["filed"] > ev["end"]).all()


# --------------------------------------------------------------------------- #
# synthetic price/signal panel
# --------------------------------------------------------------------------- #
def test_synthetic_panel_shapes():
    prices, ev = data.synthetic_panel(n_names=12, n_quarters=24, edge=0.0, seed=1)
    assert prices.shape[1] == 12
    assert {"ticker", "end", "filed", "rem", "ab_prod", "ab_disx"}.issubset(ev.columns)
    assert (ev["filed"] > ev["end"]).all()
    assert np.allclose(ev["rem"], ev["ab_prod"] - ev["ab_disx"])
    assert len(ev) > 50


def test_synthetic_panel_deterministic():
    p1, e1 = data.synthetic_panel(n_names=8, n_quarters=20, edge=0.1, seed=42)
    p2, e2 = data.synthetic_panel(n_names=8, n_quarters=20, edge=0.1, seed=42)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
    assert np.allclose(e1["rem"].to_numpy(), e2["rem"].to_numpy())


# --------------------------------------------------------------------------- #
# real cache smoke test (skipped offline)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_cache_wellformed():
    px, ev = data.load_real()
    assert px.shape[1] >= 10 and len(ev) > 100
    assert {"ticker", "end", "filed", "rem"}.issubset(ev.columns)
    assert (ev["filed"] <= pd.Timestamp(data.AS_OF)).all()
