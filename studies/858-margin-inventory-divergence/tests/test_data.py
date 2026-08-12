"""Data-layer invariants: the divergence construction (gross-margin math, YoY growth, the
inventory-vs-sales gap, no look-ahead) and the synthetic panel's shape — all offline, fixed
seeds. A real-cache smoke test is skipped when the (git-ignored) cache is absent."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from margin_inventory import data  # noqa: E402


def _flow_series(vals, start="2018-03-31"):
    ends = pd.date_range(start, periods=len(vals), freq="QE")
    filed = ends + pd.Timedelta(days=35)
    return pd.DataFrame({"end": ends, "filed": filed, "val": [float(v) for v in vals]})


def test_build_signal_divergence_math():
    # 8 quarters. Revenue flat at 200; cost drops 160 -> 150 at q5 so GM rises 0.20 -> 0.25.
    rev = _flow_series([200] * 8)
    cost = _flow_series([160, 160, 160, 160, 150, 150, 150, 150])
    # inventory grows 10% YoY at q5 (100 -> 110); sales growth is 0.
    inv = _flow_series([100, 100, 100, 100, 110, 110, 110, 110])
    sig = data.build_signal(rev, cost, inv)
    row = sig[sig["end"] == rev["end"].iloc[4]].iloc[0]
    # ΔGM = 0.25 - 0.20 = +0.05 ; inv_growth = 0.10, sales_growth = 0 -> gap +0.10
    assert row["d_gross_margin"] == pytest.approx(0.05, abs=1e-9)
    assert row["inv_growth"] == pytest.approx(0.10, abs=1e-9)
    assert row["sales_growth"] == pytest.approx(0.0, abs=1e-9)
    assert row["inv_sales_gap"] == pytest.approx(0.10, abs=1e-9)
    # divergence = ΔGM - gap = 0.05 - 0.10 = -0.05  (margin up but inventory outruns sales -> contradictory)
    assert row["divergence"] == pytest.approx(-0.05, abs=1e-9)


def test_build_signal_gross_margin_level():
    rev = _flow_series([200] * 6)
    cost = _flow_series([150] * 6)
    inv = _flow_series([100] * 6)
    sig = data.build_signal(rev, cost, inv)
    # GM = (200-150)/200 = 0.25 everywhere
    assert np.allclose(sig["gross_margin"].to_numpy(), 0.25)


def test_build_signal_no_lookahead_filed_date():
    rev = _flow_series([200] * 5)
    cost = _flow_series([150] * 5)
    inv = _flow_series([100, 100, 100, 100, 120])
    sig = data.build_signal(rev, cost, inv)
    # the signal for period end E must be stamped with a FILING date strictly AFTER E
    assert (sig["filed"] > sig["end"]).all()


def test_build_signal_drops_nonpositive_prior_inventory():
    rev = _flow_series([200] * 5)
    cost = _flow_series([150] * 5)
    inv = _flow_series([0, 100, 100, 100, 120])   # q1 inventory 0 -> q5 YoY denominator invalid
    sig = data.build_signal(rev, cost, inv)
    assert (sig["end"] == inv["end"].iloc[4]).sum() == 0


def test_synthetic_panel_shapes():
    prices, ev = data.synthetic_panel(n_names=12, n_quarters=20, edge=0.0, seed=1)
    assert prices.shape[1] == 12
    assert {"ticker", "end", "filed", "divergence"}.issubset(ev.columns)
    assert (ev["filed"] > ev["end"]).all()
    assert len(ev) > 50


def test_synthetic_panel_deterministic():
    p1, e1 = data.synthetic_panel(n_names=8, n_quarters=15, edge=0.1, seed=42)
    p2, e2 = data.synthetic_panel(n_names=8, n_quarters=15, edge=0.1, seed=42)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
    assert np.allclose(e1["divergence"].to_numpy(), e2["divergence"].to_numpy())


def test_synthetic_index_within_timestamp_horizon():
    # guard against the timestamp-overflow trap: the synthetic index must stay a real bdate range
    prices, _ = data.synthetic_panel(n_names=4, n_quarters=30, edge=0.0, seed=3)
    assert prices.index.max().year < 2100


@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_cache_loads_and_is_pointintime():
    px, ev = data.load_real()
    assert px.shape[1] > 5 and len(ev) > 100
    assert (ev["filed"] >= ev["end"]).all()      # filing never precedes the period end
    assert ev["filed"].max() <= pd.Timestamp(data.AS_OF)
