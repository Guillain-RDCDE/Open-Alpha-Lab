"""Data-layer tests for Study 1015 — synthetic determinism offline, real tapes gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overlay import data  # noqa: E402


def test_synthetic_tape_is_deterministic():
    a, ta = data.synthetic_tape(n_periods=400, seed=7)
    b, tb = data.synthetic_tape(n_periods=400, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert ta["starts"] == tb["starts"]


def test_synthetic_tape_seed_sensitive():
    a, _ = data.synthetic_tape(n_periods=400, seed=7)
    b, _ = data.synthetic_tape(n_periods=400, seed=8)
    assert not np.allclose(a["lr"].to_numpy(), b["lr"].to_numpy())


def test_null_is_the_same_noise_without_the_template():
    p, tp = data.synthetic_tape(n_periods=600, signal_strength=1.0, seed=3)
    z, tz = data.synthetic_tape(n_periods=600, signal_strength=0.0, seed=3)
    assert tp["starts"] == tz["starts"] and tp["n_episodes"] > 3
    diff = (p["lr"] - z["lr"]).to_numpy()
    motif = np.diff(np.concatenate([[0.0], data.template_path(tp["L"], tp["h"])]))
    s0 = tp["starts"][0]
    assert np.allclose(diff[s0:s0 + tp["L"] + tp["h"]], motif)
    mask = np.ones(600, bool)
    for s in tp["starts"]:
        mask[s:s + tp["L"] + tp["h"]] = False
    assert np.allclose(diff[mask], 0.0)
    assert tz["amp"] == 0.0 and tz["drop"] == 0.0


def test_signal_strength_scales_the_plant():
    _, t1 = data.synthetic_tape(n_periods=300, signal_strength=1.0)
    _, th = data.synthetic_tape(n_periods=300, signal_strength=0.5)
    assert t1["amp"] > th["amp"] > 0 and t1["drop"] > th["drop"] > 0


def test_episodes_never_overlap_and_fit():
    _, t = data.synthetic_tape(n_periods=1100, seed=11)
    span = t["L"] + t["h"]
    s = np.array(t["starts"])
    assert (np.diff(s) >= span).all()
    assert s[-1] + span < 1100


def test_template_path_shape():
    p = data.template_path(24, 6, amp=0.3, drop=0.2)
    assert p.size == 30
    assert p[-1] == pytest.approx(p[23] - 0.2)
    assert np.all(np.diff(p[24:]) < 0)


def test_synthetic_frame_columns_and_horizon():
    df, _ = data.synthetic_tape(n_periods=1500, seed=1)
    assert {"ret", "rf", "lr", "logp"} <= set(df.columns)
    assert np.allclose(df["logp"].to_numpy(), np.cumsum(df["lr"].to_numpy()))
    assert np.allclose(np.log1p(df["ret"]), df["lr"])
    assert df.index[-1] < pd.Timestamp("2262-01-01")
    dd, _ = data.synthetic_tape(n_periods=500, freq="D", start="2000-01-03")
    assert isinstance(dd.index, pd.DatetimeIndex) and len(dd) == 500


def test_tape_moments():
    df, _ = data.synthetic_tape(n_periods=5000, signal_strength=0.0, mu=0.01, sigma=0.05)
    mo = data.tape_moments(df["lr"], 12)
    assert mo["mu"] == pytest.approx(0.01, abs=0.003)
    assert mo["sigma"] == pytest.approx(0.05, rel=0.05)
    assert mo["mu_ann"] == pytest.approx(12 * mo["mu"])


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_tape(n_periods=200, seed=1)
    b, _ = data.synthetic_tape(n_periods=200, seed=2)
    assert data.fingerprint(a) == data.fingerprint(a) and len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


def test_as_of_and_provenance_declared():
    assert pd.Timestamp(data.AS_OF_MONTHLY) <= pd.Timestamp(data.AS_OF_DAILY)
    assert pd.Timestamp(data.AS_OF) < pd.Timestamp("2026-01-01")
    assert "TOTAL" in data.PROVENANCE["monthly"]["what"]
    assert "PRICE" in data.PROVENANCE["daily"]["what"]
    assert len(data.PROVENANCE["daily"]["sha256"]) == 64
    assert isinstance(data.have_real(), bool)


@pytest.mark.skipif(not data.have_monthly(), reason="arch Fama-French tape not installed")
def test_real_monthly_tape():
    m = data.load_monthly()
    assert m.index[0] == pd.Timestamp("1926-07-31")
    assert m.index[-1] == pd.Timestamp(data.AS_OF_MONTHLY)
    assert m.index.is_monotonic_increasing and not m.index.has_duplicates
    assert np.allclose(m["logp"], np.log1p(m["ret"]).cumsum())
    # 1929-09 to 1932-06: the crash the overlay charts are about is on this tape
    assert m.loc["1932-06-30", "logp"] - m.loc["1929-08-31", "logp"] < np.log(0.25)


@pytest.mark.skipif(not data.have_daily(), reason="skfolio tapes not cached")
def test_real_daily_tape_drops_partial_month():
    d = data.load_daily()
    assert d.index[-1] <= pd.Timestamp(data.AS_OF_DAILY)
    assert d.index[-1] >= pd.Timestamp("2022-11-28")
    assert (d["px"] > 0).all() and (d["rf"] == 0).all()
    assert d["lr"].iloc[1:].notna().all()
