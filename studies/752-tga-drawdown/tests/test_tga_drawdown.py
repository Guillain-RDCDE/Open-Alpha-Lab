"""Offline, deterministic tests for Study 752 — TGA-Drawdown.

Pure-function checks on the hardcoded TGA proxy and the synthetic control — NO network.
Run: ``pytest -q studies/752-tga-drawdown/tests``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tga_drawdown import data, strategy as st


def test_tga_series_monotone_index_and_landmarks():
    """The proxy TGA series drops partial bars and carries the faithful COVID balloon."""
    g = data.tga_series()
    assert g.index.is_monotonic_increasing
    assert (g > 0).all()                      # no zero placeholders leak through
    # the 2020 COVID surge is the record; proxy peak must be well above $1T
    assert g.max() > 1000.0
    assert g.idxmax().year == 2020


def test_injection_is_negative_change():
    """injection = -(g_t - g_{t-1}); a falling balance is a positive injection."""
    syn = data.synthetic_tga(n_months=60, edge=0.0, seed=1)
    chg = st.tga_change(syn, k=1)
    inj = st.injection(syn, k=1)
    m = chg.notna()
    assert np.allclose(inj[m].values, -chg[m].values)
    # drawdown mask agrees with a negative change
    draw = st.drawdown_mask(syn, k=1)
    assert (draw[m] == (chg[m] < 0)).all()


def test_forward_returns_are_lagged_no_lookahead():
    """A signal at t must map to the [t+lag, t+lag+H] return — strictly forward."""
    syn = data.synthetic_tga(n_months=40, edge=0.0, seed=2)
    fwd = st.forward_returns(syn, months=1, lag=1)
    spy = syn["spy"]
    # fwd at row i should equal spy[i+2]/spy[i+1]-1 (lag=1, H=1)
    i = 5
    expected = spy.iloc[i + 2] / spy.iloc[i + 1] - 1.0
    assert abs(fwd.iloc[i] - expected) < 1e-12


def test_synthetic_control_null_is_flat_and_planted_lights_up():
    """edge=0 must not manufacture significance; a large edge must clear the bar."""
    s0 = st.summarize(data.synthetic_tga(n_months=252, edge=0.0, seed=752), 1, k=1)
    s1 = st.summarize(data.synthetic_tga(n_months=252, edge=0.04, seed=752), 1, k=1)
    # null: no false positive on the headline Welch test / placebo
    assert abs(s0["t"]) < 2.0
    assert s0["p_placebo"] > 0.05
    # planted: both the Welch and HAC tests light up well past 2
    assert s1["t"] > 2.0
    assert s1["t_hac"] > 2.0
    assert s1["p_placebo"] < 0.05


@pytest.mark.skipif(not data.have_real(), reason="real-tape cache absent (offline CI)")
def test_real_tape_signal_is_noise():
    """The headline real-tape null: no horizon clears |t|=2 on the TGA proxy."""
    f = data.load_real()
    for m in st.HORIZONS:
        s = st.summarize(f, m)
        assert abs(s["t"]) < 2.0
        assert abs(s["t_hac"]) < 2.0
