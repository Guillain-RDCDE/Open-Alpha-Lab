"""Random-day null and the cross-study control."""

import numpy as np
import pandas as pd

from fear_gauge import benchmark, data, triggers


def test_forward_returns_shape_and_tail_nan(synth_market):
    fwd = benchmark.forward_returns(synth_market, 5)
    assert len(fwd) == len(synth_market)
    assert np.isnan(fwd[-5:]).all() and not np.isnan(fwd[0])


def test_conditional_vs_unconditional_columns(synth_market, synth_gauge):
    sig = triggers.first_crossings(triggers.level(synth_gauge, 30), cooldown=21)
    tbl = benchmark.conditional_vs_unconditional(synth_market, sig, horizons=(1, 5), n_iter=200)
    for col in ("mean_cond", "mean_uncond", "excess", "p_greater"):
        assert col in tbl.columns
    # excess is exactly mean_cond - mean_uncond
    assert np.allclose(tbl["excess"], tbl["mean_cond"] - tbl["mean_uncond"])


def test_pvalue_in_unit_interval(synth_market, synth_gauge):
    sig = triggers.first_crossings(triggers.spike(synth_gauge), cooldown=21)
    tbl = benchmark.conditional_vs_unconditional(synth_market, sig, horizons=(5,), n_iter=200)
    p = tbl["p_greater"].iloc[0]
    assert 0.0 <= p <= 1.0


def test_excess_vs_alternative_gap_identity(synth_market, synth_gauge):
    vix_sig = triggers.first_crossings(triggers.level(synth_gauge, 30), cooldown=21)
    price_sig = triggers.first_crossings(synth_market["Close"].pct_change() <= -0.02, cooldown=21)
    tbl = benchmark.excess_vs_alternative(synth_market, vix_sig, price_sig,
                                          horizons=(5,), n_iter=200)
    if not tbl.empty:
        row = tbl.iloc[0]
        assert np.isclose(row["gap"], row["mean_signal"] - row["mean_alt"])
        assert 0.0 <= row["p_signal_gt_alt"] <= 1.0
