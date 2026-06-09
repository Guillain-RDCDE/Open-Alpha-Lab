"""The GEX reducer respects the dealer sign convention, the realised-character functions behave,
and the synthetic panel bakes in both a VIX confound and a genuine gamma effect."""

import numpy as np
import pandas as pd

from gamma_gospel import data


# --------------------------------------------------------------------------- #
# compute_gex — the sign convention is the whole study
# --------------------------------------------------------------------------- #

def test_call_heavy_chain_is_positive_gamma():
    """More call OI than put OI => dealers long gamma => GEX > 0 (the 'shock absorber')."""
    chain = data.synthetic_chain(spot=100.0, call_oi_scale=2.0, put_oi_scale=1.0, seed=1)
    g = data.compute_gex(chain, spot=100.0)
    assert g["gex"] > 0


def test_put_heavy_chain_is_negative_gamma():
    """More put OI => dealers short gamma => GEX < 0 (the 'amplifier')."""
    chain = data.synthetic_chain(spot=100.0, call_oi_scale=1.0, put_oi_scale=2.0, seed=1)
    g = data.compute_gex(chain, spot=100.0)
    assert g["gex"] < 0


def test_walls_are_strikes_in_range():
    chain = data.synthetic_chain(spot=100.0, seed=2)
    g = data.compute_gex(chain, spot=100.0)
    assert 85.0 <= g["call_wall"] <= 115.0
    assert 85.0 <= g["put_wall"] <= 115.0


def test_compute_gex_empty_is_nan():
    g = data.compute_gex(pd.DataFrame(columns=["type", "strike", "gamma", "open_interest"]), 100.0)
    assert np.isnan(g["gex"])


# --------------------------------------------------------------------------- #
# realised character
# --------------------------------------------------------------------------- #

def test_parkinson_monotone_in_range():
    """A wider high-low range => a higher Parkinson vol."""
    narrow = data.parkinson_vol(np.array([101.0]), np.array([100.0]))[0]
    wide = data.parkinson_vol(np.array([110.0]), np.array([100.0]))[0]
    assert wide > narrow > 0


def test_bs_gamma_peaks_at_the_money_and_is_positive():
    """Black-Scholes gamma is positive everywhere and peaks at the money."""
    strikes = np.array([90.0, 100.0, 110.0])
    g = data.bs_gamma(spot=100.0, strike=strikes, t_years=30 / 365.0, iv=0.20)
    assert (g > 0).all()
    assert g[1] > g[0] and g[1] > g[2]          # ATM gamma is the largest


def test_directional_efficiency_bounds():
    """DE = 1 when the day closes at the extreme of its range; ~0 when it round-trips."""
    trend = data.directional_efficiency(np.array([100.0]), np.array([105.0]),
                                        np.array([100.0]), np.array([105.0]))[0]
    chop = data.directional_efficiency(np.array([102.5]), np.array([105.0]),
                                       np.array([100.0]), np.array([102.5]))[0]
    assert np.isclose(trend, 1.0)
    assert chop < 0.1


# --------------------------------------------------------------------------- #
# synthetic panel — confound present, genuine effect baked in
# --------------------------------------------------------------------------- #

def test_panel_has_canonical_columns(panel):
    assert list(panel.columns) == data.PANEL_COLUMNS
    assert panel["neg_gamma"].dtype.name == "boolean"


def test_vix_drives_the_regime_label(panel):
    """The confound: negative-gamma days carry a higher VIX than positive-gamma days."""
    neg = panel["neg_gamma"].astype(bool)
    assert panel.loc[neg, "vix"].mean() > panel.loc[~neg, "vix"].mean() + 1.0


def test_genuine_effect_lifts_vol_and_trend(panel):
    """With beta > 0, negative-gamma days are both more volatile and more directional, raw."""
    neg = panel["neg_gamma"].astype(bool)
    assert panel.loc[neg, "rv"].mean() > panel.loc[~neg, "rv"].mean()
    assert panel.loc[neg, "de"].mean() > panel.loc[~neg, "de"].mean()


def test_build_panel_attributes_gex_to_next_session():
    """GEX measured at day d's close grades day d+1's OHLC; the row lands on the d+1 date."""
    dates = pd.bdate_range("2024-01-02", periods=3)
    bars = pd.DataFrame(
        {"Open": [100, 101, 102], "High": [101, 104, 103],
         "Low": [99, 100, 101], "Close": [100.5, 103, 102.5]}, index=dates)
    spots = pd.Series([100.0, 102.0, 102.0], index=dates)
    vix = pd.Series([15.0, 22.0, 20.0], index=dates)
    call_heavy = data.synthetic_chain(call_oi_scale=2.0, put_oi_scale=1.0, seed=3)
    chains = {d.strftime("%Y-%m-%d"): call_heavy for d in dates}
    out = data.build_panel(chains, spots, bars, vix)
    assert dates[1] in out.index and dates[2] in out.index   # outcome dates, not the chain date
    assert dates[0] not in out.index                         # the first chain has no prior session
    assert (out["vix"] == pd.Series([15.0, 22.0], index=dates[1:])).all()  # prior-close VIX
    assert (out["gex"] > 0).all()                            # call-heavy chain => positive
