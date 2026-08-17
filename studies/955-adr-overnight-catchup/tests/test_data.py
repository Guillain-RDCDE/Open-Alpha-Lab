"""Data-layer tests — the synthetic panel, the cache convention, the offline contract.

Everything here runs with NO cache and NO network: the real-tape loader is only checked
for the shape of its *contract* (path convention, the error it raises when the cache is
absent), never for its contents.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adr_catchup import data  # noqa: E402


# --------------------------------------------------------------------------- #
# The universe and the cache convention
# --------------------------------------------------------------------------- #
def test_universe_is_well_formed():
    assert len(data.ADR_PAIRS) == 8
    for adr, (home, fx, invert, region) in data.ADR_PAIRS.items():
        assert home in data.HOME_INDICES
        assert fx in data.FX
        assert isinstance(invert, bool)
        assert region in {"Japan", "Europe", "UK"}
    # Only the yen pair is quoted local-per-USD and therefore needs inverting.
    inverted = {a for a, v in data.ADR_PAIRS.items() if v[2]}
    assert inverted == {"TM", "SONY"}


def test_cache_path_strips_caret_and_equals():
    """The shared cache convention: prices_<TICKER>_1d.parquet with ^ and = stripped."""
    assert data._cache_path("^N225", "/c").endswith(os.path.join("/c", "prices_N225_1d.parquet"))
    assert data._cache_path("JPY=X", "/c").endswith(os.path.join("/c", "prices_JPYX_1d.parquet"))
    assert data._cache_path("TM", "/c").endswith(os.path.join("/c", "prices_TM_1d.parquet"))


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_false_on_empty_cache(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


def test_as_of_is_a_complete_past_month():
    asof = pd.Timestamp(data.AS_OF)
    assert asof == asof + pd.offsets.MonthEnd(0)   # last day of its month
    assert asof < pd.Timestamp("2026-08-17")       # never in the future


def test_data_module_does_not_import_yfinance_at_module_scope():
    src = open(os.path.join(os.path.dirname(data.__file__), "data.py"),
               encoding="utf-8").read()
    head = src.split("def fetch(")[0]
    assert "import yfinance" not in head


# --------------------------------------------------------------------------- #
# The synthetic panel
# --------------------------------------------------------------------------- #
def test_synthetic_panel_shape_and_columns(raw_planted):
    panel, truth = raw_planted
    assert {"a", "h", "f", "x", "adr", "region"}.issubset(panel.columns)
    assert panel["adr"].nunique() == truth["n_names"]
    assert len(panel) == truth["n_names"] * truth["n_days"]
    assert panel.index.is_monotonic_increasing
    assert not panel[["a", "h", "f", "x"]].isna().any().any()


def test_synthetic_panel_is_deterministic():
    p1, _ = data.synthetic_panel(n_names=4, n_years=3, seed=7)
    p2, _ = data.synthetic_panel(n_names=4, n_years=3, seed=7)
    p3, _ = data.synthetic_panel(n_names=4, n_years=3, seed=8)
    assert np.allclose(p1["a"].to_numpy(), p2["a"].to_numpy())
    assert not np.allclose(p1["a"].to_numpy(), p3["a"].to_numpy())


def test_signal_strength_sets_the_planted_lag_share():
    _, t_full = data.synthetic_panel(n_names=4, n_years=3, signal_strength=1.0)
    _, t_half = data.synthetic_panel(n_names=4, n_years=3, signal_strength=0.5)
    _, t_null = data.synthetic_panel(n_names=4, n_years=3, signal_strength=0.0)
    assert t_full["lag_share"] == pytest.approx(t_full["max_lag_share"])
    assert t_half["lag_share"] == pytest.approx(0.5 * t_full["max_lag_share"])
    assert t_null["lag_share"] == 0.0


def test_x_is_the_compounded_home_and_fx_move(raw_planted):
    panel, _ = raw_planted
    recomputed = (1.0 + panel["h"]) * (1.0 + panel["f"]) - 1.0
    assert np.allclose(panel["x"].to_numpy(), recomputed.to_numpy())


def test_synthetic_dates_stay_inside_pandas_range():
    """Guard the pandas 2.x ns-Timestamp overflow trap on long daily ranges."""
    panel, truth = data.synthetic_panel(n_names=2, n_years=30)
    assert panel.index.max() < pd.Timestamp("2262-01-01")
    assert truth["n_days"] == 30 * 252


def test_synthetic_panel_vol_is_sane(raw_planted):
    panel, _ = raw_planted
    ann = panel.groupby("adr")["a"].std() * np.sqrt(252)
    assert (ann > 0.10).all() and (ann < 0.60).all()


# --------------------------------------------------------------------------- #
# build_panel / cash_returns on a hand-made close frame (no cache needed)
# --------------------------------------------------------------------------- #
def _fake_prices(n=400):
    idx = pd.bdate_range("2015-01-02", periods=n)
    rng = np.random.default_rng(0)
    def walk(v):
        return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, v, n))), index=idx)
    return pd.DataFrame({
        "TM": walk(0.014), "^N225": walk(0.011), "JPY=X": walk(0.005),
        "SAP": walk(0.016), "^GDAXI": walk(0.012), "EURUSD=X": walk(0.005),
        "^IRX": pd.Series(np.full(n, 4.0), index=idx),
    })


def test_build_panel_builds_one_row_per_name_per_date():
    px = _fake_prices()
    pairs = {"TM": data.ADR_PAIRS["TM"], "SAP": data.ADR_PAIRS["SAP"]}
    panel = data.build_panel(px, pairs=pairs)
    assert set(panel["adr"].unique()) == {"TM", "SAP"}
    assert panel.index.is_monotonic_increasing
    assert len(panel) == 2 * (len(px) - 1)


def test_build_panel_inverts_only_the_yen_leg():
    px = _fake_prices()
    panel = data.build_panel(px, pairs={k: data.ADR_PAIRS[k] for k in ("TM", "SAP")})
    jpy = (px["JPY=X"] / px["JPY=X"].shift(1) - 1.0).dropna()
    tm = panel[panel["adr"] == "TM"]
    assert np.allclose(tm["f"].to_numpy(), (1.0 / (1.0 + jpy) - 1.0).to_numpy())
    eur = (px["EURUSD=X"] / px["EURUSD=X"].shift(1) - 1.0).dropna()
    sap = panel[panel["adr"] == "SAP"]
    assert np.allclose(sap["f"].to_numpy(), eur.to_numpy())


def test_build_panel_drops_a_shut_market_instead_of_padding_it():
    """A home-market holiday must remove two rows, not print a 0% day and a two-day move.

    The close frame is a union of calendars, so a shut market is a NaN. ``pct_change``'s
    legacy ``fill_method="pad"`` default (still the default on pandas 2.x) would carry the
    stale close forward: a spurious 0% on the holiday, and yesterday-plus-today smeared
    into the next row — which is exactly the contamination a catch-up study must not have.
    ``build_panel`` divides explicitly instead, so both rows drop and the result is
    identical on pandas 2.x and 3.x.
    """
    px = _fake_prices(120)
    shut = px.index[50]
    px.loc[shut, "^N225"] = np.nan
    panel = data.build_panel(px, pairs={"TM": data.ADR_PAIRS["TM"]})
    assert shut not in panel.index                      # the holiday itself is gone
    assert px.index[51] not in panel.index              # and so is the two-day move
    assert px.index[49] in panel.index and px.index[52] in panel.index
    assert len(panel) == (len(px) - 1) - 2
    assert (panel["h"] != 0.0).all()                    # no padded 0% day survived


def test_cash_returns_are_lagged_and_positive():
    px = _fake_prices()
    c = data.cash_returns(px)
    assert c.iloc[0] != c.iloc[0] or np.isnan(c.iloc[0])   # first value is NaN (shifted)
    assert c.dropna().mean() == pytest.approx(4.0 / 100 / 252, rel=1e-9)


def test_fingerprint_is_stable_and_content_sensitive():
    px = _fake_prices(50)
    f1 = data.fingerprint(px)
    assert f1 == data.fingerprint(px.copy())
    bumped = px.copy()
    bumped.iloc[10, 0] *= 1.001
    assert data.fingerprint(bumped) != f1
