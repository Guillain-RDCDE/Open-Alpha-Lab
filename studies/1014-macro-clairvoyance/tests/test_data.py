"""Data-layer tests for Study 1014 — synthetic determinism offline, gated real-tape checks."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clairvoyance import data  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="bundled arch/statsmodels tapes unavailable")


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_quarters=120, seed=1014)
    b, _ = data.synthetic_panel(n_quarters=120, seed=1014)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_seed_sensitive():
    a, _ = data.synthetic_panel(n_quarters=120, seed=1014)
    b, _ = data.synthetic_panel(n_quarters=120, seed=1015)
    assert not np.allclose(a["ex"], b["ex"])


def test_synthetic_shape_and_index():
    p, truth = data.synthetic_panel(n_quarters=150, seed=1014)
    assert list(p.columns) == ["mkt", "rf", "ex", *data.MACRO_VARS]
    assert len(p) == 150 and truth["n_quarters"] == 150
    assert isinstance(p.index, pd.DatetimeIndex) and p.index.is_monotonic_increasing
    assert p.index[-1] < pd.Timestamp("2262-01-01")
    assert np.allclose(p["mkt"] - p["rf"], p["ex"])


def test_signal_strength_scales_the_plant():
    _, t1 = data.synthetic_panel(signal_strength=1.0)
    _, th = data.synthetic_panel(signal_strength=0.5)
    _, t0 = data.synthetic_panel(signal_strength=0.0)
    assert t0["beta_eff"] == 0.0 and t1["beta_eff"] > th["beta_eff"] > 0.0


def test_null_keeps_the_same_macro_paths():
    a, _ = data.synthetic_panel(signal_strength=1.0, seed=7)
    b, _ = data.synthetic_panel(signal_strength=0.0, seed=7)
    assert np.allclose(a[list(data.MACRO_VARS)], b[list(data.MACRO_VARS)])
    assert not np.allclose(a["ex"], b["ex"])


def test_planted_lead_is_visible_in_the_raw_correlation():
    p, _ = data.synthetic_panel(signal_strength=1.0)
    assert p["ex"].corr(p["g"].shift(-1)) > 0.3
    assert abs(p["ex"].corr(p["g"])) < p["ex"].corr(p["g"].shift(-1))


def test_good_sign_covers_every_variable():
    assert set(data.GOOD_SIGN) == set(data.MACRO_VARS) == set(data.LABELS)


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(n_quarters=80, seed=1)
    b, _ = data.synthetic_panel(n_quarters=80, seed=2)
    assert data.fingerprint(a) == data.fingerprint(a) and len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


@needs_real
def test_real_panel_spans_the_macro_tape(real_panel):
    p = real_panel
    assert p.index[0] == pd.Timestamp("1959-06-30")
    assert p.index[-1] == pd.Timestamp(data.AS_OF)
    assert len(p) == 202
    assert not p[["ex", *data.MACRO_VARS]].isna().any().any()
    assert p.index.is_monotonic_increasing and not p.index.has_duplicates


@needs_real
def test_as_of_is_not_in_the_future_and_quarters_are_complete():
    q = data.load_equity_quarterly()
    assert q.index[-1] <= pd.Timestamp(data.AS_OF)
    assert pd.Timestamp(data.AS_OF) < pd.Timestamp.today()


@needs_real
def test_equity_is_total_return_and_bills_are_positive(real_panel):
    p = real_panel
    # CRSP total return 1959-2009: roughly 9-11% a year; bills 0-15%.
    assert 0.06 < p["mkt"].mean() * 4 < 0.14
    assert (p["rf"] >= 0).all() and p["rf"].max() < 0.05


@needs_real
def test_macro_units_are_sane(real_panel):
    p = real_panel
    assert 2.0 < p["g"].mean() < 4.5           # annualised real GDP growth, %
    assert np.allclose(p["du"].iloc[1:], p["unemp"].diff().iloc[1:])   # Δ of the level
    assert p["unemp"].iloc[-1] == pytest.approx(9.6)                   # 2009Q3
    assert 2.0 < p["infl"].mean() < 6.0


@needs_real
def test_macro_file_matches_its_pin():
    assert data.macro_file_sha256() == data.PROVENANCE["macro_sha256"]


@needs_real
def test_real_fingerprint_is_pinned(real_panel):
    assert data.fingerprint(real_panel[["mkt", "rf", *data.MACRO_VARS]]) == "6008ef31746a"
