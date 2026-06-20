"""The synthetic panel is well-formed, deterministic, plants delisting correctly, and the
real loader is cache- and opt-in-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from survivorship_bias import data  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
PANEL_CACHE = os.path.join(CACHE, "survivorship_panel.parquet")


# ---- synthetic shape & determinism (never skip) ----------------------------
def test_synthetic_shape(biased_panel):
    rets, alive, truth = biased_panel
    assert rets.shape == (truth["n_periods"], truth["n_assets"])
    assert alive.shape == rets.shape
    assert list(rets.columns) == list(alive.columns)


def test_no_deaths_means_no_nans_and_full_alive(null_panel):
    """death_rate=0 → nobody delists → no NaNs and every cell alive."""
    rets, alive, truth = null_panel
    assert truth["n_dead"] == 0
    assert alive.to_numpy().all()
    assert np.isfinite(rets.to_numpy()).all()


def test_deaths_plant_nans_and_drop_survivors(biased_panel):
    """A 30% death rate removes ~30% of names from the SURVIVORS view and plants NaNs."""
    rets, alive, truth = biased_panel
    assert truth["n_dead"] == int(round(0.30 * truth["n_assets"]))
    surv = data.survivors_only(rets, alive)
    # Survivors are strictly fewer than the full universe, by ~the death count.
    assert surv.shape[1] == truth["n_assets"] - truth["n_dead"]
    # The full panel has NaNs (delisted cells); the survivors view has none.
    assert rets.isna().to_numpy().any()
    assert not surv.isna().to_numpy().any()


def test_delisting_books_a_terminal_loss(biased_panel):
    """Each doomed name's last LIVE bar is a large negative (the delisting wipe-out)."""
    rets, alive, truth = biased_panel
    for col in truth["doomed_cols"]:
        s = rets[col].dropna()
        # The terminal live return is the −80% wipe (allowing the prior bleed).
        assert s.iloc[-1] < -0.5


def test_survivors_keep_full_history(biased_panel):
    """A surviving name trades every month (alive in the final row → no NaNs)."""
    rets, alive, _ = biased_panel
    surv = data.survivors_only(rets, alive)
    assert surv.notna().to_numpy().all()
    assert len(surv) == len(rets)


def test_synthetic_is_deterministic():
    a, aa, _ = data.synthetic_panel(n_periods=120, n_assets=60, death_rate=0.25, seed=7)
    b, ab, _ = data.synthetic_panel(n_periods=120, n_assets=60, death_rate=0.25, seed=7)
    assert np.allclose(np.nan_to_num(a.to_numpy(), nan=-9.0),
                       np.nan_to_num(b.to_numpy(), nan=-9.0))
    assert (aa.to_numpy() == ab.to_numpy()).all()
    c, _, _ = data.synthetic_panel(n_periods=120, n_assets=60, death_rate=0.25, seed=8)
    assert not np.allclose(np.nan_to_num(a.to_numpy(), nan=-9.0),
                           np.nan_to_num(c.to_numpy(), nan=-9.0))


def test_index_uses_period_range_not_overflow():
    rets, _, _ = data.synthetic_panel(n_periods=240, n_assets=10, seed=1)
    assert rets.index.is_monotonic_increasing
    assert rets.index[0].year >= 1900


def test_fingerprint_stable_and_sensitive(biased_panel):
    rets, _, _ = biased_panel
    assert data.fingerprint(rets) == data.fingerprint(rets)
    other, _, _ = data.synthetic_panel(n_periods=240, n_assets=200, death_rate=0.30, seed=99)
    assert data.fingerprint(rets) != data.fingerprint(other)


# ---- real loader: opt-in guard + cache contract ----------------------------
def test_load_real_requires_survivorship_optin():
    with pytest.raises(PermissionError):
        data.load_real(allow_survivorship_bias=False)


def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(fetch=False, allow_survivorship_bias=True, cache_dir=str(tmp_path))


@pytest.mark.skipif(not os.path.exists(PANEL_CACHE), reason="offline CI: no real cache")
def test_real_panel_loads_from_cache():
    rets = data.load_real(fetch=False, allow_survivorship_bias=True)
    assert rets.shape[1] >= 1
    assert rets.shape[0] >= 24
