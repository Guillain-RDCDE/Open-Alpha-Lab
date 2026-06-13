"""Data-layer invariants for Study 88 (Dogs-of-the-Dow), incl. the survivorship guard."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dogs_of_the_dow import data  # noqa: E402


def test_anchor_set_is_exactly_30():
    assert len(data._anchor_set()) == 30
    assert len(set(data._anchor_set())) == 30  # no duplicate tickers


def test_membership_always_30_at_every_rebalance():
    """A point-in-time index must hold exactly 30 names at every January in the window."""
    for year in range(2000, 2027):
        m = data.members_asof(f"{year - 1}-12-31")
        assert len(m) == 30, f"{year}: {len(m)} members"
        assert len(set(m)) == 30


def test_membership_before_anchor_is_empty():
    """No silent back-fill: the study does not sample before the 1999-11 anchor."""
    assert data.members_asof("1998-06-30") == []
    assert data.members_asof("1999-10-31") == []


def test_membership_is_point_in_time_not_current():
    """The 2000 set must contain names later REMOVED (GM, EK) and lack later ADDS (AAPL).

    This is the survivorship guard: if the timeline silently used today's 30 names back to
    2000, removed losers (GM, Eastman Kodak) would vanish and added winners (Apple, Nvidia)
    would appear -- exactly the bias that manufactures a fake result.
    """
    set2000 = set(data.members_asof("1999-12-31"))
    set2025 = set(data.members_asof("2025-12-31"))
    # Names that were in at the start and later dropped (the losers survivorship would hide):
    assert {"GM", "EK"} <= set2000
    assert "GM" not in set2025 and "EK" not in set2025
    # Names added only much later (the winners survivorship would back-date):
    assert "AAPL" not in set2000 and "NVDA" not in set2000
    assert "AAPL" in set2025 and "NVDA" in set2025


def test_recent_membership_matches_real_dow():
    """A sanity anchor on the public record: the modern set is the actual current Dow."""
    m = set(data.members_asof("2025-12-31"))
    for t in ("AAPL", "MSFT", "NVDA", "AMZN", "JPM", "V", "UNH", "GS"):
        assert t in m


def test_synthetic_panel_shape_and_determinism():
    a = data.synthetic_panel(planted=True, seed=7)
    b = data.synthetic_panel(planted=True, seed=7)
    assert a["yields"].shape == (a["truth"]["n_years"], a["truth"]["n_names"])
    assert np.allclose(a["fwd_returns"].to_numpy(), b["fwd_returns"].to_numpy())
    assert data.fingerprint(a["yields"]) == data.fingerprint(b["yields"])


def test_synthetic_planted_flag_recorded():
    assert data.synthetic_panel(planted=True)["truth"]["excess"] > 0
    assert data.synthetic_panel(planted=False)["truth"]["excess"] == 0.0


def test_fingerprint_changes_with_data():
    a = data.synthetic_panel(planted=True, seed=1)
    b = data.synthetic_panel(planted=True, seed=2)
    assert data.fingerprint(a["yields"]) != data.fingerprint(b["yields"])
