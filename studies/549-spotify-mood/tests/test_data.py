"""Tests for the data layer of Study 549 (Spotify-Mood). Offline & deterministic."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spotify_mood import data  # noqa: E402


def test_synthetic_deterministic():
    a, _ = data.synthetic_series(seed=549)
    b, _ = data.synthetic_series(seed=549)
    assert data.fingerprint(a) == data.fingerprint(b)


def test_synthetic_schema_and_bounds(null_frame):
    frame, truth = null_frame
    assert list(frame.columns) == ["valence", "mkt_ret"]
    assert (frame["valence"] >= 0.0).all() and (frame["valence"] <= 1.0).all()
    assert not truth.has_edge


def test_edge_truth_flag(edge_frame):
    _, truth = edge_frame
    assert truth.has_edge and truth.predictive_beta > 0


def test_fetch_market_offline_empty(tmp_path):
    # cache-miss + fetch=False must return empty, never touch the network
    s = data.fetch_market(cache_dir=str(tmp_path), fetch=False)
    assert len(s) == 0


def test_join_offline_empty():
    import pandas as pd

    out = data.join_synthetic_valence(pd.Series(dtype=float))
    assert out.empty
