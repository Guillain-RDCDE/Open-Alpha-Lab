"""Data-layer tests for Study 971 — the synthetic tape and the cache contract."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tape_audit import data  # noqa: E402


def test_synthetic_tape_is_deterministic():
    a, _ = data.synthetic_tape(n_years=4, seed=971)
    b, _ = data.synthetic_tape(n_years=4, seed=971)
    for fl in data.FLAVOURS:
        assert np.allclose(a[fl].select_dtypes("number").to_numpy(),
                           b[fl].select_dtypes("number").to_numpy())


def test_synthetic_tape_seed_sensitive():
    a, _ = data.synthetic_tape(n_years=4, seed=971)
    b, _ = data.synthetic_tape(n_years=4, seed=972)
    assert not np.allclose(a["daily_tr"].to_numpy(), b["daily_tr"].to_numpy())


def test_all_four_flavours_are_present_and_ordered():
    frames, truth = data.synthetic_tape(n_years=5, seed=971)
    assert set(frames) == set(data.FLAVOURS)
    for fl, df in frames.items():
        assert df.index.is_monotonic_increasing and not df.index.has_duplicates
        assert (df["close"] > 0).all()
    assert len(frames["daily_tr"]) > len(frames["weekly"]) > len(frames["monthly"])
    assert truth["n_dividends"] > 0


def test_the_split_is_an_event_not_a_price_jump():
    """The provider's convention: OHLC arrives already split-adjusted.

    The split therefore shows up only in the events column, and *neither* price series jumps.
    Reproducing that convention in the synthetic tape is what makes the audit's split check
    meaningful on the real one — see the note in ``strategy.split_check``.
    """
    frames, truth = data.synthetic_tape(n_years=8, split_at=0.5, seed=971)
    raw, tr = frames["daily_raw"], frames["daily_tr"]
    k = truth["split_index"]
    assert raw["stock_splits"].iloc[k] == 2.0
    assert 0.9 < raw["close"].iloc[k] / raw["close"].iloc[k - 1] < 1.1
    assert 0.9 < tr["close"].iloc[k] / tr["close"].iloc[k - 1] < 1.1


def test_corruption_plants_exactly_what_it_claims(clean_tape):
    frames, _ = clean_tape
    broken, planted = data.corrupt_tape(frames)
    assert set(planted) == {"dropped_session", "unapplied_split_from", "dropped_dividend"}
    assert len(broken["daily_tr"]) == len(frames["daily_tr"]) - 1
    assert (broken["daily_raw"]["dividends"] > 0).sum() == (
        frames["daily_raw"]["dividends"] > 0).sum() - 1


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_tape(n_years=3, seed=971)
    b, _ = data.synthetic_tape(n_years=3, seed=972)
    fp = data.fingerprint(a["daily_tr"])
    assert fp == data.fingerprint(a["daily_tr"]) and len(fp) == 12
    assert fp != data.fingerprint(b["daily_tr"])


def test_load_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load(data.TICKERS[0], "daily_tr", cache_dir=str(tmp_path))


def test_have_real_is_false_on_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


def test_universe_is_declared():
    assert len(data.TICKERS) == len(set(data.TICKERS)) >= 2
    assert pd.Timestamp(data.AS_OF) > pd.Timestamp(data.START)


@pytest.mark.skipif(not data.have_real(),
                    reason="no cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_has_all_four_flavours():
    for fl in data.FLAVOURS:
        df = data.load(data.TICKERS[0], fl)
        assert len(df) > 50 and df.index[-1] <= pd.Timestamp(data.AS_OF)
