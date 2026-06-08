"""The data layer: the synthetic universe and the panel builders."""

import numpy as np
import pandas as pd

from pairs_trading import data


def test_synthetic_universe_shape(universe):
    panel, frames, true_pairs = universe
    assert panel.shape[1] == len(frames) == 6 * 2 + 18      # twins (x2) + noise
    assert len(true_pairs) == 6
    assert list(panel.columns) == list(frames.keys())
    # every frame carries OHLCV
    for f in frames.values():
        assert {"Open", "High", "Low", "Close", "Volume"} <= set(f.columns)


def test_twin_legs_share_a_factor(universe):
    """A true twin's two legs co-move far tighter than two unrelated noise names."""
    panel, _, true_pairs = universe
    rets = panel.pct_change().dropna()
    p = true_pairs[0]
    twin_corr = rets[p.a].corr(rets[p.b])
    noise_corr = rets["NS00"].corr(rets["NS01"])
    assert twin_corr > 0.5
    assert twin_corr > noise_corr


def test_close_panel_no_forward_fill(frames):
    panel = data.close_panel(frames)
    # synthetic names all trade every session, so the panel is dense here
    assert not panel.isna().any().any()
    assert panel.index.is_monotonic_increasing


def test_dollar_volume_panel(frames):
    dvol = data.dollar_volume_panel(frames)
    assert (dvol.dropna() > 0).all().all()


def test_market_return_is_centered_small(panel):
    mkt = data.market_return(panel)
    assert isinstance(mkt, pd.Series)
    assert np.isfinite(mkt.dropna()).all()
