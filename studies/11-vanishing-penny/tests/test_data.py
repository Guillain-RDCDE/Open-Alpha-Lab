"""The data layer — synthetic generator shape/determinism, and the cache-only real path."""

import numpy as np
import pandas as pd

from prediction_arb import data


def test_synthetic_shape_and_clock(gap, truth):
    assert gap.shape == (6000, 48)
    assert truth.half_life_min == 6.0
    # 1-minute clock, monotone, named
    assert gap.index.freqstr == "min"
    assert gap.index.is_monotonic_increasing


def test_gap_sits_near_no_arb(gap):
    """Most of the time the book is at no-arb: the median |gap| is tiny vs the threshold."""
    assert float(np.abs(gap.to_numpy()).mean()) < 0.03
    # but real dislocations do open — some samples clear a fat 5% penny
    assert float(np.abs(gap.to_numpy()).max()) > 0.05


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_markets(seed=7, n_markets=4, n_steps=500)
    b, _ = data.synthetic_markets(seed=7, n_markets=4, n_steps=500)
    pd.testing.assert_frame_equal(a, b)
    c, _ = data.synthetic_markets(seed=8, n_markets=4, n_steps=500)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_build_gap_panel_is_cache_only(tmp_path):
    """With an empty cache, no market can be priced — empty frame, never a network call."""
    manifest = [data.MarketRef("trump-pa", "0xYES", "0xNO")]
    out = data.build_gap_panel(manifest, cache_dir=str(tmp_path))
    assert out.empty


def test_fetch_caches_and_builds_gap(tmp_path):
    """Inject a fake session: fetch writes parquet, build assembles g = 1 - (yes+no)."""
    class FakeResp:
        def __init__(self, hist):
            self._hist = hist

        def raise_for_status(self):
            pass

        def json(self):
            return {"history": self._hist}

    class FakeSession:
        def __init__(self, price):
            self.price = price

        def get(self, url, params, timeout):
            # two minute-spaced prints at a constant price
            return FakeResp([{"t": 1_700_000_000, "p": self.price},
                             {"t": 1_700_000_060, "p": self.price}])

    cache = str(tmp_path)
    data.fetch_prices_history("0xYES", 0, 1, cache_dir=cache, session=FakeSession(0.60))
    data.fetch_prices_history("0xNO", 0, 1, cache_dir=cache, session=FakeSession(0.34))
    assert data.cached_tokens(cache) == {"0xYES", "0xNO"}

    gap = data.build_gap_panel([data.MarketRef("m", "0xYES", "0xNO")], cache_dir=cache)
    # 1 - (0.60 + 0.34) = 0.06 guaranteed penny
    assert np.allclose(gap["m"].dropna().to_numpy(), 0.06)
