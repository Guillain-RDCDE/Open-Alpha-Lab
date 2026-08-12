"""Real-cache tests — skipped entirely when the git-ignored _cache/ is absent (CI-safe)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cppi import data, strategy as st  # noqa: E402

_HAVE = data.have_real()
pytestmark = pytest.mark.skipif(not _HAVE, reason="real _cache/ absent (offline / CI)")


def test_load_prices_shape_and_columns():
    px = data.load_prices()
    assert list(px.columns) == list(data.TICKERS)
    assert not px.empty
    assert px.index.min().year == 2007        # BIL inception bounds the join
    assert px.index.max() <= __import__("pandas").Timestamp(data.AS_OF)


def test_real_race_runs_and_is_sane():
    px = data.load_prices()
    ret = st.to_returns(px)
    r = st.race(ret, multiplier=5.0, floor_frac=0.80)
    # buy-and-hold SPY excess Sharpe sits in a believable band on this window.
    assert 0.3 < r["buy_hold"]["sharpe"] < 0.9
    # The mechanical floor genuinely bounds the CPPI drawdown far above buy-and-hold's.
    assert r["dd_cppi"] > r["dd_bh"] + 0.20
    assert r["n_breach"] == 0                  # SPY never fell > 20% in a single day here
    # But it is NOT a Sharpe edge: CPPI underperforms buy-and-hold on excess Sharpe.
    assert r["sharpe_vs_bh"] < 0.0
