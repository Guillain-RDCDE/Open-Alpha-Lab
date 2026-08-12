"""Real-cache tests — skipped entirely when the git-ignored _cache/ is absent (CI-safe)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cash_call import data, strategy as st  # noqa: E402

_HAVE = data.have_real()
pytestmark = pytest.mark.skipif(not _HAVE, reason="real _cache/ absent (offline / CI)")


def test_load_prices_shape_and_columns():
    px = data.load_prices()
    assert list(px.columns) == ["SPY", "BIL", "IRX"]
    assert not px.empty
    assert px.index.min().year == 2007        # BIL inception bounds the join
    assert px.index.max() <= __import__("pandas").Timestamp(data.AS_OF)


def test_real_race_runs_and_is_sane():
    px = data.load_prices()
    r = st.race(px)
    # buy-and-hold SPY excess Sharpe sits in a believable band on this window.
    assert 0.35 < r["buy_hold"]["sharpe"] < 0.75
    # Capital protection is real: 90/10's drawdown is far shallower than buy-and-hold's.
    assert r["dd_tt"] > r["dd_bh"] + 0.20
    # But there is NO robust risk-adjusted edge: the Sharpe roughly ties (small, near zero) and the
    # convexity alpha is insignificant.
    assert -0.4 < r["sharpe_vs_bh"] < 0.1
    assert abs(r["t_alpha"]) < 2.0


def test_real_premium_markup_kills_the_tie():
    # At a realistic variance-risk-premium markup the fair-price parity flips clearly negative.
    px = data.load_prices()
    r_fair = st.race(px, prem_mult=1.0)
    r_rich = st.race(px, prem_mult=1.5)
    assert r_rich["sharpe_vs_bh"] < r_fair["sharpe_vs_bh"] - 0.15
    assert r_rich["sharpe_vs_bh"] < -0.15
