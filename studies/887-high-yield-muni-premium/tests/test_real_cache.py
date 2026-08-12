"""Real-tape sanity checks — SKIPPED when the (git-ignored) _cache/ is absent.

These never run on CI (no cache in a fresh checkout); they guard the local headline
numbers against silent drift when the cache IS present.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hy_muni import data, strategy as st  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(data.PRICES_CACHE),
                                reason="real _cache/ absent (offline/CI) — synthetic tests cover the machinery")


def test_common_sample_and_premium_sign():
    px = data.load_prices()
    m = st.monthly_returns(px, asof=data.AS_OF)
    mc = st.align_common(m, ["HYD", "MUB", "TFI", "HYG", "BIL"])
    assert len(mc) > 180                       # ~208 months of HYD era
    sp = st.premium_series(mc)
    h = st.hac_mean(sp.values)
    assert h["mean_bps"] > 0                    # HY-muni does out-earn IG-muni on average
    assert 1.0 < h["tstat"] < 2.5              # thin: below the desk bar (the honest finding)


def test_tax_equivalent_yield_beats_taxable_hy():
    px = data.load_prices()
    pr = data.load_price_only()
    m = st.monthly_returns(px, asof=data.AS_OF)
    inc = st.monthly_income(px, pr, asof=data.AS_OF)
    mc = st.align_common(m, ["HYD", "MUB", "HYG", "BIL"])
    iy = st.income_yields(inc.loc[mc.index], ["HYD", "HYG"])
    tey = st.tax_equivalent_yield(iy["HYD"], data.TOP_MARGINAL_RATE)
    assert tey > iy["HYG"]                      # the tax wrapper flips the yield ranking
