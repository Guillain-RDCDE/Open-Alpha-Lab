"""Real-tape sanity checks — SKIPPED when the git-ignored _cache/ is absent (e.g. on CI).

These never fetch; they only read the cached parquet/csv if a prior `data.fetch()` built it.
The offline synthetic suite in test_opt_roll.py is the machinery proof.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from opt_roll import data, strategy as st  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(data.PRICES_CACHE), reason="real cache absent (offline / CI)"
)


def test_common_window_and_columns():
    prices = data.load_prices()
    rets = data.monthly_returns(prices, asof=data.AS_OF)
    ex = st.excess_frame(rets, cash=data.CASH)
    common = st.common_sample(ex, ["USCI"] + data.BENCHMARKS)
    assert 150 < len(common) < 220                     # ~190 months
    assert str(common.index.max().date()) <= data.AS_OF
    for c in ["USCI"] + data.BENCHMARKS:
        assert c in ex.columns


def test_optimized_wins_full_sample_but_not_significantly():
    prices = data.load_prices()
    ex = st.excess_frame(data.monthly_returns(prices, asof=data.AS_OF), cash=data.CASH)
    r = st.sharpe_race(ex, "USCI", "DBC")
    assert r["sharpe_adv"] > 0.0                        # right sign
    assert abs(r["t_diff"]) < 2.0                       # not significant on the difference
    assert r["adv_ci_lo"] < 0.0 < r["adv_ci_hi"]        # bootstrap CI includes zero


def test_era_sign_flip_2016_2020():
    prices = data.load_prices()
    ex = st.excess_frame(data.monthly_returns(prices, asof=data.AS_OF), cash=data.CASH)
    eras = [("recovery 2016-2020", "2016-01", "2020-12")]
    e = st.era_race(ex, "USCI", "DBC", eras)[0]
    assert e["diff_ann_pct"] < 0.0                      # USCI underperforms in the recovery era
    assert e["t_diff"] < -2.0                           # significantly so
