"""The hedging-pressure engine recovers the baked premium (and finds none on the null), the book pays
when the premium is real and not otherwise, and the alignment helper carries positioning forward."""

import numpy as np
import pandas as pd

from hedgers_toll import data, hedging, strategy, decompose, extension


def test_deterministic_and_premium(prem):
    r, hp, truth = prem
    r2, _, _ = data.synthetic_commodities(hp_strength=0.0045, seed=29)
    assert np.allclose(r.to_numpy(), r2.to_numpy())
    assert truth.has_premium


def test_hedging_premium_recovers(prem_ret, prem_hp):
    pr = hedging.hedging_premium(prem_ret, prem_hp)
    assert pr["mean_ic"] > 0.02 and pr["ic_t"] > 3.0
    assert pr["premium_present"]


def test_hedging_premium_flat_on_null(null_ret, null_hp):
    pr = hedging.hedging_premium(null_ret, null_hp)
    assert abs(pr["ic_t"]) < 3.0


def test_book_pays_with_premium(prem_ret, prem_hp):
    cmp = strategy.compare(prem_ret, prem_hp, cost_bps=10.0)
    assert cmp["long_short"]["sharpe"] > 0.5
    pt = decompose.premium_tstat(prem_ret, prem_hp, cost_bps=10.0)
    assert pt["t_stat"] > 3.0


def test_book_no_edge_on_null(null_ret, null_hp):
    cmp = strategy.compare(null_ret, null_hp, cost_bps=10.0)
    assert cmp["long_short"]["sharpe"] < 0.5


def test_align_hp_is_causal():
    """align_hp forward-fills weekly positioning onto the return index, never using a future reading."""
    import numpy as np
    idx_ret = pd.date_range("2020-01-07", periods=10, freq="W-TUE")
    idx_hp = pd.date_range("2020-01-08", periods=10, freq="W-WED")   # offset, non-matching dates
    ret = pd.DataFrame(np.zeros((10, 2)), index=idx_ret, columns=["A", "B"])
    hp = pd.DataFrame(np.arange(20).reshape(10, 2).astype(float), index=idx_hp, columns=["A", "B"])
    a = data.align_hp(hp, ret)
    assert list(a.index) == list(idx_ret)             # reindexed onto the return grid
    assert a.iloc[0].isna().all()                      # first return week precedes the first hp reading


def test_window_sweep_and_leg_split(prem_ret, prem_hp):
    sw = extension.window_sweep(prem_ret, prem_hp, cost_bps=10.0)
    assert "ls_sharpe" in sw.columns and len(sw) >= 3
    ls = extension.leg_split(prem_ret, prem_hp, cost_bps=10.0)
    assert {"long_only_top_sharpe", "long_short_sharpe", "basket_sharpe"}.issubset(ls)
