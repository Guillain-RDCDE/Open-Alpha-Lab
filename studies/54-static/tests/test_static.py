"""The synthetic panel is deterministic; idio-vol is estimated cleanly; the textbook long-low/short-high
trade pays only when high-idio-vol stocks really underperform; the null shows nothing; costs reduce the
hedge. All offline on the seeded synthetic world."""

import numpy as np

from static import data, strategy as st


def test_world_deterministic(puzzle_world):
    df, mkt, truth = puzzle_world
    df2, mkt2, _ = data.synthetic_panel(idiovol_premium=0.0030, seed=54)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert np.allclose(mkt.to_numpy(), mkt2.to_numpy())
    assert truth.has_premium


def test_textbook_trade_pays_when_puzzle_real(puzzle_world):
    df, mkt, _ = puzzle_world
    sig = st.idiovol_signal(df, mkt)
    h = st.stats(st.cross_section_hedge(df, sig, long_high=False))
    assert h["sharpe"] > 0.3       # low-idiovol wins on the control


def test_null_world_no_edge(null_world):
    df, mkt, _ = null_world
    sig = st.idiovol_signal(df, mkt)
    assert abs(st.stats(st.cross_section_hedge(df, sig, long_high=False))["sharpe"]) < 0.5


def test_idiovol_signal_finite_and_positive(puzzle_world):
    df, mkt, _ = puzzle_world
    iv = st.idiovol_signal(df, mkt).iloc[-1].dropna()
    assert (iv > 0).all() and np.isfinite(iv).all()


def test_costs_reduce_hedge(puzzle_world):
    df, mkt, _ = puzzle_world
    h = st.cross_section_hedge(df, st.idiovol_signal(df, mkt), long_high=False)
    assert st.net_of_cost(h, 20.0).mean() < h.mean()
