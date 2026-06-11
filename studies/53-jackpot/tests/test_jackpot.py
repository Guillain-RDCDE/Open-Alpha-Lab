"""The synthetic panel is deterministic; MAX measures what it should; the textbook long-low/short-high
trade pays only when high-MAX stocks really underperform; the null shows nothing; costs reduce the
hedge. All offline on the seeded synthetic world."""

import numpy as np

from jackpot import data, strategy as st


def test_world_deterministic(lottery_world):
    df, truth = lottery_world
    df2, _ = data.synthetic_panel(lottery_premium=0.0030, seed=53)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.has_premium


def test_textbook_trade_pays_when_lottery_real(lottery_world):
    df, _ = lottery_world
    sig = st.max_signal(df)
    h = st.stats(st.cross_section_hedge(df, sig, long_high=False))   # long low-MAX
    assert h["sharpe"] > 0.3       # the lottery effect, recovered on the control


def test_null_world_no_edge(null_world):
    df, _ = null_world
    sig = st.max_signal(df)
    assert abs(st.stats(st.cross_section_hedge(df, sig, long_high=False))["sharpe"]) < 0.5


def test_max_signal_is_finite_and_ordered(lottery_world):
    df, _ = lottery_world
    sig = st.max_signal(df).dropna(how="all")
    assert np.isfinite(sig.to_numpy()[np.isfinite(sig.to_numpy())]).all()
    # MAX (mean of top daily returns) should exceed the stock's overall mean daily return
    assert (sig.iloc[-1] > df.mean()).mean() > 0.8


def test_costs_reduce_hedge(lottery_world):
    df, _ = lottery_world
    h = st.cross_section_hedge(df, st.max_signal(df), long_high=False)
    assert st.net_of_cost(h, 20.0).mean() < h.mean()
