"""The synthetic world is deterministic; E/P forecasts equities when it should and not in the null; the
Fed signal (E/P − yield) never beats E/P alone (the bond term is inert by construction); and the timing
rule and signal are computed correctly. All offline on the seeded synthetic world."""

import numpy as np

from paper_moon import data, strategy as st


def test_world_deterministic(ep_world):
    df, truth = ep_world
    df2, _ = data.synthetic_world(ep_predicts=0.6, seed=47)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.ep_informative


def test_ep_forecasts_when_it_should(ep_world):
    df, _ = ep_world
    assert st.predictive_corr(df["ep"], df["tr"], horizon=1) > 0.05   # E/P carries information


def test_null_world_no_forecast(null_world):
    df, _ = null_world
    assert abs(st.predictive_corr(df["ep"], df["tr"], horizon=1)) < 0.08


def test_bond_term_is_inert(ep_world):
    """The Fed signal adds the irrelevant yield, so it cannot out-forecast E/P alone."""
    df, _ = ep_world
    pc_ep = st.predictive_corr(df["ep"], df["tr"], horizon=1)
    pc_fed = st.predictive_corr(st.fed_signal(df["ep"], df["y10"]), df["tr"], horizon=1)
    assert pc_ep >= pc_fed - 1e-9     # E/P at least as good — the bond comparison adds nothing


def test_signal_and_timing_mechanics(ep_world):
    df, _ = ep_world
    sig = st.fed_signal(df["ep"], df["y10"])
    assert np.allclose(sig, df["ep"] - df["y10"])
    timing = st.fed_timing(df["tr"], df["bond"], sig)
    # when last month's signal was negative the rule holds bonds, not equities
    pos = (sig.shift(1) > 0).reindex(timing.index)
    assert np.allclose(timing[~pos], df["bond"].shift(1).reindex(timing.index)[~pos])
