"""GEM is deterministic; relative & absolute momentum pick the right door; the decision is causal
(lagged); costs only hurt; the crash filter protects on the synthetic premium world; and GEM invents
no edge in the null. All offline on the seeded synthetic world — no network, no real data."""

import numpy as np
import pandas as pd

from paper_tiger import data, strategy as st


def test_world_deterministic(premium_world):
    p, _, truth = premium_world
    p2, _, _ = data.synthetic_world(premium=0.12, crash=True, seed=40)
    assert np.allclose(p.to_numpy(), p2.to_numpy())  # seeded → reproducible
    assert truth.has_premium


def test_relative_momentum_picks_stronger_equity():
    """When US has clearly out-run INTL and both beat the T-bill, the door is US."""
    idx = pd.date_range("2000-01-31", periods=20, freq="ME")
    prices = pd.DataFrame(
        {
            "US": np.linspace(100, 200, 20),    # strong uptrend
            "INTL": np.linspace(100, 110, 20),  # weak uptrend
            "BOND": np.linspace(100, 102, 20),
        },
        index=idx,
    )
    tbill = pd.Series(0.0, index=idx)  # zero cash hurdle → absolute filter never binds here
    hold = st.gem_hold(prices, tbill, lookback=12).dropna()
    assert (hold == "US").all()


def test_absolute_momentum_steps_into_bonds_on_a_crash():
    """When BOTH equities fall below the T-bill hurdle, the book must sit in BOND."""
    idx = pd.date_range("2000-01-31", periods=20, freq="ME")
    prices = pd.DataFrame(
        {
            "US": np.linspace(200, 100, 20),    # both equities falling…
            "INTL": np.linspace(200, 120, 20),  # …so 12m momentum is negative
            "BOND": np.linspace(100, 105, 20),
        },
        index=idx,
    )
    tbill = pd.Series(0.001, index=idx)  # small positive cash return → equities trail it
    hold = st.gem_hold(prices, tbill, lookback=12).dropna()
    assert (hold == "BOND").all()


def test_decision_is_causal():
    """gem_hold is lagged: flipping the LAST month's price cannot change any earlier decision."""
    p, tb, _ = data.synthetic_world(premium=0.12, crash=True, seed=7)
    h1 = st.gem_hold(p, tb, lookback=12)
    p2 = p.copy()
    p2.iloc[-1] *= 1.5  # shock only the last month
    h2 = st.gem_hold(p2, tb, lookback=12)
    assert h1.iloc[:-1].equals(h2.iloc[:-1])


def test_costs_only_hurt(premium_world):
    p, tb, _ = premium_world
    gross = (1 + st.gem_returns(p, tb, cost_bps=0.0)).prod()
    net = (1 + st.gem_returns(p, tb, cost_bps=50.0)).prod()
    assert net < gross  # paying to switch can only reduce the compounded return


def test_crash_filter_protects_on_premium_world(premium_world):
    """The whole point: GEM's drawdown is shallower than buy-and-hold equity."""
    p, tb, _ = premium_world
    gem = st.gem_returns(p, tb, cost_bps=20.0)
    bh = st.buy_and_hold(p, "US").loc[gem.index]
    # max_drawdown is negative; "shallower" means closer to zero (greater)
    assert st.summary(gem)["max_drawdown"] > st.summary(bh)["max_drawdown"]
    assert st.summary(gem)["sharpe"] > 0.0


def test_no_invented_edge_in_the_null(null_world):
    """In a world with no premium, GEM must NOT beat a plain fixed blend on Sharpe."""
    p, tb, _ = null_world
    gem = st.gem_returns(p, tb, cost_bps=20.0)
    blend = st.blend(p).loc[gem.index]
    assert st.summary(gem)["sharpe"] <= st.summary(blend)["sharpe"] + 0.20


def test_blend_is_weighted_average(premium_world):
    p, _, _ = premium_world
    rets = st.to_returns(p)
    b = st.blend(p, weights={"US": 0.6, "BOND": 0.4})
    expected = (0.6 * rets["US"] + 0.4 * rets["BOND"]).dropna()
    assert np.allclose(b.to_numpy(), expected.loc[b.index].to_numpy())


def test_summary_excess_sharpe_below_raw(premium_world):
    """With a positive cash rate, the excess-of-cash Sharpe must sit below the raw Sharpe
    (and only the Sharpe moves — CAGR/vol/drawdown stay raw)."""
    p, tb, _ = premium_world
    gem = st.gem_returns(p, tb, cost_bps=20.0)
    raw, ex = st.summary(gem), st.summary(gem, rf=tb.reindex(gem.index))
    assert ex["sharpe"] < raw["sharpe"]
    for k in ("cagr", "vol_ann", "max_drawdown"):
        assert raw[k] == ex[k]
    # scalar convention: annualised rate, same direction
    assert st.summary(gem, rf=0.02)["sharpe"] < raw["sharpe"]


def test_spanning_alpha_zero_on_a_static_mix(premium_world):
    """A fixed mix of the factors IS spanned: alpha ~ 0, beta = the weights, R² = 1."""
    p, _, _ = premium_world
    rets = st.to_returns(p).dropna()
    fac = rets[["US", "BOND"]]
    mix = 0.6 * rets["US"] + 0.4 * rets["BOND"]
    sp = st.spanning_alpha(mix, fac)
    assert abs(sp["alpha_bps_m"]) < 1e-6
    assert abs(sp["betas"]["US"] - 0.6) < 1e-9 and abs(sp["betas"]["BOND"] - 0.4) < 1e-9
    assert sp["r2"] > 0.999999


def test_spanning_alpha_detects_an_injected_constant(premium_world):
    """Add a deliberate 50 bp/month on top of a spanned mix: the intercept must find it, t >> 2."""
    p, _, _ = premium_world
    rets = st.to_returns(p).dropna()
    fac = rets[["US", "BOND"]]
    juiced = 0.6 * rets["US"] + 0.4 * rets["BOND"] + 0.0050
    sp = st.spanning_alpha(juiced, fac)
    assert abs(sp["alpha_bps_m"] - 50.0) < 1e-6
    assert sp["tstat"] > 2.0
