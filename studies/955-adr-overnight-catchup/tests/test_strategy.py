"""Inference primitives, backtest invariants, and the study's spine — all synthetic.

The spine: on a panel with a *planted* catch-up lag the stale-price regression must find a
positive loading on yesterday's home move, the home move must still predict once the ADR's
own move is controlled for, and the gross catch-up book must earn. On the null — where the
ADR prices the home move in full the same day — every one of those must be quiet.

The second spine, and the study's methodological point: a *residual-reversal* rule is not
specific. Planting pure bid-ask bounce with **zero** catch-up makes the residual rule fire
anyway, while the catch-up tests stay silent. The test-suite pins that separation down, so
the real-tape reading (residual reverts a little, home move adds nothing) is interpretable.

Everything runs offline on the deterministic synthetic panels; no cache, no network.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adr_catchup import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_newey_west_widens_with_autocorrelation():
    """An AR(1) series must get a smaller |t| from HAC than the naive i.i.d. t."""
    rng = np.random.default_rng(2)
    n, phi = 4000, 0.6
    e = rng.normal(0, 0.01, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    x += 0.0015
    assert abs(st.newey_west_t(x)) < abs(st.one_sample_t(x))


def test_hac_ols_recovers_planted_coefficients():
    rng = np.random.default_rng(3)
    n = 8000
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = 0.5 + 2.0 * x1 - 1.0 * x2 + rng.normal(0, 0.5, n)
    b, t = st.hac_ols(y, [x1, x2])
    assert b[0] == pytest.approx(0.5, abs=0.05)
    assert b[1] == pytest.approx(2.0, abs=0.05)
    assert b[2] == pytest.approx(-1.0, abs=0.05)
    assert t[1] > 10 and t[2] < -10


def test_hac_ols_finds_nothing_in_pure_noise():
    rng = np.random.default_rng(4)
    n = 6000
    b, t = st.hac_ols(rng.normal(0, 1, n), [rng.normal(0, 1, n)])
    assert abs(t[1]) < 3


def test_block_bootstrap_ci_brackets_the_point(planted):
    panel, _ = planted
    bt = st.book(panel, "catchup", cost_bps=0.0, borrow_bps=0.0)
    ci = st.block_bootstrap_ci(bt["gross"], n_boot=400, seed=955)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert ci["n_obs"] > 500


def test_summary_fields_and_drawdown_sign():
    r = pd.Series(np.full(500, 0.0004), index=pd.bdate_range("2010-01-04", periods=500))
    s = st.summary(r)
    assert s["sharpe"] > 0 and s["max_drawdown"] == pytest.approx(0.0)
    assert st.max_drawdown(pd.Series([0.1, -0.5, 0.1])) < -0.4


# --------------------------------------------------------------------------- #
# The residual and its one execution lag
# --------------------------------------------------------------------------- #
def test_residual_uses_only_past_data_for_beta(planted):
    """Perturbing the tail of the panel must not change any earlier residual."""
    panel, _ = data.synthetic_panel(n_names=3, n_years=5, seed=11)
    p1 = st.add_residual(panel)
    bumped = panel.copy()
    tail = bumped.index >= bumped.index[int(0.8 * len(bumped))]
    bumped.loc[tail, "a"] = bumped.loc[tail, "a"] * 5.0
    p2 = st.add_residual(bumped)
    cut = p1.index < p1.index[int(0.7 * len(p1))]
    assert np.allclose(p1.loc[cut, "e"].fillna(-9).to_numpy(),
                       p2.loc[cut, "e"].fillna(-9).to_numpy())


def test_residual_beta_is_shifted_by_one(planted):
    panel, _ = planted
    one = panel[panel["adr"] == panel["adr"].iloc[0]]
    # A 252-day rolling beta shifted one day is NaN for the first 252 rows.
    assert one["beta"].iloc[:252].isna().all()
    assert one["beta"].iloc[252:].notna().any()


def test_portfolio_has_exactly_one_execution_lag(planted):
    """The book must earn day t+1's return from weights formed at the close of day t."""
    panel, _ = planted
    bt = st.book(panel, "catchup", cost_bps=0.0, borrow_bps=0.0)
    sig = panel.pivot_table(index=panel.index, columns="adr", values="x")
    ret = panel.pivot_table(index=panel.index, columns="adr", values="a")
    w = np.sign(sig)
    w = w.div(w.abs().sum(axis=1), axis=0).fillna(0.0)
    manual = (w.shift(1) * ret).sum(axis=1, min_count=1).reindex(bt.index)
    assert np.allclose(bt["gross"].to_numpy(), manual.to_numpy(), equal_nan=True)
    # And the *contemporaneous* (cheating) version must differ, and be richer: trading
    # on the same day's home move captures the whole same-day loading as well.
    cheat = (w * ret).sum(axis=1, min_count=1).reindex(bt.index)
    assert not np.allclose(bt["gross"].to_numpy(), cheat.to_numpy())
    assert cheat.mean() > bt["gross"].mean() * 1.2


def test_direction_flips_the_sign_of_the_book(planted):
    panel, _ = planted
    up = st.catchup_portfolio(panel, signal_col="x", direction=+1,
                              cost_bps=0.0, borrow_bps=0.0)
    dn = st.catchup_portfolio(panel, signal_col="x", direction=-1,
                              cost_bps=0.0, borrow_bps=0.0)
    assert np.allclose(up["gross"].to_numpy(), -dn["gross"].to_numpy())
    with pytest.raises(ValueError):
        st.catchup_portfolio(panel, direction=0)


def test_portfolio_gross_exposure_is_one(planted):
    panel, _ = planted
    bt = st.book(panel, "catchup", cost_bps=0.0, borrow_bps=0.0)
    assert bt["short_exposure"].max() <= 1.0 + 1e-9
    assert abs(bt["net_exposure"]).max() <= 1.0 + 1e-9


def test_dollar_neutral_variant_is_actually_neutral(planted):
    panel, _ = planted
    bt = st.book(panel, "catchup", cost_bps=0.0, borrow_bps=0.0, dollar_neutral=True)
    assert bt["net_exposure"].abs().max() < 1e-9


def test_costs_and_borrow_monotonically_reduce_the_net(planted):
    panel, _ = planted
    means = [st.book(panel, "catchup", cost_bps=c, borrow_bps=0.0)["net"].mean()
             for c in (0.0, 5.0, 25.0)]
    assert means[0] > means[1] > means[2]
    borrows = [st.book(panel, "catchup", cost_bps=0.0, borrow_bps=b)["net"].mean()
               for b in (0.0, 50.0, 300.0)]
    assert borrows[0] > borrows[1] > borrows[2]


def test_borrow_is_charged_only_on_the_short_leg(planted):
    panel, _ = planted
    bt = st.book(panel, "catchup", cost_bps=0.0, borrow_bps=100.0)
    expected = bt["short_exposure"] * 100.0 * 1e-4 / st.TRADING_DAYS
    assert np.allclose(bt["borrow"].to_numpy(), expected.to_numpy())
    assert bt["short_exposure"].mean() > 0.3   # a genuinely two-sided book


def test_zero_cost_zero_borrow_leaves_gross_untouched(planted):
    panel, _ = planted
    bt = st.book(panel, "catchup", cost_bps=0.0, borrow_bps=0.0)
    assert np.allclose(bt["gross"].to_numpy(), bt["net"].to_numpy())


def test_breakeven_cost_matches_the_sweep(planted):
    panel, _ = planted
    be = st.breakeven_cost_bps(panel, "catchup")
    at_be = st.book(panel, "catchup", cost_bps=be, borrow_bps=0.0)["net"].mean()
    assert abs(at_be) < 1e-9


# --------------------------------------------------------------------------- #
# The study's spine — the detector fires on a planted lag
# --------------------------------------------------------------------------- #
def test_planted_lag_is_recovered_by_the_stale_regression(planted):
    panel, truth = planted
    res = st.stale_catchup_test(panel)
    assert res["beta_lag"] > 0.05
    assert res["t_lag"] > 4
    assert res["beta_same_day"] > res["beta_lag"]


def test_planted_home_move_survives_the_own_move_control(planted):
    """The discriminating test: with a real lag, x_t predicts beyond a_t."""
    panel, _ = planted
    disc = st.pooled_predictive(panel, xcols=("a", "x"))
    assert disc["coef_x"] > 0
    assert disc["t_x"] > 4


def test_planted_lag_book_earns_gross(planted):
    panel, _ = planted
    s = st.summary(st.book(panel, "catchup", cost_bps=0.0, borrow_bps=0.0)["gross"])
    assert s["sharpe"] > 0.5 and s["tstat"] > 3


def test_planted_lag_beats_the_no_home_data_control(planted):
    """The catch-up book must beat a control that never looks at the home tape."""
    panel, _ = planted
    s = st.summary(st.book(panel, "catchup", cost_bps=0.0, borrow_bps=0.0)["gross"])
    ctrl = st.summary(st.book(panel, "reversal", cost_bps=0.0, borrow_bps=0.0)["gross"])
    assert s["sharpe"] > ctrl["sharpe"] + 0.4


# --------------------------------------------------------------------------- #
# Is the loading a loading, or a handful of enormous days?
# --------------------------------------------------------------------------- #
def _tail_only_panel(n_days=5000, n_names=2, seed=7):
    """A panel where the catch-up lag exists ONLY after the biggest home moves.

    Everything else is identical to the null: on an ordinary day the ADR prices the home
    move in full, same day. This is the shape :func:`strategy.tail_sensitivity` has to be
    able to tell apart from a genuine linear loading.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2004-01-05", periods=n_days)
    frames = []
    for i in range(n_names):
        x = rng.standard_t(3, n_days) * 0.008
        big = np.abs(x) > np.quantile(np.abs(x), 0.98)
        lag_share = np.where(big, 0.60, 0.0)
        x_lag = np.concatenate([[0.0], x[:-1]])
        share_lag = np.concatenate([[0.0], lag_share[:-1]])
        a = 0.5 * ((1 - lag_share) * x + share_lag * x_lag) + rng.normal(0, 0.013, n_days)
        frames.append(pd.DataFrame(
            {"a": a, "h": x, "f": 0.0, "x": x, "adr": f"T{i}", "region": "T"},
            index=pd.DatetimeIndex(dates, name="date")))
    return pd.concat(frames).sort_values(["date", "adr"])


def test_tail_sensitivity_spares_a_linear_planted_lag(planted):
    """The knife must not cut a genuine, linear loading — otherwise it proves nothing."""
    panel, _ = planted
    tbl = st.tail_sensitivity(panel).set_index(["mode", "q"])
    full = tbl.loc[("full", 1.0), "beta_lag"]
    for (mode, q), row in tbl.iterrows():
        if mode == "full":
            continue
        assert abs(row["beta_lag"] - full) < 0.20 * abs(full), (mode, q, row["beta_lag"])
        assert row["t_lag"] > 4


def test_tail_sensitivity_flags_a_tail_only_lag():
    """And it must fire when the loading exists only on the biggest home moves."""
    panel = _tail_only_panel()
    tbl = st.tail_sensitivity(panel).set_index(["mode", "q"])
    full = tbl.loc[("full", 1.0), "beta_lag"]
    assert full > 0.02                                   # the pooled reading looks alive
    assert tbl.loc[("trim", 0.99), "beta_lag"] < 0.5 * full   # ...and the body is empty
    assert tbl.loc[("trim", 0.95), "beta_lag"] < 0.5 * full


def test_lag_bucket_table_is_monotone_on_a_planted_lag(planted):
    panel, _ = planted
    tbl = st.lag_bucket_table(panel)
    assert len(tbl) == 5
    assert tbl["x_lag_mean"].is_monotonic_increasing
    assert tbl["a_mean_bps"].iloc[-1] > tbl["a_mean_bps"].iloc[0] + 5.0


def test_lag_bucket_table_is_flat_on_the_null(null_panel):
    panel, _ = null_panel
    tbl = st.lag_bucket_table(panel)
    spread = tbl["a_mean_bps"].iloc[-1] - tbl["a_mean_bps"].iloc[0]
    assert abs(spread) < 5.0


def test_consecutive_only_keeps_most_rows_and_agrees(planted):
    """On a gap-free synthetic panel the restriction is a near-no-op."""
    panel, _ = planted
    full = st.stale_catchup_test(panel)
    strict = st.consecutive_only(panel)
    assert strict["share_kept"] > 0.95
    assert strict["beta_lag"] == pytest.approx(full["beta_lag"], abs=0.01)


def test_beta_lag_bootstrap_brackets_and_separates(planted, null_panel):
    p, _ = planted
    ci = st.beta_lag_bootstrap(p, n_boot=300, seed=955)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert ci["ci_low"] > 0 and ci["frac_le_zero"] == 0.0
    assert ci["point"] == pytest.approx(ci["point_pooled"], abs=0.02)
    n, _ = null_panel
    cin = st.beta_lag_bootstrap(n, n_boot=300, seed=955)
    assert cin["ci_low"] < 0 < cin["ci_high"]


# --------------------------------------------------------------------------- #
# ... and stays quiet on the null
# --------------------------------------------------------------------------- #
def test_null_has_no_stale_loading(null_panel):
    panel, _ = null_panel
    res = st.stale_catchup_test(panel)
    assert abs(res["t_lag"]) < 3
    assert abs(res["beta_lag"]) < 0.05


def test_null_home_move_does_not_predict(null_panel):
    panel, _ = null_panel
    disc = st.pooled_predictive(panel, xcols=("a", "x"))
    assert abs(disc["t_x"]) < 3


def test_null_book_earns_nothing_gross(null_panel):
    panel, _ = null_panel
    bt = st.book(panel, "catchup", cost_bps=0.0, borrow_bps=0.0)
    assert abs(st.summary(bt["gross"])["tstat"]) < 3


# --------------------------------------------------------------------------- #
# The confound: the residual rule is NOT specific to catch-up
# --------------------------------------------------------------------------- #
def test_pure_bounce_fires_the_residual_rule_with_zero_catchup():
    """Plant bid-ask bounce and NO stale information: the residual rule still fires.

    This is why the study refuses to read a negative residual slope as evidence of ADR
    catch-up. The stale-catchup regression — which asks whether *yesterday's home move*
    still loads — is immune to bounce and correctly stays silent.
    """
    panel, _ = data.synthetic_panel(n_names=6, n_years=8, signal_strength=0.0,
                                    bounce_vol=0.006, seed=955)
    p = st.add_residual(panel)
    fm = st.fama_macbeth(p, xcols=("e",))
    assert fm["slope_e"] < 0 and fm["t_e"] < -3           # the residual rule fires...
    assert abs(st.stale_catchup_test(p)["t_lag"]) < 3     # ...on zero stale information


def test_bounce_biases_the_own_move_control_upward():
    """Document the one place bounce *helps* the catch-up story, so we never over-read it.

    Regressing ``a_{t+1}`` on ``a_t`` and ``x_t`` together, pure bounce manufactures a
    spuriously **positive** loading on ``x_t``: the negative own-move coefficient is
    applied to the ``b x_t`` slice inside ``a_t``, and ``x_t`` enters to undo it. So this
    secondary test is biased *in favour* of catch-up — which is exactly why a zero reading
    on the real tape is such damning evidence.
    """
    panel, _ = data.synthetic_panel(n_names=6, n_years=8, signal_strength=0.0,
                                    bounce_vol=0.006, seed=955)
    p = st.add_residual(panel)
    disc = st.pooled_predictive(p, xcols=("a", "x"))
    assert disc["coef_a"] < 0
    assert disc["coef_x"] > 0 and disc["t_x"] > 3   # spurious, with zero planted catch-up


def test_bounce_does_not_fool_the_catchup_book():
    panel, _ = data.synthetic_panel(n_names=6, n_years=8, signal_strength=0.0,
                                    bounce_vol=0.006, seed=955)
    p = st.add_residual(panel)
    s = st.summary(st.book(p, "catchup", cost_bps=0.0, borrow_bps=0.0)["gross"])
    assert abs(s["tstat"]) < 3


def test_null_is_quiet_across_seeds():
    """Across independent null draws the stale loading is centred on zero."""
    ts = np.array([
        st.stale_catchup_test(
            st.add_residual(data.synthetic_panel(n_names=6, n_years=6,
                                                 signal_strength=0.0, seed=955 + s)[0])
        )["t_lag"]
        for s in range(5)
    ])
    assert abs(ts.mean()) < 1.5
    assert (np.abs(ts) >= 3.0).sum() <= 1


def test_signal_strength_monotonically_raises_the_detected_lag():
    betas = []
    for ss in (0.0, 0.5, 1.0):
        p = st.add_residual(data.synthetic_panel(n_names=6, n_years=6,
                                                 signal_strength=ss, seed=955)[0])
        betas.append(st.stale_catchup_test(p)["beta_lag"])
    assert betas[0] < betas[1] < betas[2]


# --------------------------------------------------------------------------- #
# Robustness helpers return well-formed output
# --------------------------------------------------------------------------- #
def test_era_cut_returns_both_halves(planted):
    panel, _ = planted
    split = panel.index[len(panel) // 2].strftime("%Y-%m-%d")
    eras = st.era_cut(panel, split=split)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["beta_lag"]) and np.isfinite(e["gross_sharpe"])


def test_region_cut_covers_every_region(planted):
    panel, _ = planted
    tbl = st.region_cut(panel)
    assert set(tbl.index) == set(panel["region"].unique())
    assert {"beta_lag", "t_lag", "fm_slope_e", "coef_x_ctrl"}.issubset(tbl.columns)


def test_cost_and_borrow_sweeps_are_monotone(planted):
    panel, _ = planted
    cs = st.cost_sweep(panel, grid=(0.0, 5.0, 25.0), borrow_bps=0.0)
    assert cs["ann_return"].is_monotonic_decreasing
    bs = st.borrow_sweep(panel, grid=(0.0, 50.0, 300.0), cost_bps=0.0)
    assert bs["ann_return"].is_monotonic_decreasing


def test_fast_and_slow_fama_macbeth_agree(planted):
    """The single-regressor fast path must equal the general loop, date by date."""
    panel, _ = planted
    fast = st.fama_macbeth(panel, xcols=("e",))
    df = panel.dropna(subset=["a_next", "e"])
    slopes, dates = [], []
    for dt, g in df.groupby(level=0, sort=True):
        if len(g) < 4:
            continue
        b, _ = st.hac_ols(g["a_next"].to_numpy(), [g["e"].to_numpy()], lags=1)
        slopes.append(b[1]); dates.append(dt)
    slow = pd.Series(slopes, index=pd.DatetimeIndex(dates))
    assert np.allclose(fast["series_e"].to_numpy(), slow.to_numpy(), atol=1e-10)


def test_per_name_betas_shape(planted):
    panel, truth = planted
    tbl = st.per_name_betas(panel)
    assert len(tbl) == truth["n_names"]
    assert (tbl["beta_same_day"] > 0).all()


def test_synthetic_detect_separates_planted_from_null():
    pl = st.synthetic_detect(data.synthetic_panel(n_names=6, n_years=6,
                                                  signal_strength=1.0, seed=955)[0])
    nl = st.synthetic_detect(data.synthetic_panel(n_names=6, n_years=6,
                                                  signal_strength=0.0, seed=955)[0])
    assert pl["t_lag"] > 4 > abs(nl["t_lag"])
    assert pl["gross_sharpe"] > nl["gross_sharpe"]
