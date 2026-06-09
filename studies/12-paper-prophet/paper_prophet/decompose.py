"""The teardown — separate the *forecast* from the *sizing*, then charge it costs.

Everything here consumes a :class:`paper_prophet.stack.WalkForward` and answers one question in
five numbers:

    1. **Is the direction real?**  ``directional_edge`` — the per-day bet ``sign·ret``, HAC-t'd,
       and the hit-rate vs the 50% coin.
    2. **Is the Sharpe the forecast or the sizing?**  ``sharpe_decomposition`` — the full stack vs
       the *same GARCH sizing on a constant-long signal* (pure vol-targeting) vs the *forecast with
       flat sizing* vs plain buy-and-hold. The ARIMA increment is stack − vol-targeting.
    3. **Is the edge just managed beta?**  ``alpha_vs_beta`` — regress the stack on the market and
       on the vol-managed-market factor; what survives is genuine alpha.
    4. **Does it pay?**  ``cost_sweep`` — charge the forecast's turnover a realistic round-trip and
       watch the ARIMA increment cross zero.
    5. **Does it "win every trade"?**  ``win_rate`` — the per-day hit rate on the sized bet.

Plus ``in_sample_inflation`` — the control the article itself warns about: fit once on the whole
series and grade in-sample ("a curve fit with a label") vs the honest walk-forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.analytics import mean_tstat_hac
from quantlab.stats import annualized_sharpe

from .stack import WalkForward, generate_signals


def directional_edge(wf: WalkForward) -> dict:
    """Hit-rate vs 50% and the HAC-t on the per-day directional bet ``sign·ret``.

    The directional bet strips out the sizing: it is +ret when the sign was right, −ret when wrong.
    Its mean is the forecast's *raw* edge; HAC-t (Newey–West) handles the mild autocorrelation. The
    hit-rate is graded only on days with a non-zero realised return.
    """
    f = wf.frame
    edge = (f["sign"] * f["ret"]).rename("dir_edge")  # ret is in percent (article's scale)
    nz = f[f["ret"] != 0.0]
    hits = (np.sign(nz["forecast"]) == np.sign(nz["ret"])).mean()
    hac = mean_tstat_hac(edge)
    return {
        "hit_rate": float(hits),
        "n_graded": int(nz.shape[0]),
        # ret is already ×100 (percent); mean_tstat_hac's mean_bps assumes a raw fraction, so divide
        # by 100 to recover true basis points of daily directional edge. The HAC t-stat is
        # scale-invariant and is the decisive number.
        "dir_edge_bps_per_day": float(hac["mean_bps"]) / 100.0,
        "dir_edge_hac_t": float(hac["tstat"]),
        "active_frac": float((f["sign"] != 0).mean()),
    }


def _legs(wf: WalkForward) -> pd.DataFrame:
    """The four comparable daily return streams, in percent (the article's scale)."""
    f = wf.frame
    return pd.DataFrame(
        {
            # full stack: direction × sizing
            "stack": f["sign"] * f["size"] * f["ret"],
            # pure vol-targeting: constant-long × the *same* GARCH sizing
            "voltgt": 1.0 * f["size"] * f["ret"],
            # forecast with flat sizing: direction only, no GARCH
            "flat_forecast": f["sign"] * f["ret"],
            # plain long
            "buyhold": f["ret"],
        },
        index=f.index,
    )


def _increment_ci(stack: pd.Series, voltgt: pd.Series, n_boot: int = 2000,
                  alpha: float = 0.05, seed: int = 0) -> dict:
    """Joint bootstrap CI for the *difference of Sharpes* Sharpe(stack) − Sharpe(voltgt).

    Resamples the two daily return streams together (same day indices) so their dependence is
    preserved, recomputes both annualised Sharpes per resample and takes the difference. Returns
    the percentile interval and the share of resamples where the increment is ≤ 0.
    """
    a = np.asarray(stack, dtype=float)
    b = np.asarray(voltgt, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = annualized_sharpe(pd.Series(a[idx])) - annualized_sharpe(pd.Series(b[idx]))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci_low": float(lo), "ci_high": float(hi), "frac_le_zero": float((diffs <= 0).mean())}


def sharpe_decomposition(wf: WalkForward, seed: int = 0) -> dict:
    """Attribute the stack's Sharpe to the forecast vs the sizing.

    Returns annualised Sharpe for each leg, the fraction of the stack's Sharpe reproduced by the
    constant-long vol-targeting control, and the **ARIMA increment** — the difference of Sharpes
    ``Sharpe(stack) − Sharpe(vol-targeting)`` (the same quantity the cost sweep tracks), with a
    joint-bootstrap read on whether it is distinguishable from zero.
    """
    legs = _legs(wf)
    sharpes = {c: annualized_sharpe(legs[c]) for c in legs.columns}
    s_stack, s_voltgt = sharpes["stack"], sharpes["voltgt"]
    increment = s_stack - s_voltgt
    boot = _increment_ci(legs["stack"], legs["voltgt"], seed=seed)
    return {
        "sharpe_stack": s_stack,
        "sharpe_voltgt": s_voltgt,
        "sharpe_flat_forecast": sharpes["flat_forecast"],
        "sharpe_buyhold": sharpes["buyhold"],
        "voltgt_reproduces_frac": float(s_voltgt / s_stack) if s_stack else np.nan,
        "arima_increment_sharpe": float(increment),
        "arima_increment_ci": (boot["ci_low"], boot["ci_high"]),
        "arima_increment_frac_negative": boot["frac_le_zero"],
    }


def alpha_vs_beta(wf: WalkForward) -> dict:
    """How much of the stack is managed beta? Regress on the market and on the vol-managed market.

    Two OLS regressions on daily returns:
        stack ~ a + b·buyhold              → b = exposure to plain SPY, a = alpha over plain beta
        stack ~ a + b·voltgt               → alpha over the *vol-managed* SPY factor (Moreira–Muir);
                                             if this a ≈ 0, the stack is the vol-targeting factor.
    Alpha is reported in % per day. The decisive number is the second regression's alpha: if it
    collapses toward zero (and R² rises) the stack is spanned by the vol-managed factor — managed
    beta, not a distinct edge.
    """
    legs = _legs(wf)

    def ols(y, x):
        X = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
        return coef[0], coef[1], float(r2), resid

    y = legs["stack"].to_numpy()
    a_m, b_m, r2_m, _ = ols(y, legs["buyhold"].to_numpy())
    a_v, b_v, r2_v, _ = ols(y, legs["voltgt"].to_numpy())
    return {
        "beta_market": float(b_m),
        "alpha_vs_market_pct_day": float(a_m),
        "r2_market": r2_m,
        "beta_voltgt": float(b_v),
        "alpha_vs_voltgt_pct_day": float(a_v),
        "r2_voltgt": r2_v,
    }


def _position(wf: WalkForward) -> pd.Series:
    """Signed position size per day for the stack (sign × size); constant-long uses +size."""
    f = wf.frame
    sig = pd.Series(1.0, index=f.index) if wf.constant_long else f["sign"]
    return (sig * f["size"]).rename("position")


def cost_sweep(wf: WalkForward, spreads_bps=(0.0, 0.5, 1.0, 2.0, 5.0)) -> pd.DataFrame:
    """Charge the forecast's turnover and watch the ARIMA increment cross zero.

    Turnover each day = |position_t − position_{t-1}|; cost = turnover × spread (bps, one-way per
    unit traded). The forecast's sign-flips create turnover the constant-long control does not, so
    the *net* ARIMA increment (stack_net − voltgt_net) falls as the spread rises. Returns a table
    over ``spreads_bps`` with each leg's net annualised Sharpe and the net increment.
    """
    legs = _legs(wf)
    f = wf.frame
    pos_stack = (f["sign"] * f["size"])
    pos_voltgt = (1.0 * f["size"])
    turn_stack = pos_stack.diff().abs().fillna(pos_stack.abs())
    turn_voltgt = pos_voltgt.diff().abs().fillna(pos_voltgt.abs())

    rows = []
    for bps in spreads_bps:
        c = bps / 1e4 * 100.0  # bps -> % (returns are in %), charged per unit turnover
        net_stack = legs["stack"] - c * turn_stack
        net_voltgt = legs["voltgt"] - c * turn_voltgt
        rows.append(
            {
                "spread_bps": bps,
                "sharpe_stack_net": annualized_sharpe(net_stack),
                "sharpe_voltgt_net": annualized_sharpe(net_voltgt),
                "arima_increment_net": annualized_sharpe(net_stack) - annualized_sharpe(net_voltgt),
                "stack_turnover_ann": float(turn_stack.mean() * 252),
            }
        )
    return pd.DataFrame(rows).set_index("spread_bps")


def win_rate(wf: WalkForward) -> dict:
    """The per-day win rate of the sized bet — the "win every single trade" headline, measured."""
    legs = _legs(wf)
    stack = legs["stack"]
    active = stack[stack != 0.0]
    return {
        "win_rate": float((active > 0).mean()),
        "n_active": int(active.shape[0]),
        "mean_daily_pct": float(stack.mean()),
    }


def in_sample_inflation(returns: pd.Series, wf_walk: WalkForward, lookback: int = 252) -> dict:
    """The control the article warns about: in-sample fit-then-grade vs honest walk-forward.

    Fits ARIMA(1,0,1) **once on the whole series** and grades its (in-sample) one-step predictions
    on the same data — "a curve fit with a label" — then compares the directional hit-rate to the
    walk-forward one. The gap is the inflation a reader gets for free by cheating.
    """
    from statsmodels.tsa.arima.model import ARIMA
    import warnings

    r = returns.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(np.asarray(r, dtype=float), order=(1, 0, 1)).fit()
        pred = res.predict(start=0, end=len(r) - 1)
    sign_is = np.sign(pred)
    nz = np.asarray(r) != 0
    hit_is = float((sign_is[nz] == np.sign(np.asarray(r))[nz]).mean())
    wf_edge = directional_edge(wf_walk)
    return {
        "hit_rate_in_sample": hit_is,
        "hit_rate_walk_forward": wf_edge["hit_rate"],
        "inflation_pp": (hit_is - wf_edge["hit_rate"]) * 100.0,
    }


def run_all(returns: pd.Series, lookback: int = 252, max_steps: int | None = None,
            seed: int = 0) -> dict:
    """Run the whole teardown once and return every headline number in one dict."""
    wf = generate_signals(returns, lookback=lookback, max_steps=max_steps)
    return {
        "directional_edge": directional_edge(wf),
        "sharpe_decomposition": sharpe_decomposition(wf, seed=seed),
        "alpha_vs_beta": alpha_vs_beta(wf),
        "cost_sweep": cost_sweep(wf),
        "win_rate": win_rate(wf),
        "_wf": wf,
    }
