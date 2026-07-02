"""The engine and its honest controls — Study 578 (Cross-Asset-Correlation-Regime).

The claim, at full strength: **average cross-asset correlation is a fragility gauge.** When the
mean pairwise correlation across a diverse asset panel rises (everything moves together), the
usual diversification is gone and a stress event is nearer — so a HIGH / rising correlation regime
should *predict* lower forward returns and higher forward volatility on a risk asset (SPY).

This module measures that, honestly:

1. **The correlation index.** Over a trailing window (default 63 trading days ~= 1 quarter) compute
   the full pairwise correlation matrix of the panel's daily returns and take the mean of its
   off-diagonal entries — one number per day, the *average cross-asset correlation* ``avg_corr``.

2. **The regime split.** Classify each day HIGH if ``avg_corr`` sits above a rolling/expanding
   quantile threshold (default the trailing 70th percentile, using only *past* data — no
   look-ahead), else LOW. This is the fragility regime.

3. **The forward test.** For each day, measure the risk asset's forward return and forward realised
   volatility over the next ``horizon`` days (default 21 ~= 1 month). Compare the HIGH-regime and
   LOW-regime forward-return distributions with a two-sample (Welch) *t*; the claim predicts
   HIGH forward return < LOW forward return (a *negative* HIGH-minus-LOW spread) and HIGH forward
   vol > LOW forward vol.

4. **A tradable overlay.** A risk-off rule: hold SPY in the LOW regime, go flat (cash) in the HIGH
   regime, entered with a one-day execution lag. Report gross AND net (round-trip cost per regime
   switch) and its Sharpe vs buy-and-hold — the honest tradability read.

5. **Placebo null.** Block-shuffle the regime labels against the forward series and read the
   spread's tail probability, so serial correlation in overlapping forward windows can't manufacture
   significance.

6. **A synthetic positive control.** A deterministic world where fragility is planted; the engine
   must recover a negative HIGH-minus-LOW forward-return spread (and stay flat at the null),
   seed-robust over >= 20 seeds.

Execution convention: the correlation index and the regime threshold use only *trailing* data
(the window ending at day *t* and the expanding quantile of days <= *t*), so the regime label is
fully public at day *t*'s close; the forward return/vol are measured strictly *after* *t*, and the
tradable overlay enters one bar later (a documented one-day execution lag). No future data enters
the signal (no look-ahead). Forward windows overlap, so the two-sample *t* overstates precision —
we lean on the block-bootstrap placebo and read the *t* as a sign/effect-size guide, not a
z-table p-value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISK_ASSET = "SPY"


# ---------------------------------------------------------------------------
# The average cross-asset correlation index
# ---------------------------------------------------------------------------
def avg_pairwise_corr(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """Trailing mean off-diagonal pairwise correlation across the panel, one value per day.

    For each day *t* uses the ``window`` days ending at *t* (inclusive) — trailing only, no
    look-ahead. Days with fewer than ``window`` observations, or fewer than 3 usable assets, are
    NaN. Returns a Series indexed like ``returns``.
    """
    if returns.empty:
        return pd.Series(dtype=float)
    R = returns.to_numpy(dtype=float)
    n, k = R.shape
    out = np.full(n, np.nan)
    iu = np.triu_indices(k, k=1)
    for t in range(window - 1, n):
        block = R[t - window + 1 : t + 1]
        # need enough non-nan per column
        good = np.isfinite(block).sum(axis=0) >= max(10, window // 3)
        if good.sum() < 3:
            continue
        b = block[:, good]
        # pairwise corr on the good columns, ignoring rows with any nan among them
        b = b[np.isfinite(b).all(axis=1)]
        if b.shape[0] < 10:
            continue
        c = np.corrcoef(b, rowvar=False)
        gk = c.shape[0]
        iu2 = np.triu_indices(gk, k=1)
        out[t] = float(np.nanmean(c[iu2]))
    return pd.Series(out, index=returns.index, name="avg_corr")


def _expanding_quantile(s: pd.Series, q: float, min_periods: int) -> pd.Series:
    """Expanding quantile using only past-and-present data (no look-ahead)."""
    return s.expanding(min_periods=min_periods).quantile(q)


def regime_labels(
    avg_corr: pd.Series, q: float = 0.70, min_periods: int = 252
) -> pd.Series:
    """HIGH (1) / LOW (0) fragility regime from the correlation index.

    A day is HIGH if its ``avg_corr`` exceeds the *expanding* ``q``-quantile of the index up to and
    including that day (only past data — no look-ahead). The first ``min_periods`` days are NaN
    (not enough history to set a threshold). Returns a Series of {0.0, 1.0, NaN}.
    """
    if avg_corr.empty:
        return pd.Series(dtype=float)
    thr = _expanding_quantile(avg_corr.dropna(), q, min_periods)
    thr = thr.reindex(avg_corr.index)
    lab = (avg_corr > thr).astype(float)
    lab[thr.isna() | avg_corr.isna()] = np.nan
    return lab.rename("regime")


# ---------------------------------------------------------------------------
# Forward return / forward vol of the risk asset
# ---------------------------------------------------------------------------
def forward_return(risk_ret: pd.Series, horizon: int = 21) -> pd.Series:
    """Forward simple return of the risk asset over the next ``horizon`` days (strictly after t)."""
    logret = np.log1p(risk_ret.astype(float))
    fwd_log = logret.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
    return (np.expm1(fwd_log)).rename("fwd_ret")


def forward_vol(risk_ret: pd.Series, horizon: int = 21) -> pd.Series:
    """Forward annualised realised vol of the risk asset over the next ``horizon`` days."""
    r = risk_ret.astype(float)
    fwd_std = r.shift(-1).rolling(horizon).std(ddof=1).shift(-(horizon - 1))
    return (fwd_std * np.sqrt(TRADING_DAYS)).rename("fwd_vol")


# ---------------------------------------------------------------------------
# The regime forward test — the sign IS the claim
# ---------------------------------------------------------------------------
def _welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    diff = a.mean() - b.mean()
    t = diff / se if se > 0 else float("nan")
    return float(diff), float(t)


def regime_forward_test(
    returns: pd.DataFrame,
    window: int = 63,
    horizon: int = 21,
    q: float = 0.70,
    min_periods: int = 252,
) -> dict:
    """Compare the risk asset's forward return / forward vol across HIGH vs LOW correlation regimes.

    Returns a dict with the mean forward return in each regime, the HIGH-minus-LOW spread and its
    two-sample *t* (the claim predicts a *negative* spread), the same for forward vol (claim: HIGH
    vol > LOW vol, positive spread), regime sample sizes, and the aligned per-day arrays (for the
    placebo and plots). To avoid the leaked overlap between the trailing window and the forward
    window, the forward series is measured strictly after day *t* (see ``forward_return``).
    """
    if returns.empty or RISK_ASSET not in returns.columns:
        return {}
    ac = avg_pairwise_corr(returns, window=window)
    lab = regime_labels(ac, q=q, min_periods=min_periods)
    fr = forward_return(returns[RISK_ASSET], horizon=horizon)
    fv = forward_vol(returns[RISK_ASSET], horizon=horizon)

    df = pd.DataFrame({"regime": lab, "fwd_ret": fr, "fwd_vol": fv, "avg_corr": ac}).dropna(
        subset=["regime", "fwd_ret"]
    )
    if df.empty:
        return {}
    hi = df[df["regime"] == 1.0]
    lo = df[df["regime"] == 0.0]

    r_diff, r_t = _welch_t(hi["fwd_ret"].to_numpy(), lo["fwd_ret"].to_numpy())
    v_diff, v_t = _welch_t(
        hi["fwd_vol"].dropna().to_numpy(), lo["fwd_vol"].dropna().to_numpy()
    )
    return {
        "window": window,
        "horizon": horizon,
        "q": q,
        "n_high": int(len(hi)),
        "n_low": int(len(lo)),
        "high_fwd_ret": float(hi["fwd_ret"].mean()),
        "low_fwd_ret": float(lo["fwd_ret"].mean()),
        "ret_spread": float(r_diff),          # HIGH - LOW (claim: negative)
        "ret_t": float(r_t),
        "high_fwd_vol": float(hi["fwd_vol"].mean()),
        "low_fwd_vol": float(lo["fwd_vol"].mean()),
        "vol_spread": float(v_diff),          # HIGH - LOW (claim: positive)
        "vol_t": float(v_t),
        "avg_corr_mean": float(ac.mean(skipna=True)),
        "_df": df,
    }


# ---------------------------------------------------------------------------
# Placebo / block-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(res: dict, n_perm: int = 2000, block: int = 21, seed: int = 578) -> float:
    """Block-shuffle placebo p-value for the HIGH-minus-LOW forward-return spread.

    Overlapping forward windows induce serial correlation, so a plain label shuffle would be too
    optimistic. We circularly rotate the regime labels in ``block``-sized chunks (preserving the
    forward series' autocorrelation) ``n_perm`` times and count how often the shuffled |spread|
    meets or beats the observed |spread|. A real regime signal sits in the tail.
    """
    if not res or "_df" not in res:
        return float("nan")
    df = res["_df"]
    reg = df["regime"].to_numpy(dtype=float)
    y = df["fwd_ret"].to_numpy(dtype=float)
    obs = abs(res["ret_spread"])
    n = len(y)
    if n < 3 * block:
        return float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    count = 0
    for _ in range(n_perm):
        # break labels into blocks, permute block order, re-concatenate
        order = rng.permutation(n_blocks)
        chunks = [reg[i * block : (i + 1) * block] for i in order]
        perm = np.concatenate(chunks)[:n]
        hi = y[perm == 1.0]
        lo = y[perm == 0.0]
        if len(hi) < 2 or len(lo) < 2:
            continue
        spread = hi.mean() - lo.mean()
        if abs(spread) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# The tradable overlay — risk-off in the HIGH regime
# ---------------------------------------------------------------------------
def overlay_backtest(
    returns: pd.DataFrame,
    window: int = 63,
    q: float = 0.70,
    min_periods: int = 252,
    cost_bps: float = 5.0,
) -> dict:
    """Risk-off overlay: long SPY in LOW regime, flat in HIGH regime, one-day execution lag.

    The regime label is public at day *t*'s close; the position for day *t*+1 is set from it (the
    single documented execution lag). A cost of ``cost_bps`` one-way is charged on each change of
    position (entering/leaving cash). Returns gross and net annualised return, vol, Sharpe and max
    drawdown for the overlay and for buy-and-hold, plus the number of switches.
    """
    if returns.empty or RISK_ASSET not in returns.columns:
        return {}
    ac = avg_pairwise_corr(returns, window=window)
    lab = regime_labels(ac, q=q, min_periods=min_periods)
    spy = returns[RISK_ASSET].astype(float)

    pos = (lab == 0.0).astype(float)          # 1 when LOW regime (invested), 0 when HIGH (cash)
    pos = pos.shift(1)                         # one-day execution lag
    df = pd.DataFrame({"spy": spy, "pos": pos, "lab": lab}).dropna(subset=["spy", "pos"])
    if df.empty:
        return {}

    switches = int((df["pos"].diff().abs() > 0).sum())
    cost = df["pos"].diff().abs().fillna(0.0) * (cost_bps * 1e-4)
    gross = df["pos"] * df["spy"]
    net = gross - cost

    def _stats(r: pd.Series) -> dict:
        r = r.dropna()
        if r.empty:
            return {"cagr": float("nan"), "vol": float("nan"), "sharpe": float("nan"),
                    "maxdd": float("nan")}
        ann = float(r.mean() * TRADING_DAYS)
        vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
        sharpe = ann / vol if vol > 0 else float("nan")
        eq = (1.0 + r).cumprod()
        dd = float((eq / eq.cummax() - 1.0).min())
        return {"cagr": ann, "vol": vol, "sharpe": sharpe, "maxdd": dd}

    return {
        "switches": switches,
        "cost_bps": cost_bps,
        "overlay_gross": _stats(gross),
        "overlay_net": _stats(net),
        "buy_hold": _stats(df["spy"]),
        "frac_high": float((df["lab"] == 1.0).mean()),
    }


# ---------------------------------------------------------------------------
# Robustness sweep — vary window/horizon/quantile
# ---------------------------------------------------------------------------
def robustness_sweep(returns: pd.DataFrame, specs: list[tuple[int, int, float]]) -> pd.DataFrame:
    """Re-run the regime forward test over a grid of (window, horizon, q) and collect the sign.

    Returns a tidy frame: one row per spec with the HIGH-minus-LOW forward-return spread and its
    two-sample *t*. A signal worth its stamp keeps its sign (negative spread) across the grid.
    """
    rows = []
    for (w, h, q) in specs:
        res = regime_forward_test(returns, window=w, horizon=h, q=q)
        if not res:
            continue
        rows.append(
            {
                "window": w, "horizon": h, "q": q,
                "ret_spread": res["ret_spread"], "ret_t": res["ret_t"],
                "vol_spread": res["vol_spread"], "vol_t": res["vol_t"],
                "n_high": res["n_high"], "n_low": res["n_low"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(
    data_mod, fragility: float, n_seeds: int = 25, base_seed: int = 578,
    window: int = 63, horizon: int = 21, q: float = 0.70,
) -> tuple[float, float]:
    """Average the HIGH-minus-LOW forward-return spread and its *t* over ``n_seeds`` worlds.

    The house rule: any synthetic-dependent claim averages a Welch-style stat over >= 20 seeds so
    no single lucky RNG seed can manufacture significance. Returns ``(mean_spread, mean_t)`` across
    seeds for a planted ``fragility`` (the claim plants a *negative* spread; the null is flat).
    """
    spreads, ts = [], []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_series(fragility=fragility, seed=s)
        res = regime_forward_test(panel, window=window, horizon=horizon, q=q, min_periods=252)
        if res:
            spreads.append(res["ret_spread"])
            ts.append(res["ret_t"])
    return float(np.nanmean(spreads)), float(np.nanmean(ts))
