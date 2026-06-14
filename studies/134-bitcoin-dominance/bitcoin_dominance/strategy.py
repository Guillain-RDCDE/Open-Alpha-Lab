"""The strategy and its honest controls — Study 134 (Bitcoin-Dominance).

The folk recipe: when Bitcoin's share of total crypto market cap (BTC dominance) is
*falling* or *low*, altcoins are said to enter "alt season" — a rotation where capital
flows from BTC into smaller coins, lifting alts relative to BTC. We test the
operational version: does *falling* BTC dominance (measured over the prior N days)
forecast *positive* alt-minus-BTC forward returns over the next week?

Two controls keep the verdict honest:

- **Base-rate control**: the unconditional alt-minus-BTC return. If alts outperform BTC
  on average (because alts have higher secular beta in a bull market), then "signal = on"
  just by being long alt-minus-BTC. Our signal must beat this base rate.
- **Random-period control**: the same forward-return computation on randomly selected
  periods (matched count). Tests whether the dominance timing adds anything vs the
  secular alt-outperformance trend.

Three signal variants prevent us from cherry-picking the best lookback:

- ``dom_chg_10d`` — 10-day dominance change (medium-term rotation signal)
- ``dom_chg_20d`` — 20-day dominance change (intermediate)
- ``dom_level_pct`` — current dominance vs rolling 52-week percentile (low = oversold BTC)

No look-ahead: the signal is formed on day t's closing dominance; the forward return is
measured from day t+1 to day t+N (1-week = 5 trading days, our canonical horizon).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import compute_dominance, ALT_TICKERS, BTC_TICKER, SUPPLY_M


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def alt_basket_return(
    panel: pd.DataFrame,
    horizon: int = 5,
    btc_col: str = BTC_TICKER,
    alt_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Forward alt-minus-BTC log-return from day t+1 to day t+horizon.

    The "alt return" is the equal-weighted average log-return of all alt tickers
    available on day t (tickers in the panel minus BTC). The spread is alt_return −
    btc_return. Signal is formed on t; return is realised from t+1 to t+horizon.

    Returns a DataFrame indexed by signal date t with columns:
      ``btc_fwd``, ``alt_fwd``, ``spread_fwd`` (all log-returns).
    """
    if alt_cols is None:
        alt_cols = [c for c in panel.columns if c != btc_col]

    log_panel = np.log(panel)
    rows = []
    idx = panel.index
    n = len(idx)

    for i in range(n - horizon):
        # Log-return from close of day i+1 to close of day i+horizon (entry at t+1)
        btc_fwd = log_panel[btc_col].iloc[i + horizon] - log_panel[btc_col].iloc[i + 1]
        alt_rets = [
            log_panel[c].iloc[i + horizon] - log_panel[c].iloc[i + 1]
            for c in alt_cols
            if not np.isnan(log_panel[c].iloc[i]) and not np.isnan(log_panel[c].iloc[i + horizon])
        ]
        if not alt_rets:
            continue
        alt_fwd = float(np.mean(alt_rets))
        rows.append({
            "signal_date": idx[i],
            "btc_fwd": btc_fwd,
            "alt_fwd": alt_fwd,
            "spread_fwd": alt_fwd - btc_fwd,
        })

    out = pd.DataFrame(rows).set_index("signal_date")
    out.index.name = "date"
    return out


def dominance_signal(
    panel: pd.DataFrame,
    lookback: int = 10,
    btc_col: str = BTC_TICKER,
    supply_map: dict | None = None,
) -> pd.Series:
    """N-day change in BTC dominance (negative = falling BTC share = 'alt season' signal).

    Returns the signed dominance change (dom_t − dom_{t-lookback}) as a Series indexed
    by date. A negative value means BTC's share of basket cap has fallen — the recipe's
    trigger for "alt season is here". We keep the raw change (not a binary threshold) so
    we can measure the continuous relationship in a regression and avoid data-mining a
    cut-off.
    """
    if supply_map is None:
        supply_map = SUPPLY_M
    dom = compute_dominance(panel, btc_col=btc_col, supply_map=supply_map)
    return dom.diff(lookback).rename(f"dom_chg_{lookback}d")


def dominance_percentile(
    panel: pd.DataFrame,
    window: int = 252,
    btc_col: str = BTC_TICKER,
    supply_map: dict | None = None,
) -> pd.Series:
    """Rolling percentile rank of BTC dominance vs its trailing ``window``-day history.

    Values near 0 = dominance historically low ("alts dominating"); values near 1 =
    dominance historically high ("BTC season"). The recipe predicts that low-percentile
    readings forecast positive alt-minus-BTC forward returns.
    """
    if supply_map is None:
        supply_map = SUPPLY_M
    dom = compute_dominance(panel, btc_col=btc_col, supply_map=supply_map)
    return dom.rolling(window, min_periods=window // 2).rank(pct=True).rename("dom_pct")


# ---------------------------------------------------------------------------
# Signal–outcome pairing and inference
# ---------------------------------------------------------------------------
def signal_outcome_pairs(
    panel: pd.DataFrame,
    signal: pd.Series,
    horizon: int = 5,
    btc_col: str = BTC_TICKER,
) -> pd.DataFrame:
    """Join the signal series with the forward alt-minus-BTC spread return.

    Returns a DataFrame with columns ``signal``, ``spread_fwd``, ``btc_fwd``,
    ``alt_fwd``, indexed by signal date. Only rows where both signal and forward
    return are available are kept.
    """
    fwd = alt_basket_return(panel, horizon=horizon, btc_col=btc_col)
    joined = fwd.join(signal.rename("signal"), how="inner")
    joined = joined.dropna(subset=["signal", "spread_fwd"])
    return joined


def random_period_control(
    pairs: pd.DataFrame,
    n_draws: int = 500,
    seed: int = 134,
) -> pd.DataFrame:
    """Bootstrap control: shuffle the signal dates and recompute the mean spread.

    On each draw we randomly permute the signal column (keeping the spread_fwd fixed),
    then compute the mean spread in "alt season" (signal < 0, i.e. falling dominance)
    periods. This breaks any timing link between signal and outcome while preserving
    the marginal distribution of each — the standard permutation null.

    Returns a DataFrame with columns ``n_signal``, ``mean_spread`` for each draw.
    """
    rng = np.random.default_rng(seed)
    spreads = pairs["spread_fwd"].to_numpy(dtype=float)
    signals = pairs["signal"].to_numpy(dtype=float)
    rows = []
    for _ in range(n_draws):
        sig_shuf = rng.permutation(signals)
        mask = sig_shuf < 0  # "alt season" in this permuted draw
        if mask.sum() < 5:
            continue
        rows.append({
            "n_signal": int(mask.sum()),
            "mean_spread": float(spreads[mask].mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics (HAC t-stat, same Newey-West kernel as the rest of the lab)
# ---------------------------------------------------------------------------
def summarize(
    returns: np.ndarray | pd.Series,
    label: str = "",
) -> dict:
    """Headline per-period statistics: n, mean, std, skew, and HAC t-stat.

    The HAC (Newey-West) t-stat is the inference-bar number: |t| >= 2 on the real tape
    is required for Signal=REAL. With a few hundred weekly periods the power is
    moderate; with fewer (early alt history) we say so.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "label": label,
        "n": int(n),
        "mean_pct": float(r.mean() * 100) if n else float("nan"),
        "std_pct": float(r.std(ddof=1) * 100) if n > 1 else float("nan"),
        "skew": float(pd.Series(r).skew()) if n > 2 else float("nan"),
        "tstat": float("nan"),
        "low_power": n < 50,
    }
    if n > 5:
        mu = r.mean()
        e = r - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


def regression_tstat(
    y: np.ndarray | pd.Series,
    x: np.ndarray | pd.Series,
) -> dict:
    """OLS slope of y on x with HAC standard error (Newey-West).

    Tests whether the dominance change *continuously* predicts the alt-minus-BTC spread
    (a negative slope would confirm the claim: lower/more-falling dominance → higher spread).

    Returns: ``slope``, ``intercept``, ``r2``, ``tstat`` (HAC), ``n``.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    n = len(y)
    if n < 10:
        return {"slope": float("nan"), "intercept": float("nan"),
                "r2": float("nan"), "tstat": float("nan"), "n": int(n)}

    # OLS
    xc = np.column_stack([np.ones(n), x])
    beta, _, _, _ = np.linalg.lstsq(xc, y, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])
    yhat = xc @ beta
    resid = y - yhat
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # HAC SE for slope (Newey-West on (x * resid))
    score = x * resid
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    s_mu = score.mean()
    e = score - s_mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    xx = float((x - x.mean()) @ (x - x.mean())) / n
    se_slope = np.sqrt(max(lrv, 0.0) / n) / xx if xx > 0 else float("nan")
    tstat = slope / se_slope if np.isfinite(se_slope) and se_slope > 0 else float("nan")

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "tstat": float(tstat),
        "n": int(n),
    }
