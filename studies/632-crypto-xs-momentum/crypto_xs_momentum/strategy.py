"""Strategy + inference for Study 632 — Crypto Cross-Sectional Momentum.

The claim (Liu, Tsyvinski & Wu 2022, *Common Risk Factors in Cryptocurrency*, JF; Liu &
Tsyvinski 2021, RFS): sort coins each week on their past 1–4 week return; last week's
winners keep winning — cross-sectional momentum is the one robust return factor crypto
has. We rebuild it as a weekly quintile study:

* **Formation.** At the close of week *t* (Sunday 23:59 UTC), rank every eligible coin on
  its cumulative return over the past ``k`` weeks (weeks *t−k+1 … t*). Eligibility: a
  valid formation return AND a tradable close at *t*.
* **Holding — exactly ONE execution lag.** Positions formed on week-*t* information earn
  the week *t+1* close-to-close return. Nothing formed at *t* touches week *t*.
* **Portfolios.** Winners = top quintile (``max(2, n//5)`` names, equal-weight), Losers =
  bottom quintile. WML = Winners − Losers. Long-only leg = Winners vs the equal-weight
  market of all eligible coins (the honest benchmark — in crypto everything is beta).
* **Delisting rule.** A coin's crash weeks are in the panel as long as prices printed
  (LUNA's −100% week, FTT's −93% week ARE held if the coin ranked into a quintile). Only
  a name's final pre-halt week (no next close on the venue) drops out of the tradable
  cross-section — you could not have bought a halted coin.

Inference, the desk's shared bar:

* **Newey-West (HAC) t** on the weekly WML mean (lag 4) — weekly long-shorts overlap
  regimes, so plain t's are banned.
* **Random-rank placebo averaged over ≥ 20 seeds** — same quintile sizes, ranks drawn at
  random; the machinery must NOT pay a shuffled signal.
* **Costs.** One-way cost × traded NAV (every buy and every sell pays one leg), weekly
  turnover measured, shorts pay borrow (10%/yr on short NAV — Binance margin scale;
  perp funding, historically positive, would usually *pay* the short and is noted).
* **Sub-periods + regime split.** Calendar sub-periods with their own HAC t's, plus a
  bull/bear split on a *lagged* BTC 30-week-SMA regime flag (known at formation, applied
  to the next week) with a Welch t on the difference — the third-axis 2022-bear question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import WEEKS_PER_YEAR

MIN_NAMES = 10          # a week needs at least this many eligible coins to count


# --------------------------------------------------------------------------- #
# Portfolio engine
# --------------------------------------------------------------------------- #
def weekly_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close weekly simple returns; NaN gaps stay NaN (no fill across halts)."""
    return panel.pct_change(fill_method=None)


def run_xs_momentum(panel: pd.DataFrame, k: int = 1, min_names: int = MIN_NAMES,
                    rank_matrix: np.ndarray | None = None) -> pd.DataFrame:
    """Weekly winners/losers quintile returns with one-week execution lag.

    For each week *t* with ``n >= min_names`` eligible coins (valid ``k``-week formation
    return through *t*, valid close at *t*, valid return over *t+1*), rank on the
    formation return and earn the *t+1* return. Returns a frame indexed by the HOLDING
    week (*t+1*) with columns::

        win, lose, wml, mkt, n, to_win, to_lose

    ``to_win`` / ``to_lose`` are the week's traded NAV of each leg (sum of |weight
    changes|, buys + sells — each unit pays the one-way cost once).
    ``rank_matrix`` (same shape as the returns matrix) substitutes the formation scores —
    used by the random-rank placebo only.
    """
    rets = weekly_returns(panel)
    form = (1.0 + rets).rolling(k).apply(np.prod, raw=True) - 1.0 if k > 1 else rets
    r = rets.to_numpy()
    f = form.to_numpy() if rank_matrix is None else rank_matrix
    px = panel.to_numpy()
    n_weeks, n_coins = r.shape

    rows = []
    prev_w_win = np.zeros(n_coins)
    prev_w_lose = np.zeros(n_coins)
    for t in range(k, n_weeks - 1):
        elig = np.isfinite(f[t]) & np.isfinite(px[t]) & np.isfinite(r[t + 1])
        n = int(elig.sum())
        if n < min_names:
            prev_w_win = np.zeros(n_coins)
            prev_w_lose = np.zeros(n_coins)
            continue
        q = max(2, n // 5)
        idx = np.where(elig)[0]
        order = idx[np.argsort(f[t][idx])]
        losers, winners = order[:q], order[-q:]

        w_win = np.zeros(n_coins)
        w_win[winners] = 1.0 / q
        w_lose = np.zeros(n_coins)
        w_lose[losers] = 1.0 / q
        to_win = float(np.abs(w_win - prev_w_win).sum())
        to_lose = float(np.abs(w_lose - prev_w_lose).sum())
        prev_w_win, prev_w_lose = w_win, w_lose

        rw = float(r[t + 1][winners].mean())
        rl = float(r[t + 1][losers].mean())
        rows.append({
            "week": panel.index[t + 1], "win": rw, "lose": rl, "wml": rw - rl,
            "mkt": float(r[t + 1][idx].mean()), "n": n,
            "to_win": to_win, "to_lose": to_lose,
        })
    return pd.DataFrame(rows).set_index("week")


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray, lags: int = 4) -> float:
    """Newey-West t of mean(x) vs 0 with Bartlett kernel (autocorrelation-robust)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return float("nan")
    e = x - x.mean()
    s = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        s += 2.0 * w * float(e[lag:] @ e[:-lag]) / n
    se = np.sqrt(max(s, 1e-30) / n)
    return float(x.mean() / se)


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) − mean(b), unequal variances."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def summarize(series: pd.Series, lags: int = 4) -> dict:
    """Weekly mean (bps), annualised %, HAC t, Sharpe (ann), n weeks, hit rate."""
    x = series.dropna()
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    return {
        "n": int(len(x)),
        "mean_bps": mu * 1e4,
        "ann_pct": (np.expm1(np.log1p(mu) * WEEKS_PER_YEAR)) * 100,
        "hac_t": hac_t(x.to_numpy(), lags=lags),
        "sharpe": (mu / sd * np.sqrt(WEEKS_PER_YEAR)) if sd > 0 else float("nan"),
        "hit_pct": float((x > 0).mean() * 100),
    }


def placebo_random_ranks(panel: pd.DataFrame, k: int = 1, n_seeds: int = 20,
                         seed0: int = 632) -> dict:
    """Random-rank placebo, AVERAGED over ``n_seeds`` seeds (single-seed baselines banned).

    Replaces the formation score with i.i.d. noise (same eligibility, same quintile
    sizes, same lag). Reports the across-seed mean/std of the weekly WML mean and the
    across-seed mean HAC t — all of which must sit at ~0 for the machinery to be honest.
    """
    rets = weekly_returns(panel).to_numpy()
    means, ts = [], []
    for i in range(n_seeds):
        rng = np.random.default_rng(seed0 + i)
        fake = rng.standard_normal(rets.shape)
        fake[~np.isfinite(rets)] = np.nan     # keep the real eligibility pattern
        res = run_xs_momentum(panel, k=k, rank_matrix=fake)
        means.append(float(res["wml"].mean()))
        ts.append(hac_t(res["wml"].to_numpy()))
    means = np.asarray(means)
    return {"n_seeds": n_seeds, "mean_bps": float(means.mean() * 1e4),
            "std_bps": float(means.std(ddof=1) * 1e4), "mean_hac_t": float(np.mean(ts))}


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def net_wml(res: pd.DataFrame, cost_bps: float, borrow_ann: float = 0.10) -> pd.Series:
    """Net weekly WML: gross − one-way cost × traded NAV (both legs) − short borrow.

    Every unit of traded NAV (buys + sells, ``to_win + to_lose``) pays the one-way cost
    once. The short leg pays ``borrow_ann`` (annual, on 1× short NAV) pro-rata per week —
    Binance margin scale for majors; perp funding (historically positive ≈ longs pay
    shorts) would usually offset this, so the charge is conservative.
    """
    c = cost_bps / 1e4
    drag = c * (res["to_win"] + res["to_lose"]) + borrow_ann / WEEKS_PER_YEAR
    return res["wml"] - drag


def net_long_only(res: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Net weekly winners-leg return: gross − one-way cost × the leg's traded NAV."""
    return res["win"] - (cost_bps / 1e4) * res["to_win"]


# --------------------------------------------------------------------------- #
# Regimes (third axis)
# --------------------------------------------------------------------------- #
def btc_regime(panel: pd.DataFrame, window: int = 30) -> pd.Series:
    """Lagged bull flag: BTC weekly close above its ``window``-week SMA, shifted +1.

    The flag is computed on closes up to week *t* and applied to the week *t+1* return —
    the same one-week lag as the strategy, so the split is knowable at formation.
    """
    btc = panel["BTC"]
    bull = (btc > btc.rolling(window).mean()).shift(1)
    return bull


def regime_split(res: pd.DataFrame, bull: pd.Series) -> dict:
    """WML stats in lagged-bull vs lagged-bear weeks + Welch t of the difference."""
    flag = bull.reindex(res.index)
    ok = flag.notna()
    up = res.loc[ok & (flag == True), "wml"]      # noqa: E712
    dn = res.loc[ok & (flag == False), "wml"]     # noqa: E712
    return {"bull": summarize(up), "bear": summarize(dn),
            "welch_t_diff": welch_t(up.to_numpy(), dn.to_numpy())}


def sub_period(res: pd.DataFrame, start: str, end: str) -> dict:
    """Summarize WML over holding weeks in [start, end]."""
    m = (res.index >= pd.Timestamp(start)) & (res.index <= pd.Timestamp(end))
    return summarize(res.loc[m, "wml"])
