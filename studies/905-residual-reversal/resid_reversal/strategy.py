"""Strategy + inference for Study 905 — Residual Reversal.

The claim (Blitz, Huij, Lansdorp & Verbeek 2013): the raw one-week reversal — long last
week's losers, short last week's winners — mostly harvests **bid-ask bounce** and
**common-factor** moves, so it dies at the spread. Clean it: regress each name's weekly
return on the market, keep the **residual**, and reverse on that instead. A long
past-week-residual-loser / short past-week-residual-winner book, on a liquid subset,
should earn a spread that survives Newey-West and costs better than the raw version.

This is distinct from:

* [329-one-month-reversal](../../329-one-month-reversal/) — the classic monthly Jegadeesh
  reversal on **raw** returns (no factor cleaning). We test the *residualised* weekly
  version and put the raw weekly reversal beside it as the foil.
* [800-high-frequency-reversal](../../800-high-frequency-reversal/) — very-short-horizon
  (daily / intraday) raw reversal. We work at the **weekly** horizon on the market-model
  **residual**.
* [377-bid-ask-bounce](../../377-bid-ask-bounce/) — isolates the microstructure bounce
  itself. Here the bounce is the *contaminant* the residual + liquidity screen aim to
  strip, not the signal.
* [237-residual-momentum](../../237-residual-momentum/) — residual **momentum** (Blitz-
  Huij-Martens): the *continuation* of the residual over a long formation window. This is
  the opposite sign at the opposite horizon — short-term residual **reversal**.

Method:

* **Weekly returns.** Resample each name's Close to Friday, simple weekly return. The
  market proxy is the equal-weight cross-sectional mean weekly return.
* **Market-model residual.** Per name, a trailing-``beta_window``-week rolling regression
  of the weekly return on the market (vectorised rolling cov/var → ``beta``, rolling
  means → ``alpha``); the residual of week ``w`` is ``r_w − alpha_w − beta_w · mkt_w``,
  using only data through week ``w``.
* **Point-in-time sort.** On each week ``w`` rank names by the residual (or raw return)
  known at the close of ``w−1`` (one ``shift``) and hold week ``w``. Long the bottom
  ``frac`` (residual losers), short the top ``frac`` (residual winners); equal weight. A
  **dollar-volume liquidity screen** first keeps only the top ``liq_frac`` names by
  trailing dollar volume (bid-ask bounce is an illiquid-name artefact).
* **Inference.** Newey-West (HAC) *t* on the weekly long-short spread; a one-sample *t*
  and a pooled Welch *t* (loser book vs winner book) cross-check; a permutation placebo
  breaks the signal→outcome link; a costed timer charges the round-trip friction + borrow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_WEEKS = 52


# --------------------------------------------------------------------------- #
# Return panels
# --------------------------------------------------------------------------- #
def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple close-to-close returns (index=date, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    return closes.pct_change()


def weekly_returns(panel: dict[str, pd.DataFrame], rule: str = "W-FRI") -> pd.DataFrame:
    """Weekly simple returns per name (Friday-stamped), index=week, columns=ticker."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    wk = closes.resample(rule).last()
    return wk.pct_change()


def weekly_dollar_volume(panel: dict[str, pd.DataFrame], rule: str = "W-FRI") -> pd.DataFrame:
    """Weekly average dollar volume (Close × Volume) per name — the liquidity yardstick."""
    dv = {}
    for s in panel:
        df = panel[s]
        if "Volume" not in df.columns:
            continue
        dollar = (df["Close"] * df["Volume"]).sort_index()
        dv[s] = dollar.resample(rule).mean()
    if not dv:
        return pd.DataFrame(index=weekly_returns(panel, rule).index)
    return pd.DataFrame(dv).sort_index()


def market_return(wret: pd.DataFrame) -> pd.Series:
    """Equal-weight cross-sectional mean weekly return — the single-factor proxy."""
    return wret.mean(axis=1)


# --------------------------------------------------------------------------- #
# Market-model residual (vectorised rolling regression)
# --------------------------------------------------------------------------- #
def residual_returns(wret: pd.DataFrame, beta_window: int = 52) -> pd.DataFrame:
    """Weekly market-model residual per name, ``r_w − alpha_w − beta_w · mkt_w``.

    ``beta_w`` / ``alpha_w`` come from a trailing ``beta_window``-week rolling OLS of the
    name's weekly return on the equal-weight market, ending at week ``w`` (uses data only
    through ``w``). Vectorised via rolling moments: ``beta = cov(y,x)/var(x)``,
    ``alpha = mean(y) − beta·mean(x)``. The decomposition of week ``w`` is realized
    (contemporaneous); the reversal sort shifts by one week so a position formed for week
    ``w`` uses the residual known at the close of ``w−1`` — zero look-ahead.
    """
    x = market_return(wret)
    y = wret
    xm = x.rolling(beta_window, min_periods=beta_window).mean()
    ym = y.rolling(beta_window, min_periods=beta_window).mean()
    exy = y.mul(x, axis=0).rolling(beta_window, min_periods=beta_window).mean()
    cov = exy.sub(ym.mul(xm, axis=0))
    var = (x * x).rolling(beta_window, min_periods=beta_window).mean() - xm ** 2
    beta = cov.div(var.where(var > 0), axis=0)
    alpha = ym.sub(beta.mul(xm, axis=0))
    resid = y.sub(alpha).sub(beta.mul(x, axis=0))
    return resid


# --------------------------------------------------------------------------- #
# The cross-sectional reversal sort
# --------------------------------------------------------------------------- #
def reversal_spreads(
    signal: pd.DataFrame,
    fwd: pd.DataFrame,
    liq: pd.DataFrame | None = None,
    frac: float = 0.3,
    liq_frac: float = 0.6,
    min_names: int = 8,
) -> pd.DataFrame:
    """Weekly equal-weight loser-minus-winner reversal spread on ``signal``.

    On each week ``w`` names are ranked by ``signal`` known at the close of ``w−1`` (one
    ``shift``). Before ranking, an optional **dollar-volume liquidity screen** keeps only
    the top ``liq_frac`` names by ``liq`` at ``w−1`` (bid-ask bounce is an illiquid-name
    artefact). ``lo`` = mean forward week-``w`` return of the bottom ``frac`` (the
    *losers* — low past signal, the long); ``hi`` = mean of the top ``frac`` (the
    *winners* — the short). ``spread = lo − hi`` (long losers, short winners — reversal).
    Weeks with fewer than ``min_names`` screened-and-ranked names are dropped.
    """
    S = signal.shift(1).to_numpy(dtype=float)              # known at close of w-1
    R = fwd.to_numpy(dtype=float)
    if liq is not None:
        L = liq.shift(1).reindex(columns=signal.columns).to_numpy(dtype=float)
    else:
        L = None
    idx = signal.index
    out_spread, out_lo, out_hi, out_n, out_t = [], [], [], [], []
    for i in range(len(idx)):
        row = S[i]
        valid = ~np.isnan(row) & ~np.isnan(R[i])
        if L is not None:
            lv = L[i]
            liq_ok = ~np.isnan(lv)
            m = valid & liq_ok
            nliq = int(m.sum())
            if nliq >= min_names:
                k_liq = max(min_names, int(np.ceil(nliq * liq_frac)))
                thresh = np.sort(lv[m])[::-1][min(k_liq, nliq) - 1]
                valid = m & (lv >= thresh)
        vidx = np.where(valid)[0]
        n = len(vidx)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * frac)))
        order = vidx[np.argsort(row[vidx], kind="stable")]
        low = order[:k]        # losers  -> long
        high = order[-k:]      # winners -> short
        rr = R[i]
        lo = float(np.mean(rr[low]))
        hi = float(np.mean(rr[high]))
        out_spread.append(lo - hi); out_lo.append(lo); out_hi.append(hi)
        out_n.append(n); out_t.append(idx[i])
    return pd.DataFrame(
        {"spread": out_spread, "lo": out_lo, "hi": out_hi, "n": out_n}, index=out_t
    ).sort_index()


def raw_reversal_spreads(wret: pd.DataFrame, liq: pd.DataFrame | None = None,
                         frac: float = 0.3, liq_frac: float = 0.6,
                         min_names: int = 8) -> pd.DataFrame:
    """The RAW weekly reversal foil — sort on last week's *raw* return."""
    return reversal_spreads(wret, wret, liq, frac, liq_frac, min_names)


def residual_reversal_spreads(wret: pd.DataFrame, liq: pd.DataFrame | None = None,
                              beta_window: int = 52, frac: float = 0.3,
                              liq_frac: float = 0.6, min_names: int = 8) -> pd.DataFrame:
    """The residual reversal — sort on last week's market-model *residual*."""
    resid = residual_returns(wret, beta_window)
    return reversal_spreads(resid, wret, liq, frac, liq_frac, min_names)


# --------------------------------------------------------------------------- #
# Inference primitives (shared house set)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 8) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Headline stats
# --------------------------------------------------------------------------- #
def reversal_stats(spreads: pd.DataFrame, nw_lags: int = 8) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    sp_c = sp[~np.isnan(sp)]
    sharpe = (float(np.nanmean(sp)) / np.nanstd(sp, ddof=1) * np.sqrt(TRADING_WEEKS)
              if len(sp_c) > 1 and np.nanstd(sp, ddof=1) > 0 else float("nan"))
    return {
        "n_weeks": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "lo_bps": float(np.nanmean(spreads["lo"].to_numpy()) * 1e4),
        "hi_bps": float(np.nanmean(spreads["hi"].to_numpy()) * 1e4),
        "welch_t": welch_t(spreads["lo"].to_numpy(), spreads["hi"].to_numpy()),
        "gross_sharpe": sharpe,
        "hit_rate": float(np.nanmean(sp_c > 0)) if len(sp_c) else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    wret: pd.DataFrame,
    liq: pd.DataFrame | None = None,
    beta_window: int = 52,
    frac: float = 0.3,
    liq_frac: float = 0.6,
    min_names: int = 8,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 905,
) -> dict:
    """Keep the residual-reversal sort but read each week's forward return from a
    **column-permuted** panel (signal→outcome link broken, each week's cross-sectional
    distribution preserved). p = share of permuted worlds whose spread mean is >=
    observed (right-tail test on the long-loser/short-winner spread)."""
    cols = list(wret.columns)
    ncol = len(cols)
    resid = residual_returns(wret, beta_window)
    sig = resid.shift(1)
    obs = float(residual_reversal_spreads(
        wret, liq, beta_window, frac, liq_frac, min_names)["spread"].mean())

    ret_mat = wret.to_numpy(dtype=float)
    pos_of = {c: j for j, c in enumerate(cols)}
    Lsh = liq.shift(1) if liq is not None else None
    rows_idx, lows, highs = [], [], []
    row_lookup = {t: r for r, t in enumerate(wret.index)}
    for t in wret.index:
        s = sig.loc[t]
        f = wret.loc[t]
        m = s.notna() & f.notna()
        if Lsh is not None:
            lv = Lsh.loc[t].reindex(s.index)
            mm = m & lv.notna()
            nliq = int(mm.sum())
            if nliq >= min_names:
                k_liq = max(min_names, int(np.ceil(nliq * liq_frac)))
                keep = lv[mm].sort_values(ascending=False).index[:k_liq]
                m = s.index.isin(keep) & m.to_numpy()
                m = pd.Series(m, index=s.index)
        s2 = s[m]
        if len(s2) < min_names:
            continue
        k = max(1, int(np.floor(len(s2) * frac)))
        order = s2.sort_values()
        rows_idx.append(row_lookup[t])
        lows.append(np.array([pos_of[c] for c in order.index[:k]]))
        highs.append(np.array([pos_of[c] for c in order.index[-k:]]))
    rows_idx = np.asarray(rows_idx)

    means = []
    if len(rows_idx):
        M = ret_mat[rows_idx]
        kl = max(len(a) for a in lows)
        kh = max(len(a) for a in highs)

        def _pad(books, kmax):
            P = np.zeros((len(books), kmax), dtype=int)
            V = np.zeros((len(books), kmax), dtype=bool)
            for j, a in enumerate(books):
                P[j, :len(a)] = a
                V[j, :len(a)] = True
            return P, V

        LOW, LOWv = _pad(lows, kl)
        HIGH, HIGHv = _pad(highs, kh)
        rows_ar = np.arange(len(rows_idx))[:, None]

        def _masked_mean(pos, valid, perm):
            vals = M[rows_ar, perm[pos]]
            vals = np.where(valid, vals, np.nan)
            return np.nanmean(vals, axis=1)

        for seed in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed)
            for _ in range(n_draws_per_seed):
                perm = rng.permutation(ncol)
                lo_v = _masked_mean(LOW, LOWv, perm)
                hi_v = _masked_mean(HIGH, HIGHv, perm)
                means.append(np.nanmean(lo_v - hi_v))
    means = np.asarray(means)
    return {
        "obs_bps": obs * 1e4,
        "placebo_mean_bps": float(means.mean() * 1e4) if len(means) else float("nan"),
        "placebo_sd_bps": float(means.std(ddof=1) * 1e4) if len(means) > 1 else float("nan"),
        "p_value": float((means >= obs).mean()) if len(means) else float("nan"),
        "n_draws": len(means),
        "draws_bps": means * 1e4,
    }


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(
    spreads: pd.DataFrame,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
) -> dict:
    """Cost the long-loser / short-winner residual book.

    Weekly reversal turns the book over almost entirely each week (the whole point is to
    trade last week's move), so we charge 2 sides × one-way cost × NAV per rebalance on
    the 2×-NAV long-short book, plus borrow on the short leg — one rebalance per week.
    """
    sp = spreads["spread"].to_numpy(dtype=float)
    sp = sp[~np.isnan(sp)]
    n = len(sp)
    round_trip_cost = 2.0 * cost_bps / 1e4          # both sides, one-way each, per week
    borrow_weekly = (borrow_bps_yr / 1e4) / 52.0
    net = sp - round_trip_cost - borrow_weekly
    gross_mean = float(sp.mean())
    net_mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net_mean / sd * np.sqrt(TRADING_WEEKS) if sd and sd > 0 else float("nan")
    return {
        "n_weeks": n,
        "gross_bps": gross_mean * 1e4,
        "net_bps": net_mean * 1e4,
        "cost_bps_per_week": (round_trip_cost + borrow_weekly) * 1e4,
        "ann_net_pct": net_mean * TRADING_WEEKS * 100,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.DataFrame], beta_window: int = 52,
                     frac: float = 0.3, use_liq: bool = False) -> dict:
    """Run the residual-reversal headline stats on a synthetic panel; also report the raw
    reversal for contrast. ``use_liq=False`` isolates the reversal machinery."""
    wret = weekly_returns(panel)
    liq = weekly_dollar_volume(panel) if use_liq else None
    sp_res = residual_reversal_spreads(wret, liq, beta_window, frac)
    sp_raw = raw_reversal_spreads(wret, liq, frac)
    r = reversal_stats(sp_res)
    raw = reversal_stats(sp_raw)
    return {"spread_bps": r["spread_bps"], "t_nw": r["t_nw"], "welch_t": r["welch_t"],
            "n_weeks": r["n_weeks"], "raw_spread_bps": raw["spread_bps"],
            "raw_t_nw": raw["t_nw"]}
