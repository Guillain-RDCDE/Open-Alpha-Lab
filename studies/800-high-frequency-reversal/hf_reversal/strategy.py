"""Strategy + honest controls — Study 800 (High-Frequency / Weekly Reversal).

The recipe (Jegadeesh 1990; Lehmann 1990, at the one-week horizon): each Friday, rank the
cross-section by *last week's* five-day return; long the bottom quintile (last week's
**losers**), short the top quintile (last week's **winners**); hold one week, rebalance.
The thesis is short-horizon overreaction / liquidity provision — the crowd over-pushes a
one-week move and the laggards snap back next week.

The decisive question for a *weekly* reversal is microstructure. Three honest tests settle it:

1. **Loser vs Winner** — the classic dollar-neutral long-short spread.
2. **The skip-a-week gap (the bid-ask-bounce killer).** ``skip=0`` uses last week's close
   both to *form* the signal and to *price* the entry — the textbook bid-ask-bounce
   illusion (a name that closed on the bid prints a low return and "reverts" upward purely
   because the next print lands on the ask). ``skip=1`` inserts one week between formation
   and holding, defusing the bounce. If the edge only survives ``skip=0`` it is largely
   microstructure, not a tradable signal.
3. **The bid-ask-bounce haircut (empirical).** Charge each leg its OWN Corwin-Schultz
   effective spread on the turnover it generates — losers are illiquid and pay more than a
   flat basis-point assumption admits — plus a flat cost sweep and a short-borrow charge.

Execution / look-ahead: ONE documented lag, applied once in :func:`trailing_return` — the
signal for holding week t is the return of week ``t-1-skip`` (known at that close); the
return earned is week t's. Costs are counted one-way x NAV per leg; the short (winner) leg
additionally pays borrow.

**Survivorship**: the real universe is the current S&P 500 projected backwards; every
positive real-tape spread is an upper bound, named on the Signal axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_WEEKS = 52


# ---------------------------------------------------------------------------
# Inference primitives
# ---------------------------------------------------------------------------
def welch_t(a, b) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def one_sample_t(x) -> float:
    """One-sample t of mean(x) vs 0 (i.i.d. SE)."""
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x) -> tuple[float, int]:
    """Newey-West (Bartlett-kernel) HAC t of mean(x) vs 0, automatic lag length.

    Weekly overlapping-book returns are mildly autocorrelated; the HAC t is the desk's
    serial-correlation-robust primary for a return series. Returns ``(t, lags)``.
    """
    r = np.asarray(x, dtype=float); r = r[~np.isnan(r)]
    n = len(r)
    if n < 3:
        return float("nan"), 0
    mu = r.mean()
    e = r - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return (float(mu / se) if se > 0 else float("nan")), lags


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n (the hit-rate CI)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def trailing_return(prices: pd.DataFrame, skip: int = 0) -> pd.DataFrame:
    """One-week formation return, ``skip`` weeks before the holding week.

    The signal aligned to holding week t is the simple weekly return of week
    ``t - 1 - skip``. ``skip=0`` is last week's return (the same Friday close forms the
    signal AND prices the entry — bid-ask-bounce exposed); ``skip=1`` leaves a one-week
    gap that defuses the bounce. One shift, applied once (the study's whole look-ahead
    discipline). Burn-in rows are NaN.
    """
    ret = prices.pct_change()
    signal = ret.shift(1 + skip)
    signal.index.name = "date"
    return signal


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def quintile_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    spreads: pd.DataFrame | None = None,
    q: float = 0.20,
    min_stocks: int = 20,
) -> pd.DataFrame:
    """Weekly loser-minus-winner spread plus equal-weight market, turnover, leg spreads.

    Each holding week t, names are sorted by ``signal.loc[t]`` into quantiles. The bottom
    ``q`` (last week's losers) is the long leg, the top ``q`` (last week's winners) the
    short leg. The realised return is week t's simple return (the single lag already lives
    in ``trailing_return``, so ``signal.loc[t]`` is strictly pre-period — no extra shift,
    no look-ahead).

    Columns (indexed by date): ``loser, winner, market, spread, loser_turn, winner_turn,
    turn, loser_sprd, winner_sprd, n_total, n_loser, n_winner``. ``*_turn`` is the one-way
    fraction of that leg replaced vs the prior rebalance; ``*_sprd`` is the mean
    Corwin-Schultz effective spread of the leg's holdings that week (NaN if no spread panel).
    """
    fwd = prices.pct_change()
    rows: list[dict] = []
    prev_lo: set = set(); prev_wi: set = set()
    for t in signal.index:
        if t not in fwd.index:
            continue
        s = signal.loc[t].dropna()
        f = fwd.loc[t].dropna()
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            prev_lo = set(); prev_wi = set()
            continue
        s = s.loc[common]; f = f.loc[common]
        lo_th = s.quantile(q); hi_th = s.quantile(1.0 - q)
        lo_mask = s <= lo_th; wi_mask = s >= hi_th
        cur_lo = set(s[lo_mask].index); cur_wi = set(s[wi_mask].index)
        lo_turn = 1.0 - len(prev_lo & cur_lo) / len(cur_lo) if (prev_lo and cur_lo) else 1.0
        wi_turn = 1.0 - len(prev_wi & cur_wi) / len(cur_wi) if (prev_wi and cur_wi) else 1.0
        prev_lo, prev_wi = cur_lo, cur_wi

        lo_sprd = wi_sprd = float("nan")
        if spreads is not None and t in spreads.index:
            sp = spreads.loc[t]
            lo_sprd = float(sp.reindex(list(cur_lo)).mean())
            wi_sprd = float(sp.reindex(list(cur_wi)).mean())

        rows.append({
            "date": t,
            "loser": float(f[lo_mask].mean()),
            "winner": float(f[wi_mask].mean()),
            "market": float(f.mean()),
            "spread": float(f[lo_mask].mean() - f[wi_mask].mean()),
            "loser_turn": float(lo_turn), "winner_turn": float(wi_turn),
            "turn": float((lo_turn + wi_turn) / 2.0),
            "loser_sprd": lo_sprd, "winner_sprd": wi_sprd,
            "n_total": int(len(common)), "n_loser": int(lo_mask.sum()),
            "n_winner": int(wi_mask.sum()),
        })
    res = pd.DataFrame(rows).set_index("date")
    res.index = pd.DatetimeIndex(res.index)
    return res


def random_portfolio_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.20,
    n_draws: int = 100,
    seed: int = 800,
    min_stocks: int = 20,
) -> pd.Series:
    """Null distribution of random-quantile excess returns (loser-leg size).

    Each week, draw ``n_draws`` random subsets the size of the loser leg and record their
    excess over the equal-weight market. Centred at zero by construction, so any loser
    excess beyond this cloud is signal, not concentration."""
    rng = np.random.default_rng(seed)
    fwd = prices.pct_change()
    out: list[float] = []
    for t in signal.index:
        if t not in fwd.index:
            continue
        s = signal.loc[t].dropna(); f = fwd.loc[t].dropna()
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            continue
        f = f.loc[common]; m = f.to_numpy(); mu = m.mean()
        n_lo = max(1, int(round(len(common) * q))); idx = np.arange(len(m))
        for _ in range(n_draws):
            pick = rng.choice(idx, size=n_lo, replace=False)
            out.append(float(m[pick].mean() - mu))
    return pd.Series(out, name="random_excess")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(weekly_ret: pd.Series) -> dict:
    """Headline stats for a weekly return series, with a Newey-West HAC t (the primary)."""
    r = pd.Series(weekly_ret).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "lags",
                                     "hit_rate", "max_drawdown", "n")}
    mu = float(r.mean()); std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")
    tstat, lags = newey_west_t(r.to_numpy())
    eq = (1.0 + r).cumprod(); max_dd = float((eq / eq.cummax() - 1.0).min())
    return {"mean": mu, "vol": std, "sharpe": sr, "tstat": tstat, "lags": lags,
            "hit_rate": float((r > 0).mean()), "max_drawdown": max_dd, "n": n}


def annualise(stats: dict, periods: int = TRADING_WEEKS) -> dict:
    """Weekly stats -> annualised mean / vol / Sharpe."""
    out = dict(stats)
    if np.isfinite(out.get("mean", np.nan)):
        out["mean_ann"] = float(out["mean"] * periods)
    if np.isfinite(out.get("vol", np.nan)):
        out["vol_ann"] = float(out["vol"] * np.sqrt(periods))
    if out.get("vol_ann", 0):
        out["sharpe_ann"] = float(out["mean_ann"] / out["vol_ann"])
    return out


def beta_alpha(leg: pd.Series, market: pd.Series, periods: int = TRADING_WEEKS) -> tuple[float, float]:
    """Market beta and annualised alpha of ``leg`` on ``market`` (both weekly)."""
    common = leg.dropna().index.intersection(market.dropna().index)
    x = market.loc[common].to_numpy(); y = leg.loc[common].to_numpy()
    if len(x) < 3 or np.var(x) == 0:
        return float("nan"), float("nan")
    beta = float(np.cov(y, x)[0, 1] / np.var(x))
    alpha = float((y.mean() - beta * x.mean()) * periods)
    return beta, alpha


# ---------------------------------------------------------------------------
# The costed timer — flat cost sweep, borrow, and the empirical bounce haircut
# ---------------------------------------------------------------------------
def net_spread(res: pd.DataFrame, one_way_bps: float, borrow_bps_annual: float = 0.0) -> pd.Series:
    """Dollar-neutral spread net of a FLAT one-way cost (+ optional short borrow).

    Both legs rebalance weekly; each leg is charged ``turn x one_way_bps`` one-way x NAV,
    round-trip (buy in, sell out) — ``drag = 2 legs x turn x 2 (round trip) x bps``. The
    short (winner) leg additionally pays borrow ``borrow_bps_annual / 52`` every week held.
    """
    turn = res["turn"].reindex(res.index).fillna(res["turn"].mean())
    drag = 2.0 * turn * 2.0 * one_way_bps * 1e-4
    borrow = (borrow_bps_annual / TRADING_WEEKS) * 1e-4
    return (res["spread"] - drag - borrow).rename("net_spread")


def bounce_haircut_spread(res: pd.DataFrame, borrow_bps_annual: float = 0.0,
                          spread_mult: float = 1.0) -> pd.Series:
    """Spread net of the EMPIRICAL bid-ask-bounce haircut — each leg pays its OWN spread.

    Instead of a flat basis-point assumption, charge each leg the Corwin-Schultz effective
    spread of the names it actually holds (``loser_sprd`` / ``winner_sprd``), scaled by the
    fraction of the leg turned over that week and by ``spread_mult`` (the share of the
    roundtrip spread actually crossed — 1.0 = pay the full estimated spread). This is the
    honest microstructure haircut: illiquid losers pay more than winners, so it bites the
    long leg hardest. The short leg additionally pays borrow.
    """
    lo_cost = res["loser_sprd"].fillna(res["loser_sprd"].mean()) * res["loser_turn"] * spread_mult
    wi_cost = res["winner_sprd"].fillna(res["winner_sprd"].mean()) * res["winner_turn"] * spread_mult
    borrow = (borrow_bps_annual / TRADING_WEEKS) * 1e-4
    return (res["spread"] - lo_cost - wi_cost - borrow).rename("bounce_net_spread")


def break_even_cost(res: pd.DataFrame) -> float:
    """Flat one-way cost (bps) at which the mean net spread hits zero."""
    to = float(res["turn"].mean()); mu = float(res["spread"].mean())
    if to <= 0:
        return float("nan")
    return mu / (2.0 * to * 2.0) * 1e4


# ---------------------------------------------------------------------------
# Synthetic-control detector (the machinery proof)
# ---------------------------------------------------------------------------
def detect_spread(prices: pd.DataFrame, skip: int = 0, q: float = 0.20) -> dict:
    """Run the loser-minus-winner quintile spread on a panel; return its HAC summary."""
    res = quintile_returns(trailing_return(prices, skip=skip), prices, q=q)
    s = summarize(res["spread"].dropna())
    return {"mean_bps": s["mean"] * 1e4, "tstat": s["tstat"], "n": s["n"]}
