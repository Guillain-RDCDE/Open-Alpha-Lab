"""Strategy + inference for Study 766 — Memecoin-Season.

The claim (crypto-Twitter folklore, every cycle since 2021): in a "memecoin season" the
dog-coins blow past Bitcoin by orders of magnitude, and a nimble **momentum rotation** — each
week, hop onto whichever of {BTC, DOGE, SHIB} has run hardest lately — harvests the mania net
of costs, beating a boring BTC HODL.

We test the literal, mechanical version:

* **The rotation.** Weekly bars. At each Friday close, rank {BTC, DOGE, SHIB} by trailing
  ``lookback``-week return (the momentum signal, known at that close). Hold the single top
  coin over the **next** week (one-week execution lag — no look-ahead). Optionally sit in cash
  if every coin's momentum is negative ("risk-off" variant). Costs charged per leg × NAV; a
  full switch from one coin to another is two legs.

* **The benchmarks.** (a) BTC buy-and-hold — the thing the claim says you beat. (b) An
  equal-weight, weekly-rebalanced BTC/DOGE/SHIB basket — the naive "just own the memecoins"
  alternative. Both over the identical window the rotation could act in.

* **Is the edge real?** The paired *t*-stat of the rotation's weekly return **in excess of
  BTC** (the desk bar: |t| ≥ 2). Plus a **random-rotation placebo** — each week pick a coin at
  random (thousands of seeds) — to ask whether the *momentum signal* adds anything over merely
  being randomly parked in one of three crypto assets.

* **Where does any edge live?** A pre-2022 (the 2021 mania) vs 2022-onward split, because a
  single euphoric year can carry an entire backtest.

* **Synthetic positive control** — a 3-asset weekly world with TUNABLE momentum persistence;
  the rotation must beat equal-weight only when persistence is actually planted.

**Survivorship, on the Signal axis:** DOGE and SHIB are the two memecoins that *survived* out
of thousands. Every number here is an upper bound a real-time trader could not have banked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52
CASH = "CASH"


# --------------------------------------------------------------------------- #
# Core rotation engine
# --------------------------------------------------------------------------- #
def weekly_returns(weekly_px: pd.DataFrame) -> pd.DataFrame:
    """Simple weekly returns per asset (first row dropped)."""
    return weekly_px.pct_change().dropna(how="all")


def momentum_signal(weekly_px: pd.DataFrame, lookback: int = 4) -> pd.DataFrame:
    """Trailing ``lookback``-week simple return per asset; value at week t is known at t's close."""
    return weekly_px.pct_change(lookback)


def rotation_choice(weekly_px: pd.DataFrame, lookback: int = 4,
                    cash_option: bool = False) -> pd.Series:
    """The coin chosen AT each week t (to be held over week t+1): the top trailing-momentum asset.

    With ``cash_option``, weeks where *every* asset's trailing momentum is <= 0 choose CASH.
    Returns a Series of asset names (or CASH) indexed by the decision week.
    """
    mom = momentum_signal(weekly_px, lookback).dropna(how="all")
    choice = mom.idxmax(axis=1)
    if cash_option:
        all_neg = (mom <= 0).all(axis=1)
        choice = choice.where(~all_neg, CASH)
    return choice.dropna()


def _one_hot(held: pd.Series, assets: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Weight matrix: 1.0 in the held asset's column each week, 0 elsewhere (all-0 row for CASH)."""
    w = pd.DataFrame(0.0, index=index, columns=assets)
    for t, a in held.items():
        if t in w.index and a in assets:
            w.loc[t, a] = 1.0
    return w


def run_rotation(weekly_px: pd.DataFrame, lookback: int = 4, cost_bps: float = 30.0,
                 cash_option: bool = False) -> dict:
    """Momentum rotation backtest, gross and net of costs, one-week execution lag.

    The coin chosen at week t's close is held over week t+1 (``choice.shift(1)``) — the single
    documented lag, no look-ahead. Costs: ``cost_bps`` per leg × NAV, charged on the absolute
    weight change each week (a full coin-to-coin switch = 2 legs = 2 × cost_bps; a first entry
    from cash = 1 leg). Returns weekly gross/net series, the held-asset series, and turnover.
    """
    rets = weekly_returns(weekly_px)
    assets = list(weekly_px.columns)
    choice = rotation_choice(weekly_px, lookback, cash_option)
    held = choice.shift(1).dropna()                      # held during week t = chosen at t-1
    held = held[held.index.isin(rets.index)]
    w = _one_hot(held, assets, rets.index).loc[held.index]
    r = rets.loc[held.index]

    gross = (w * r).sum(axis=1)
    dw = w.diff()
    dw.iloc[0] = w.iloc[0]                                # first entry = 1 leg
    turnover = dw.abs().sum(axis=1)                       # legs traded (0, 1 or 2) per week
    cost = turnover * (cost_bps / 1e4)
    net = gross - cost
    return {
        "gross_ret": gross, "net_ret": net, "cost": cost,
        "held": held, "turnover": turnover,
        "n_switches": int((turnover >= 2).sum()),
        "avg_turnover": float(turnover.mean()),
    }


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
def btc_hodl(weekly_px: pd.DataFrame) -> pd.Series:
    """BTC buy-and-hold weekly returns (the benchmark the claim says you beat)."""
    return weekly_returns(weekly_px)["BTC"]


def equal_weight(weekly_px: pd.DataFrame) -> pd.Series:
    """Equal-weight, weekly-rebalanced BTC/DOGE/SHIB basket (gross — the naive 'own them all')."""
    return weekly_returns(weekly_px).mean(axis=1)


# --------------------------------------------------------------------------- #
# Summary + inference
# --------------------------------------------------------------------------- #
def summarize(weekly_ret: pd.Series) -> dict:
    """Total return, CAGR, annualised vol, Sharpe, max drawdown, hit-rate for a weekly series."""
    r = weekly_ret.dropna()
    wealth = (1.0 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25 if len(r) > 1 else np.nan
    total = float(wealth.iloc[-1] - 1.0)
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)) if r.std(ddof=1) > 0 else np.nan
    dd = float((wealth / wealth.cummax() - 1.0).min())
    return {
        "n": int(len(r)), "years": float(years), "total_pct": total * 100,
        "cagr_pct": cagr * 100, "vol_pct": vol * 100, "sharpe": sharpe,
        "maxdd_pct": dd * 100, "hit_rate": float((r > 0).mean()),
    }


def excess_tstat(strat: pd.Series, bench: pd.Series) -> dict:
    """Paired t-stat of mean weekly (strat - bench). |t| >= 2 is the desk's REAL bar."""
    d = (strat - bench).dropna()
    if len(d) < 3 or d.std(ddof=1) == 0:
        return {"mean_excess_pct": float("nan"), "t": float("nan"), "n": len(d)}
    t = float(d.mean() / d.std(ddof=1) * np.sqrt(len(d)))
    return {"mean_excess_pct": float(d.mean()) * 100, "t": t, "n": int(len(d))}


# --------------------------------------------------------------------------- #
# Random-rotation placebo
# --------------------------------------------------------------------------- #
def random_rotation_placebo(weekly_px: pd.DataFrame, cost_bps: float = 30.0,
                            n_seeds: int = 4000, base_seed: int = 766) -> dict:
    """Each week pick a coin uniformly at random (no momentum); repeat ``n_seeds`` times.

    Same cost model and one-week lag as the real rotation. Reports the distribution of random
    rotations' total return and Sharpe, and the share that match/beat the momentum rotation
    (right-tailed p-values) — i.e. does the momentum *signal* add anything over coin-flip
    allocation among the same three assets?
    """
    rets = weekly_returns(weekly_px)
    assets = list(weekly_px.columns)
    idx = rets.index
    n = len(idx)
    mom = run_rotation(weekly_px, cost_bps=cost_bps)
    mom_total = summarize(mom["net_ret"])["total_pct"]
    mom_sharpe = summarize(mom["net_ret"])["sharpe"]

    totals, sharpes = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        pick = rng.integers(0, len(assets), size=n)
        chosen = np.array(assets)[pick]
        w = pd.DataFrame(0.0, index=idx, columns=assets)
        for i, a in enumerate(chosen):
            w.iloc[i, assets.index(a)] = 1.0
        gross = (w * rets).sum(axis=1)
        dw = w.diff()
        dw.iloc[0] = w.iloc[0]
        cost = dw.abs().sum(axis=1) * (cost_bps / 1e4)
        net = gross - cost
        totals.append(summarize(net)["total_pct"])
        sharpes.append(summarize(net)["sharpe"])
    totals, sharpes = np.asarray(totals), np.asarray(sharpes)
    return {
        "n_seeds": n_seeds,
        "mom_total_pct": mom_total, "mom_sharpe": mom_sharpe,
        "rand_total_median_pct": float(np.median(totals)),
        "rand_sharpe_median": float(np.median(sharpes)),
        "p_total": float((totals >= mom_total).mean()),
        "p_sharpe": float((sharpes >= mom_sharpe).mean()),
    }


# --------------------------------------------------------------------------- #
# Sub-period split
# --------------------------------------------------------------------------- #
def subperiod_table(weekly_px: pd.DataFrame, cut: str = "2022-01-01",
                    cost_bps: float = 30.0) -> pd.DataFrame:
    """Rotation (net) vs BTC HODL, split into the pre-``cut`` mania and the post-``cut`` era."""
    mom = run_rotation(weekly_px, cost_bps=cost_bps)
    net, btc = mom["net_ret"], btc_hodl(weekly_px)
    cut_ts = pd.Timestamp(cut)
    rows = []
    for label, mask in [("mania (< %s)" % cut[:7], net.index < cut_ts),
                        ("after (>= %s)" % cut[:7], net.index >= cut_ts)]:
        n_seg = net[mask]
        b_seg = btc.reindex(n_seg.index)
        rows.append({
            "segment": label, "weeks": int(len(n_seg)),
            "rot_total_pct": summarize(n_seg)["total_pct"],
            "rot_sharpe": summarize(n_seg)["sharpe"],
            "btc_total_pct": summarize(b_seg)["total_pct"],
            "btc_sharpe": summarize(b_seg)["sharpe"],
            "excess_t": excess_tstat(n_seg, b_seg)["t"],
        })
    return pd.DataFrame(rows).set_index("segment")


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def momentum_edge_from_returns(weekly_rets: pd.DataFrame, lookback: int = 4,
                               cost_bps: float = 0.0) -> float:
    """Excess t of a momentum rotation over equal-weight, given a weekly-RETURN frame.

    Rebuilds synthetic 'prices' by compounding, runs the same rotation logic, and returns the
    paired t-stat of (rotation_net - equal_weight). Used by the synthetic control: ~0 on a
    random-walk null, strongly positive when momentum persistence is planted.
    """
    px = (1.0 + weekly_rets).cumprod()
    px.index = pd.date_range("2000-01-07", periods=len(px), freq="W-FRI")
    mom = run_rotation(px, lookback=lookback, cost_bps=cost_bps)
    ew = equal_weight(px).reindex(mom["net_ret"].index)
    return excess_tstat(mom["net_ret"], ew)["t"]
