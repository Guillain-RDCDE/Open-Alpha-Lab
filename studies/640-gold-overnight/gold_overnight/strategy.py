"""Rules + inference for Study 640 — Gold-Overnight ("Gold Trades in Its Sleep").

The claim: for two decades gold rose **overnight** (prior close → open, when US exchanges
are shut and price discovery runs through Asia and the London AM fix) and went nowhere or
fell **during London/NY hours** (open → close, the window bracketing the London PM fix) —
an anomaly the literature parks around the London gold fixes (Caminschi & Heaney 2014;
the Adrian Douglas "buy the PM fix, sell the AM fix" folklore chart).

Measurements:

* **Session split.** Mean overnight vs intraday return in bps/day, each with a
  Newey-West (HAC) *t* on the daily mean; the decisive statistic is the HAC *t* of the
  **paired daily difference** (overnight − intraday), plus a pooled Welch *t* of the two
  legs as the group-split cross-check.
* **Sign-flip placebo.** Under the null that night and day are exchangeable labels within
  a session, flipping each day's difference sign at random must reproduce the observed gap
  as often as chance allows. A deterministic, seeded 20,000-draw permutation test — a
  *distributional* null, not a single-seed baseline.
* **Sub-period decay.** Pre vs post the 2015-03-20 LBMA fix reform (an externally-dated,
  pre-registered break — the fix mechanism IS the claim), with a Welch *t* on the
  *difference between the sub-period gaps*.
* **The equity placebo.** SPY's own night/day gap on the identical calendar (study
  01-overnight-anomaly showed equities do this too); Welch *t* of gold's daily difference
  series against SPY's answers "is gold's bigger, or is this just the market-wide clock
  effect?".
* **Harvest test (third axis).** A long-only overnight overlay — buy the official close
  (MOC), sell the next open (MOO), flat all day. The rule is an unconditional calendar
  rule: it uses no information beyond the clock, so the standard one-bar execution lag is
  the MOC/MOO convention itself (documented; no same-bar cheat is possible because nothing
  is conditioned on). Costs: **2 one-way trades per day** × cost_bps × NAV. No short leg
  (the short-day variant would add 2 more legs plus borrow; the long-only overlay is the
  cheapest honest harvest, so if IT dies, the short version dies faster).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (Bartlett) t of mean(x) vs 0; lags defaults to the 4(n/100)^(2/9) rule."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 8:
        return float("nan")
    if lags is None:
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    e = x - x.mean()
    s = float(np.dot(e, e)) / n
    for l in range(1, lags + 1):
        g = float(np.dot(e[l:], e[:-l])) / n
        s += 2.0 * (1.0 - l / (lags + 1.0)) * g
    se = np.sqrt(max(s, 1e-300) / n)
    return float(x.mean() / se)


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) − mean(b) (unequal variance). NaN if either side < 2."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / se)


def signflip_pvalue(diff: np.ndarray, n_draws: int = 20_000, seed: int = 640) -> dict:
    """Sign-flip permutation p for the mean of the paired daily difference.

    Null: within a session, "overnight" and "intraday" are exchangeable labels, so each
    day's difference is symmetric around 0 — flip signs at random and ask how often the
    shuffled mean gap ≥ the observed one. Deterministic seed; chunked so 20,000 × n stays
    light. This is a permutation *distribution*, not a strategy baseline (no seed-picking).
    """
    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    chunk = 2_000
    for lo in range(0, n_draws, chunk):
        m = min(chunk, n_draws - lo)
        signs = rng.integers(0, 2, size=(m, n), dtype=np.int8) * 2 - 1
        means[lo:lo + m] = (signs.astype(np.float64) @ d) / n
    return {"obs": obs, "p_value": float((means >= obs).mean()), "draws": means}


# --------------------------------------------------------------------------- #
# Session-split statistics
# --------------------------------------------------------------------------- #
def split_stats(rets: pd.DataFrame, n_draws: int = 20_000, placebo: bool = True,
                seed: int = 640) -> dict:
    """Headline night-vs-day stats for one ticker's session-return frame.

    ``rets`` comes from ``data.session_returns`` (columns overnight / intraday /
    close2close). Means in bps/day; annualised legs are compounded (what a sleeve holding
    only that leg would have earned, gross); HAC t on each leg's mean and on the paired
    daily difference (the primary statistic); Welch t as the group cross-check.
    """
    on = rets["overnight"].to_numpy()
    idr = rets["intraday"].to_numpy()
    d = on - idr
    out = {
        "n_days": int(len(rets)),
        "start": str(rets.index.min().date()), "end": str(rets.index.max().date()),
        "on_bps": float(on.mean() * 1e4), "id_bps": float(idr.mean() * 1e4),
        "gap_bps": float(d.mean() * 1e4),
        "t_on": hac_t(on), "t_id": hac_t(idr), "t_gap": hac_t(d),
        "welch_gap": welch_t(on, idr),
        "on_ann_pct": float((np.exp(np.log1p(on).sum() / len(on) * TRADING_DAYS) - 1) * 100),
        "id_ann_pct": float((np.exp(np.log1p(idr).sum() / len(idr) * TRADING_DAYS) - 1) * 100),
        "cc_ann_pct": float((np.exp(np.log1p(rets["close2close"].to_numpy()).sum()
                                    / len(rets) * TRADING_DAYS) - 1) * 100),
        "on_total_x": float(np.exp(np.log1p(on).sum())),
        "id_total_x": float(np.exp(np.log1p(idr).sum())),
    }
    if placebo:
        out["p_signflip"] = signflip_pvalue(d, n_draws=n_draws, seed=seed)["p_value"]
    return out


def subperiod_stats(rets: pd.DataFrame, split_date: str) -> dict:
    """Pre/post split of the night-minus-day gap + a Welch t on the CHANGE in the gap.

    The decay claim ("the anomaly faded after the fix reform") is a claim about the
    *difference between sub-period gaps*, so it gets its own test: Welch t of the pre-period
    daily difference series against the post-period one — per the house rule that sub-period
    contrasts carry a test of the difference, on a split that is externally dated (the
    2015-03-20 LBMA reform), not snooped.
    """
    ts = pd.Timestamp(split_date)
    pre = rets[rets.index < ts]
    post = rets[rets.index >= ts]
    d_pre = (pre["overnight"] - pre["intraday"]).to_numpy()
    d_post = (post["overnight"] - post["intraday"]).to_numpy()
    return {
        "split": split_date,
        "pre": split_stats(pre, placebo=False),
        "post": split_stats(post, placebo=False),
        "welch_change": welch_t(d_pre, d_post),   # >0 means the gap SHRANK after the split
    }


def vs_placebo(gold: pd.DataFrame, spy: pd.DataFrame) -> dict:
    """Gold's night-minus-day gap vs SPY's, on the common calendar (the equity placebo).

    Welch t of gold's daily difference series against SPY's: is gold's clock effect bigger
    than the market-wide one study 01 already documented, or the same phenomenon?
    """
    idx = gold.index.intersection(spy.index)
    dg = (gold.loc[idx, "overnight"] - gold.loc[idx, "intraday"]).to_numpy()
    ds = (spy.loc[idx, "overnight"] - spy.loc[idx, "intraday"]).to_numpy()
    return {
        "n_days": int(len(idx)),
        "gold_gap_bps": float(dg.mean() * 1e4), "spy_gap_bps": float(ds.mean() * 1e4),
        "welch_gold_vs_spy": welch_t(dg, ds),
        "t_gold": hac_t(dg), "t_spy": hac_t(ds),
    }


# --------------------------------------------------------------------------- #
# Harvest test — the third axis
# --------------------------------------------------------------------------- #
def harvest(rets: pd.DataFrame, cost_bps: float) -> dict:
    """Long-only overnight overlay: buy MOC, sell next MOO, flat all day, vs buy & hold.

    Gross per-day return = the overnight leg. Costs: 2 one-way trades per day (enter at the
    close, exit at the open) × ``cost_bps`` × NAV. Buy & hold pays nothing after the initial
    entry (charged as ~0 over 20 years). Both legs are gross of financing (cash sleeve
    earns nothing here — conservative for the overlay, which holds cash all day). Reports
    annualised compounded net vs buy & hold and a HAC t on the net daily mean.
    """
    on = rets["overnight"].to_numpy()
    c = cost_bps / 1e4
    net = on - 2.0 * c
    n = len(net)
    return {
        "cost_bps": float(cost_bps),
        "gross_bps_day": float(on.mean() * 1e4),
        "net_bps_day": float(net.mean() * 1e4),
        "net_ann_pct": float((np.exp(np.log1p(net).sum() / n * TRADING_DAYS) - 1) * 100),
        "bh_ann_pct": float((np.exp(np.log1p(rets["close2close"].to_numpy()).sum()
                                    / n * TRADING_DAYS) - 1) * 100),
        "t_net": hac_t(net),
    }


def harvest_table(rets: pd.DataFrame, costs=(0.5, 1.0, 2.0, 5.0)) -> list[dict]:
    """The harvest test across realistic one-way cost levels."""
    return [harvest(rets, cb) for cb in costs]
