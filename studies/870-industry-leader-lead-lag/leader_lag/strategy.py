"""Strategy + inference for Study 870 — Industry-Leader Lead-Lag.

The claim (Kewei **Hou 2007**): information diffuses **within an industry** from the
*biggest* name outward. The largest-cap firm in a sector prices news first; the smaller
**followers** react with a lag, so the **leader's return this week predicts the
followers' return next week**. Trade it: each week, long the followers whose sector
leader *rose*, short the followers whose leader *fell* — collect the slow-diffusion
premium.

This is distinct from:

* [379-etf-lead-lag](../../379-etf-lead-lag/) — a **basket-vs-member** lead-lag (does the
  sector *ETF* lead its constituents?), a fund-flow / index-arb channel, not a
  **largest-firm-within-industry** diffusion between individual names;
* [506-industry-momentum](../../506-industry-momentum/) — sorts **industries against each
  other** on their own trailing returns (Moskowitz-Grinblatt), a cross-industry
  time-series signal, not a **within-industry leader→follower** cross-sectional link;
* [538-industry-relative-reversal](../../538-industry-relative-reversal/) — a name's
  **own** deviation from its industry mean *reverses*, a contrarian own-return signal,
  not a leader's return **predicting a different name** in the same industry;
* [810-price-delay](../../810-price-delay/) — how slowly a name absorbs the **market**
  factor (a name-level R² decomposition), not one industry name leading another.

Method:

* **Weekly returns.** Resample each name's daily total-return Close to Friday and take
  the weekly simple return (index = week-ending Friday, columns = ticker).
* **Point-in-time lead-lag.** For each sector, read the **leader's** return in week
  ``w`` (known at that Friday's close) and the **followers'** mean return in week
  ``w+1``. Long the sector's followers when the leader rose, short them when it fell —
  one documented week of execution lag, zero look-ahead into returns.
* **The spread.** The weekly book return is the per-sector sign-weighted mean:
  ``mean_s sign(leader_ret[s, w]) * mean_followers(ret[s, w+1])`` — +1 NAV on up-leader
  sectors' followers, −1 on down-leader sectors', averaged over sectors.
* **Inference.** Newey-West (HAC) *t* on the weekly spread; a one-sample *t* and a
  pooled Welch *t* (followers after an up-leader week vs a down-leader week) cross-check;
  a permutation placebo breaks the lead→lag time alignment; a costed timer charges the
  weekly round-trip friction plus borrow on the short leg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_WEEKS = 52


# --------------------------------------------------------------------------- #
# Weekly return panel
# --------------------------------------------------------------------------- #
def weekly_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Weekly simple returns per name (index = week-ending Friday, columns = ticker).

    Daily total-return Close resampled to the last observation of each calendar week,
    then ``pct_change``. A week with no observation for a name yields NaN there.
    """
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    wk = closes.resample("W-FRI").last()
    return wk.pct_change()


def dollar_volume(panel: dict[str, pd.DataFrame]) -> pd.Series:
    """Median daily dollar volume (Close × Volume) per name — a size/liquidity proxy
    used only to *re-designate* leaders as a robustness check (the headline uses the
    documented static market-cap ``LEADERS`` map)."""
    out = {}
    for s in panel:
        df = panel[s]
        if "Volume" in df.columns:
            out[s] = float((df["Close"] * df["Volume"]).median())
        else:
            out[s] = float("nan")
    return pd.Series(out)


def designate_leaders(sectors: dict[str, list[str]], size: pd.Series) -> dict[str, str]:
    """Leader = largest-``size`` member present in each sector (robustness alternative
    to the static ``LEADERS`` map). ``size`` is any per-name size proxy (e.g. median
    dollar volume)."""
    leaders: dict[str, str] = {}
    for sec, names in sectors.items():
        avail = [n for n in names if n in size.index and np.isfinite(size.get(n, np.nan))]
        if avail:
            leaders[sec] = max(avail, key=lambda n: size[n])
    return leaders


# --------------------------------------------------------------------------- #
# The lead-lag book -> weekly long-followers-of-up-leaders spread
# --------------------------------------------------------------------------- #
def leadlag_spreads(
    wret: pd.DataFrame,
    sectors: dict[str, list[str]],
    leaders: dict[str, str],
    min_sectors: int = 3,
) -> pd.DataFrame:
    """Weekly equal-per-sector lead-lag spread.

    For each sector ``s`` with a leader present, ``lead_w`` is the leader's week-``w``
    return and ``foll_next`` the followers' mean week-``w+1`` return. The book takes
    ``sign(lead_w)`` NAV on that sector's followers, held the following week. The weekly
    spread is the mean over sectors of ``sign(lead_w) * foll_next``; ``up`` / ``dn`` are
    the mean follower next-week returns after an up- / down-leader week (the long / short
    legs). Weeks with fewer than ``min_sectors`` valid sectors are dropped.
    """
    cols = list(wret.columns)
    pos = {c: i for i, c in enumerate(cols)}
    R = wret.to_numpy(dtype=float)          # (T weeks, N names)
    T = R.shape[0]

    # per-sector leader column index and follower column indices
    valid_secs = []
    for sec, names in sectors.items():
        ld = leaders.get(sec)
        if ld is None or ld not in pos:
            continue
        fol = [pos[n] for n in names if n != ld and n in pos]
        if not fol:
            continue
        valid_secs.append((pos[ld], np.asarray(fol, dtype=int)))
    if not valid_secs:
        return pd.DataFrame(columns=["spread", "up", "dn", "n_sectors"])

    S = len(valid_secs)
    lead = np.full((T, S), np.nan)          # leader return week w
    foll = np.full((T, S), np.nan)          # follower mean return week w
    for k, (li, fi) in enumerate(valid_secs):
        lead[:, k] = R[:, li]
        foll[:, k] = np.nanmean(R[:, fi], axis=1)

    sign = np.sign(lead[:-1])               # sign of leader in week w  (T-1, S)
    foll_next = foll[1:]                    # follower mean in week w+1 (T-1, S)
    contrib = sign * foll_next              # 0 where sign==0 or nan-propagated
    contrib = np.where(sign == 0, np.nan, contrib)

    idx = wret.index[1:]
    n_valid = np.sum(np.isfinite(contrib), axis=1)
    keep = n_valid >= min_sectors

    spread = np.nanmean(np.where(np.isfinite(contrib), contrib, np.nan), axis=1)
    up = np.nanmean(np.where(sign > 0, foll_next, np.nan), axis=1)
    dn = np.nanmean(np.where(sign < 0, foll_next, np.nan), axis=1)

    out = pd.DataFrame(
        {"spread": spread, "up": up, "dn": dn, "n_sectors": n_valid},
        index=idx,
    )
    return out[keep].dropna(subset=["spread"]).sort_index()


def leg_pools(
    wret: pd.DataFrame,
    sectors: dict[str, list[str]],
    leaders: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled follower next-week returns split by leader direction (for the Welch test):
    ``(after_up, after_down)``."""
    cols = list(wret.columns)
    pos = {c: i for i, c in enumerate(cols)}
    R = wret.to_numpy(dtype=float)
    up_vals, dn_vals = [], []
    for sec, names in sectors.items():
        ld = leaders.get(sec)
        if ld is None or ld not in pos:
            continue
        fol = [pos[n] for n in names if n != ld and n in pos]
        if not fol:
            continue
        lead = R[:-1, pos[ld]]
        fnext = np.nanmean(R[1:][:, fol], axis=1)
        up_vals.append(fnext[lead > 0])
        dn_vals.append(fnext[lead < 0])
    up = np.concatenate(up_vals) if up_vals else np.array([])
    dn = np.concatenate(dn_vals) if dn_vals else np.array([])
    return up[np.isfinite(up)], dn[np.isfinite(dn)]


# --------------------------------------------------------------------------- #
# Inference primitives  (copied from the desk's canonical set, study 803)
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


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
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
def leadlag_stats(
    spreads: pd.DataFrame,
    wret: pd.DataFrame | None = None,
    sectors: dict[str, list[str]] | None = None,
    leaders: dict[str, str] | None = None,
    nw_lags: int = 6,
) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    out = {
        "n_weeks": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "up_bps": float(np.nanmean(spreads["up"].to_numpy()) * 1e4),
        "dn_bps": float(np.nanmean(spreads["dn"].to_numpy()) * 1e4),
    }
    if wret is not None and sectors is not None and leaders is not None:
        up, dn = leg_pools(wret, sectors, leaders)
        out["welch_t"] = welch_t(up, dn)
    else:
        out["welch_t"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky time alignment of leader->follower?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    wret: pd.DataFrame,
    sectors: dict[str, list[str]],
    leaders: dict[str, str],
    min_sectors: int = 3,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 870,
) -> dict:
    """Keep the sector structure but **shuffle the week-to-week alignment** of the
    leader sign relative to the followers' next-week return (the lead→lag link is
    broken; each series' own distribution is preserved). p = share of permuted worlds
    whose spread mean is >= observed (right-tail test)."""
    cols = list(wret.columns)
    pos = {c: i for i, c in enumerate(cols)}
    R = wret.to_numpy(dtype=float)
    T = R.shape[0]

    valid_secs = []
    for sec, names in sectors.items():
        ld = leaders.get(sec)
        if ld is None or ld not in pos:
            continue
        fol = [pos[n] for n in names if n != ld and n in pos]
        if fol:
            valid_secs.append((pos[ld], np.asarray(fol, dtype=int)))
    if not valid_secs:
        return {"obs_bps": float("nan"), "p_value": float("nan"), "n_draws": 0,
                "placebo_mean_bps": float("nan"), "placebo_sd_bps": float("nan"),
                "draws_bps": np.array([])}

    S = len(valid_secs)
    lead = np.full((T, S), np.nan)
    foll = np.full((T, S), np.nan)
    for k, (li, fi) in enumerate(valid_secs):
        lead[:, k] = R[:, li]
        foll[:, k] = np.nanmean(R[:, fi], axis=1)

    obs = float(leadlag_spreads(wret, sectors, leaders, min_sectors)["spread"].mean())

    sign_all = np.sign(lead)                 # (T, S)
    foll_next = foll[1:]                     # (T-1, S)
    m = foll_next.shape[0]

    def _spread_for(sign_rows):
        contrib = np.where(sign_rows == 0, np.nan, sign_rows * foll_next)
        n_valid = np.sum(np.isfinite(contrib), axis=1)
        row = np.nanmean(np.where(np.isfinite(contrib), contrib, np.nan), axis=1)
        row = row[n_valid >= min_sectors]
        return float(np.nanmean(row)) if len(row) else np.nan

    means = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(base_seed + seed)
        for _ in range(n_draws_per_seed):
            perm = rng.permutation(m)                 # shuffle which week's sign aligns
            means.append(_spread_for(sign_all[:-1][perm]))
    means = np.asarray([x for x in means if np.isfinite(x)])
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
    """Cost the weekly long-followers-of-up-leaders / short-followers-of-down-leaders
    book. The sign flips weekly, so we charge a full round-trip (2 sides × one-way cost
    × NAV) each week on the long-short book, plus borrow on the short leg."""
    sp = spreads["spread"].to_numpy(dtype=float)
    sp = sp[~np.isnan(sp)]
    n = len(sp)
    round_trip_cost = 2.0 * cost_bps / 1e4
    borrow_weekly = (borrow_bps_yr / 1e4) / TRADING_WEEKS
    net = sp - round_trip_cost - borrow_weekly
    gross_mean = float(sp.mean()) if n else float("nan")
    net_mean = float(net.mean()) if n else float("nan")
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
def synthetic_detect(
    panel: dict[str, pd.DataFrame],
    sectors: dict[str, list[str]],
    leaders: dict[str, str],
) -> dict:
    """Run the headline lead-lag stats on a synthetic panel."""
    wret = weekly_returns(panel)
    sp = leadlag_spreads(wret, sectors, leaders)
    ts = leadlag_stats(sp, wret, sectors, leaders)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_weeks": ts["n_weeks"]}
