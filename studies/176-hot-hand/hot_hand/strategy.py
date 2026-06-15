"""The strategy and its honest controls — Study 176 (Hot-Hand).

The folk recipe splits in two, depending on who you ask at the water cooler:

- **Hot-hand believers**: after N consecutive up (or down) days, *keep going* — the
  market is "on a roll". This is **momentum** in streak form.
- **Gambler's fallacy believers**: after N consecutive up days, *fade it* — the market
  is "due for a turn". This is **mean-reversion** in streak form.

Both hypotheses are testable on the same tape, with the same fair bet:

  ``after a run of exactly N up-days (or down-days), is day N+1 more likely up or down?``

The honest control is the **unconditional next-day return** — what a naive hold earns
with no streak filter. If streaks carry information, conditioned next-day returns
should differ materially (in mean and in win-rate) from the unconditional baseline.

Two analyses share one engine (:func:`streak_table` and :func:`trade_streak`):

- **Binomial / continuation rate**: for each streak length 1..MAX_STREAK, what fraction
  of follow-on days continue? 50 % is the null; we test via exact binomial.
- **Mean next-day return with HAC t-stat**: does the conditional mean exceed the
  unconditional mean? And net of transaction costs (1 round-trip per streak-entry)?

No look-ahead: the streak is identified on close-to-close returns through day *t*;
the position is conceptually entered at *t*'s close / *t+1*'s open and exited at
*t+1*'s close (a one-day hold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

MAX_STREAK = 6  # beyond 6, n < ~20 on a 30-year daily tape — too thin to interpret


# ---------------------------------------------------------------------------
# Streak labelling
# ---------------------------------------------------------------------------

def label_streaks(close: pd.Series) -> pd.DataFrame:
    """Attach streak metadata to each trading day.

    For each day *t* (once we have at least 2 closes), compute:

    - ``ret``         — close-to-close log return on day *t*.
    - ``direction``   — +1 if ``ret > 0``, -1 if ``ret < 0``, 0 if flat.
    - ``streak_len``  — number of consecutive same-direction days ending on day *t*
                        (including day *t* itself). 0 on a flat day or day 1.
    - ``next_ret``    — log return on day *t+1* (the bet).
    - ``next_up``     — 1 if next_ret > 0, 0 otherwise (for the continuation rate).

    The last day has ``next_ret = NaN`` (no forward return yet).
    """
    log_ret = np.log(close / close.shift(1))
    # np.sign on NaN yields NaN; fill with 0 then convert to int
    direction = np.sign(log_ret).fillna(0).astype(int)

    streak_len = pd.Series(0, index=close.index, dtype=int)
    for i in range(1, len(close)):
        if direction.iat[i] == 0:
            streak_len.iat[i] = 0
        elif direction.iat[i] == direction.iat[i - 1]:
            streak_len.iat[i] = streak_len.iat[i - 1] + 1
        else:
            streak_len.iat[i] = 1

    next_ret = log_ret.shift(-1)
    next_up = (next_ret > 0).astype(float)
    next_up[next_ret.isna()] = float("nan")

    df = pd.DataFrame(
        {
            "ret": log_ret,
            "direction": direction,
            "streak_len": streak_len,
            "next_ret": next_ret,
            "next_up": next_up,
        },
        index=close.index,
    )
    return df


# ---------------------------------------------------------------------------
# Main analysis: streak table
# ---------------------------------------------------------------------------

def streak_table(
    close: pd.Series,
    max_streak: int = MAX_STREAK,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """For each streak length 1..max_streak and direction (up/down), summarise:

    - ``n``             — number of qualifying days.
    - ``continuation``  — fraction of next days that continue the streak (hot-hand rate).
    - ``binom_p``       — two-sided binomial p-value for continuation != 0.5.
    - ``mean_next_bps`` — mean next-day log return in bps.
    - ``uncond_bps``    — unconditional mean next-day return in bps (the baseline).
    - ``excess_bps``    — conditional minus unconditional (the edge, if any).
    - ``tstat``         — HAC t-stat for (mean next-day return − unconditional mean).
    - ``net_bps``       — excess_bps − cost_bps (tradability after one round-trip).

    The unconditional mean is computed across all days with a valid forward return,
    so it is the fairest benchmark: it is what you earn by always being in the market.
    """
    df = label_streaks(close)
    df_valid = df.dropna(subset=["next_ret"])
    uncond = float(df_valid["next_ret"].mean() * 1e4)

    rows = []
    for direction in (+1, -1):
        dir_label = "up" if direction == 1 else "down"
        for n in range(1, max_streak + 1):
            mask = (df["streak_len"] == n) & (df["direction"] == direction)
            sub = df.loc[mask].dropna(subset=["next_ret"])
            count = len(sub)
            if count == 0:
                continue
            continuation = float((sub["direction"].shift(-1).reindex(sub.index) == direction).mean())
            # Use next_up as the continuation proxy (same direction next day)
            # next_up is up, we need to match direction:
            if direction == 1:
                cont_rate = float((sub["next_ret"] > 0).mean())
            else:
                cont_rate = float((sub["next_ret"] < 0).mean())

            binom_res = scipy_stats.binomtest(
                int(cont_rate * count + 0.5), n=count, p=0.5, alternative="two-sided"
            )

            mean_bps = float(sub["next_ret"].mean() * 1e4)
            excess = mean_bps - uncond

            # HAC t-stat on the excess (next_ret - unconditional mean)
            excess_series = sub["next_ret"] - df_valid["next_ret"].mean()
            r = excess_series.to_numpy(dtype=float)
            r = r[np.isfinite(r)]
            tstat = float("nan")
            if len(r) > 5:
                mu = r.mean()
                e = r - mu
                lags = max(1, int(np.floor(4.0 * (len(r) / 100.0) ** (2.0 / 9.0))))
                lrv = float(e @ e) / len(r)
                for k in range(1, lags + 1):
                    w = 1.0 - k / (lags + 1.0)
                    lrv += 2.0 * w * float(e[k:] @ e[:-k]) / len(r)
                se = np.sqrt(max(lrv, 0.0) / len(r))
                tstat = float(mu / se) if se > 0 else float("nan")

            rows.append(
                {
                    "direction": dir_label,
                    "streak_len": n,
                    "n": count,
                    "continuation": cont_rate,
                    "binom_p": float(binom_res.pvalue),
                    "mean_next_bps": mean_bps,
                    "uncond_bps": uncond,
                    "excess_bps": excess,
                    "tstat": tstat,
                    "net_bps": excess - cost_bps,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Streak-conditioned trading rule
# ---------------------------------------------------------------------------

def trade_streak(
    close: pd.Series,
    min_streak: int = 3,
    direction: int = 1,
    follow_streak: bool = True,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """One-day trades triggered after a streak of length >= min_streak.

    Enters at the close of day *t* (conceptually), exits at the close of day *t+1*.

    Parameters
    ----------
    min_streak:   minimum run length before entering.
    direction:    +1 = react to up-streaks, -1 = react to down-streaks, 0 = both.
    follow_streak: True = ride the streak (hot-hand), False = fade it (gambler).
    cost_bps:     one-way cost × 2 per round-trip in bps.

    Returns a per-trade ledger with columns: ``date``, ``streak_len``, ``dir``,
    ``ret_gross``, ``ret_net``.
    """
    df = label_streaks(close)
    rows = []
    for idx, row in df.iterrows():
        sl = int(row["streak_len"])
        d = int(row["direction"])
        nr = row["next_ret"]
        if sl < min_streak or np.isnan(nr):
            continue
        if direction != 0 and d != direction:
            continue
        trade_dir = d if follow_streak else -d
        ret_gross = float(trade_dir * nr)
        rows.append(
            {
                "date": idx,
                "streak_len": sl,
                "dir": int(trade_dir),
                "ret_gross": ret_gross,
                "ret_net": ret_gross - cost_bps * 1e-4,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-trade statistics for one ledger.

    Returns trade count, win-rate, mean return (bps/trade), P&L skew, and a
    HAC t-stat on the mean — the inference-bar number that decides whether
    the edge is distinguishable from zero.
    """
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "n_trades": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "skew": float(pd.Series(r).skew()) if n > 2 else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        mu = r.mean()
        e = r - mu
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Bonferroni-corrected significance check across all streak cells
# ---------------------------------------------------------------------------

def bonferroni_summary(tbl: pd.DataFrame, alpha: float = 0.05) -> dict:
    """How many streak cells survive a Bonferroni correction?

    With MAX_STREAK * 2 = 12 sub-hypotheses (6 lengths × 2 directions), the
    Bonferroni threshold is alpha / 12 ≈ 0.004 at the 5% family-wise level.
    The fun: before correction, one or two cells might look exciting; after
    correction, none do.
    """
    n_tests = len(tbl)
    alpha_adj = alpha / max(n_tests, 1)
    sig_raw = (tbl["binom_p"] < alpha).sum()
    sig_bonf = (tbl["binom_p"] < alpha_adj).sum()
    return {
        "n_tests": n_tests,
        "alpha_raw": alpha,
        "alpha_bonf": alpha_adj,
        "sig_raw": int(sig_raw),
        "sig_bonf": int(sig_bonf),
    }
