"""Strategy + stats for Study 626 — Unemployment-Trend-Timing ("Growth-Trend Timing").

The three racers, all monthly, all sharing the same tape:

- **B&H**   : always long the index.
- **FABER** : risk-off whenever the month-end close is below its 200-day SMA
  (the pure trend rule of study 110-faber).
- **GTT**   : risk-off **only** when the trend is down AND unemployment is rising
  (latest available print above its 12-month SMA — the recession filter). When the
  economy is fine, price dips are ignored and the strategy stays long.

Exactly ONE execution lag: a signal formed at the close of month *m* sets the
position for month *m+1* (``shift(1)`` inside :func:`backtest`). The unemployment
leg additionally honours the 1-month *reporting* lag inside the data layer.
Risk-off earns the T-bill return; switches pay a one-way cost x NAV. Sharpe races
are excess-vs-excess (both legs net of the same cash series). Inference on active
returns uses a Newey-West (HAC) t — monthly active returns are serially correlated
whenever positions persist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_tstat(returns: pd.Series, lags: int | None = None) -> dict:
    """Newey-West (HAC, Bartlett kernel) t-statistic for the sample mean.

    ``lags=None`` uses the standard rule of thumb ``floor(4*(n/100)^(2/9))``.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * (float(e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean_bps": mu * 1e4, "se_bps": se * 1e4,
            "tstat": mu / se if se > 0 else np.nan, "lags": lags, "n": n}


# --------------------------------------------------------------------------- #
# Backtest engine
# --------------------------------------------------------------------------- #
def backtest(df: pd.DataFrame, risk_off: pd.Series, cost_bps: float = 5.0,
             ret_col: str = "eq_ret") -> pd.DataFrame:
    """Run a long/flat overlay from a month-end risk-off signal.

    ``risk_off[m]`` is the signal formed at the close of month *m*; the position
    held **during** month *m* is therefore ``~risk_off.shift(1)`` — the one
    execution lag. Risk-off months earn ``cash_ret``; every switch pays
    ``cost_bps`` one-way x NAV.
    """
    w = (~risk_off).astype(float).shift(1)          # 1 = long equity in month m
    w.iloc[0] = 1.0                                  # start invested
    turn = w.diff().abs().fillna(0.0)
    gross = w * df[ret_col] + (1.0 - w) * df["cash_ret"]
    net = gross - turn * cost_bps / 1e4
    return pd.DataFrame({"w": w, "gross": gross, "net": net, "turnover": turn})


def perf_stats(net: pd.Series, cash: pd.Series) -> dict:
    """CAGR, annualised vol, excess Sharpe, max drawdown, of a monthly net series."""
    n = len(net)
    level = (1.0 + net).cumprod()
    yrs = n / 12.0
    cagr = level.iloc[-1] ** (1.0 / yrs) - 1.0
    vol = net.std(ddof=1) * np.sqrt(12.0)
    ex = net - cash
    sharpe = ex.mean() / ex.std(ddof=1) * np.sqrt(12.0) if ex.std(ddof=1) > 0 else np.nan
    dd = (level / level.cummax() - 1.0).min()
    return {"cagr_pct": cagr * 100.0, "vol_pct": vol * 100.0,
            "sharpe": float(sharpe), "maxdd_pct": dd * 100.0, "n_months": n}


def spell_stats(risk_off: pd.Series, df: pd.DataFrame,
                ret_col: str = "eq_ret") -> dict:
    """Count risk-off spells and score each as whipsaw (being out LOST money) or save.

    A spell is a maximal run of risk-off months in the **held** position
    (signal shifted one month, as traded). Its timing P&L is the compounded
    cash-minus-equity return over the spell: negative = a whipsaw (the market
    kept rising, the exit cost money), positive = a save (the exit dodged losses).
    """
    held_off = risk_off.shift(1).fillna(False).astype(bool)
    ids = (held_off != held_off.shift()).cumsum()
    n_spells, whips, saves, lengths, pnls = 0, 0, 0, [], []
    for _, g in df[held_off].groupby(ids[held_off]):
        n_spells += 1
        lengths.append(len(g))
        pnl = (1.0 + g["cash_ret"]).prod() - (1.0 + g[ret_col]).prod()
        pnls.append(pnl)
        whips += int(pnl < 0)
        saves += int(pnl >= 0)
    switches = int(held_off.astype(int).diff().abs().sum())
    return {"n_spells": n_spells, "whipsaws": whips, "saves": saves,
            "switches": switches,
            "avg_len": float(np.mean(lengths)) if lengths else 0.0,
            "months_out": int(held_off.sum()),
            "pct_out": 100.0 * held_off.mean(),
            "spell_pnls": pnls}


# --------------------------------------------------------------------------- #
# The full three-way race
# --------------------------------------------------------------------------- #
def signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(faber_risk_off, gtt_risk_off) — month-end signal series."""
    faber = ~df["trend_up"].astype(bool)
    gtt = faber & df["unemp_rising"].astype(bool)
    return faber, gtt


def run_race(df: pd.DataFrame, cost_bps: float = 5.0,
             ret_col: str = "eq_ret") -> dict:
    """B&H vs Faber vs GTT: perf, spells, and HAC t on every active-return pair."""
    faber_sig, gtt_sig = signals(df)
    bh_net = df[ret_col].copy()
    fb = backtest(df, faber_sig, cost_bps, ret_col)
    gt = backtest(df, gtt_sig, cost_bps, ret_col)

    out = {
        "bh": perf_stats(bh_net, df["cash_ret"]),
        "faber": perf_stats(fb["net"], df["cash_ret"]),
        "gtt": perf_stats(gt["net"], df["cash_ret"]),
        "faber_spells": spell_stats(faber_sig, df, ret_col),
        "gtt_spells": spell_stats(gtt_sig, df, ret_col),
        "t_gtt_vs_bh": hac_tstat(gt["net"] - bh_net),
        "t_faber_vs_bh": hac_tstat(fb["net"] - bh_net),
        "t_gtt_vs_faber": hac_tstat(gt["net"] - fb["net"]),
        "series": {"bh": bh_net, "faber": fb["net"], "gtt": gt["net"],
                   "faber_w": fb["w"], "gtt_w": gt["w"]},
    }
    return out


def subperiod(df: pd.DataFrame, split: str = "2016-01-31",
              cost_bps: float = 5.0) -> dict:
    """The race before/after the GTT publication date (Livingston, Feb 2016)."""
    pre = run_race(df[df.index < pd.Timestamp(split)], cost_bps)
    post = run_race(df[df.index >= pd.Timestamp(split)], cost_bps)
    return {"pre": pre, "post": post}


# --------------------------------------------------------------------------- #
# The exposure-matched placebo — the honest null for "the filter has skill"
# --------------------------------------------------------------------------- #
def rotation_placebo(df: pd.DataFrame, cost_bps: float = 5.0,
                     ret_col: str = "eq_ret", min_offset: int = 12) -> dict:
    """Circularly rotate the unemployment filter; re-run GTT at every offset.

    The mechanical trap: GTT spends ~15pp more time in the market than Faber, so
    it picks up the unconditional equity premium in those extra months **whether
    or not the filter knows anything**. A rotated ``unemp_rising`` series keeps
    the filter's persistence and duty cycle (hence the same *average* extra
    exposure) but destroys its alignment with the market — the correct null for
    *timing skill beyond exposure*.

    Full deterministic enumeration of every circular shift (no RNG, better than
    any >= 20-seed average): ``p`` = share of rotations whose GTT-minus-Faber
    mean matches or beats the observed one.
    """
    faber_sig, gtt_sig = signals(df)
    fb_net = backtest(df, faber_sig, cost_bps, ret_col)["net"]
    obs = float((backtest(df, gtt_sig, cost_bps, ret_col)["net"] - fb_net).mean())
    u = df["unemp_rising"].to_numpy(dtype=bool)
    n = len(df)
    means = []
    for k in range(min_offset, n - min_offset):
        rot = pd.Series(np.roll(u, k), index=df.index)
        net = backtest(df, faber_sig & rot, cost_bps, ret_col)["net"]
        means.append(float((net - fb_net).mean()))
    means = np.asarray(means)
    return {"obs_bps": obs * 1e4, "placebo_mean_bps": float(means.mean()) * 1e4,
            "p_value": float((means >= obs).mean()), "n_rot": len(means),
            "draws_bps": means * 1e4}


def exposure_decomposition(df: pd.DataFrame, cost_bps: float = 5.0,
                           ret_col: str = "eq_ret") -> dict:
    """Split GTT-minus-Faber into a mechanical exposure part and a timing part.

    Mechanical part = (share of months GTT is long and Faber is flat) x the
    unconditional monthly equity excess return — what ANY filter of the same
    duty cycle collects on average. The remainder is what the *specific* months
    the filter picks are worth (its selection skill), gross of the cost saving.
    """
    faber_sig, gtt_sig = signals(df)
    fb = backtest(df, faber_sig, cost_bps, ret_col)
    gt = backtest(df, gtt_sig, cost_bps, ret_col)
    edge = float((gt["net"] - fb["net"]).mean())
    extra = (gt["w"] - fb["w"]).clip(lower=0.0)           # months GTT in, Faber out
    share_extra = float(extra.mean())
    uncond_ex = float((df[ret_col] - df["cash_ret"]).mean())
    mechanical = share_extra * uncond_ex
    return {"edge_bps": edge * 1e4, "share_extra_pct": share_extra * 100.0,
            "mechanical_bps": mechanical * 1e4,
            "timing_bps": (edge - mechanical) * 1e4}
