"""Strategy + inference for Study 618 — GBTC Premium Cycle.

The claim: one wrapper, three regimes, all mechanical — GBTC traded ~40 % ABOVE its bitcoin
while it was the only listed way in (creations open to accredited investors only, no
redemptions), ~45 % BELOW once futures/spot competition killed the premium and the
no-redemption clause trapped the BTC inside, and snapped to ~0 the day it became an ETF
(in-kind create/redeem arb switched on). We measure each leg on the reconstructed
premium series and put the inference where the trade was:

  * **Regime levels.** Mean premium per documented regime with a Newey-West (HAC) t — the
    premium level is strongly autocorrelated, so plain t's would be fantasy; we use a long
    HAC lag (63 trading days) on the level.
  * **The convergence drift (the dated trade of 2023).** Long GBTC / short BTC captures
    d log(1+prem) exactly (the BPS decay is the fee you pay). HAC t on the daily hedged
    return over (a) calendar-2023 → conversion and (b) the *ex-ante triggered* version:
    enter at the close ONE trading day after BlackRock's spot-ETF filing (2023-06-15) — the
    single documented execution lag — exit at the conversion-day close.
  * **Event days.** One-day hedged moves on the documented catalysts, z-scored against the
    discount-era daily hedged sd.
  * **The premium-era arb (third axis).** Accredited create-and-dump: create at NAV, exit at
    the market premium 6 months (126 trading days) later under the Rule-144 lockup, hedged
    short BTC. Cohort-level outcomes show who could harvest it — and how it blew up.

Costs: one-way bps × each leg (long GBTC + short BTC), shorts pay borrow (an explicit
%/yr carry on the BTC leg). Sharpe races are not the object here — the object is a dated,
bounded convergence with a terminal event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as _data

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def nw_tstat(x: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """Newey-West (HAC, Bartlett kernel) t of mean(x) vs 0. Returns (mean, t)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 8:
        return float("nan"), float("nan")
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    s = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(e[:-k] @ e[k:]) / n
    se = np.sqrt(max(s, 1e-30) / n)
    return float(mu), float(mu / se)


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variance)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Regime anatomy
# --------------------------------------------------------------------------- #
def regime_table(df: pd.DataFrame, lags: int = 63) -> list[dict]:
    """Mean/extremes of the premium per documented regime, HAC t on the level.

    The premium level is near-unit-root inside a regime, so the HAC lag is long (63 days);
    the level t's are supporting evidence — the decisive inference lives on the drift.
    """
    out = []
    for name, sub in _data.regimes(df).items():
        p = sub["prem"].to_numpy()
        mu, t = nw_tstat(p, lags=lags)
        out.append({
            "regime": name, "n": len(p),
            "start": str(sub.index.min().date()), "end": str(sub.index.max().date()),
            "mean_pct": mu * 100, "t_hac": t,
            "min_pct": p.min() * 100, "min_date": str(sub["prem"].idxmin().date()),
            "max_pct": p.max() * 100, "max_date": str(sub["prem"].idxmax().date()),
        })
    return out


def first_sustained_discount(df: pd.DataFrame, run: int = 5) -> str:
    """First date opening a run of ``run`` consecutive closes below NAV (the regime flip)."""
    neg = (df["prem"] < 0).to_numpy()
    for i in range(len(neg) - run + 1):
        if neg[i:i + run].all():
            return str(df.index[i].date())
    return "never"


def hedged_returns(df: pd.DataFrame) -> pd.Series:
    """Daily return of long-GBTC / short-BTC-per-share: exactly d log(1+prem).

    log(GBTC) - log(NAV) = log(1+prem), so the beta-hedged book earns the premium change;
    the BPS fee decay inside NAV is the carry the long leg pays (~2 %/yr pre-conversion).
    """
    return np.log1p(df["prem"]).diff().dropna()


# --------------------------------------------------------------------------- #
# The convergence trade (discount -> 0 into the conversion)
# --------------------------------------------------------------------------- #
def convergence_test(df: pd.DataFrame, start: str, end: str,
                     lags: int = 10) -> dict:
    """HAC t on the daily hedged return over [start close, end close].

    Entry at the ``start`` close, exit at the ``end`` close — the window's first hedged
    return accrues from the entry close, so nothing before the entry is used.
    """
    h = hedged_returns(df)
    win = h[(h.index > pd.Timestamp(start)) & (h.index <= pd.Timestamp(end))]
    mu, t = nw_tstat(win.to_numpy(), lags=lags)
    p0 = float(df["prem"].loc[:start].iloc[-1])
    p1 = float(df["prem"].loc[:end].iloc[-1])
    return {
        "start": start, "end": end, "n_days": int(len(win)),
        "prem_entry_pct": p0 * 100, "prem_exit_pct": p1 * 100,
        "total_log_pct": float(win.sum()) * 100,
        "mean_daily_bps": mu * 1e4, "t_hac": t, "lags": lags,
    }


def convergence_costs(df: pd.DataFrame, start: str, end: str,
                      cost_bps: float = 10.0, borrow_pa: float = 5.0) -> dict:
    """Net hedged P&L: one-way ``cost_bps`` × 4 legs (enter+exit × long GBTC + short BTC),
    and the short-BTC leg pays ``borrow_pa`` %/yr carry (futures roll / borrow). One-way ×
    NAV per leg; the long GBTC leg's sponsor fee is already inside NAV (the BPS decay)."""
    ct = convergence_test(df, start, end)
    yrs = ct["n_days"] / TRADING_DAYS
    legs = 4.0 * cost_bps / 1e4
    borrow = borrow_pa / 100.0 * yrs
    gross = ct["total_log_pct"] / 100.0
    net = gross - legs - borrow
    ct.update({
        "cost_bps": cost_bps, "borrow_pa_pct": borrow_pa, "years": yrs,
        "gross_log_pct": gross * 100, "legs_cost_pct": legs * 100,
        "borrow_cost_pct": borrow * 100, "net_log_pct": net * 100,
        "net_simple_pct": (np.expm1(net)) * 100,
    })
    return ct


def event_days(df: pd.DataFrame, sd_window: tuple[str, str]) -> list[dict]:
    """One-day hedged moves on the documented catalysts, z vs the discount-era daily sd."""
    h = hedged_returns(df)
    sd = float(h[(h.index >= sd_window[0]) & (h.index <= sd_window[1])].std(ddof=1))
    out = []
    for d, label in _data.EVENTS.items():
        ts = pd.Timestamp(d)
        if ts not in h.index:          # event on a non-trading day -> next close
            nxt = h.index[h.index >= ts]
            if len(nxt) == 0:
                continue
            ts = nxt[0]
        move = float(h.loc[ts])
        out.append({"date": str(ts.date()), "label": label,
                    "move_pct": move * 100, "z": move / sd})
    return out


# --------------------------------------------------------------------------- #
# The premium-era arb (create at NAV, dump at the premium after the lockup)
# --------------------------------------------------------------------------- #
def lockup_arb_cohorts(df: pd.DataFrame, first: str = "2015-06-01",
                       last: str = "2021-06-30", lock_days: int = 126,
                       fee_pa: float = _data.FEE_TRUST,
                       cost_bps: float = 25.0) -> pd.DataFrame:
    """Accredited create-and-dump: one cohort per month-start, hedged short BTC.

    Create at NAV (premium = 0 by construction), sell at the market premium ``lock_days``
    trading days later (the 6-month Rule-144 lockup on private-placement shares; 12 months
    before Oct-2019 amendments — 126 days is the *favourable* case). Hedged log P&L per
    cohort = log(1 + prem_exit) − fee drag over the lockup − costs (one-way ``cost_bps`` ×
    2 exit legs + entry hedge leg = 3 legs; creation itself is at NAV, no spread) − borrow
    (5 %/yr on the short-BTC leg over the lockup). Cohorts whose exit falls after the tape
    ends are dropped.
    """
    prem = df["prem"]
    starts = pd.date_range(first, last, freq="MS")
    rows = []
    for d in starts:
        pos = prem.index.searchsorted(d)
        if pos + lock_days >= len(prem):
            continue
        t_in = prem.index[pos]
        t_out = prem.index[pos + lock_days]
        yrs = lock_days / TRADING_DAYS
        gross = float(np.log1p(prem.iloc[pos + lock_days]))
        drag = fee_pa * yrs
        legs = 3.0 * cost_bps / 1e4
        borrow = 0.05 * yrs
        rows.append({"created": t_in, "exit": t_out,
                     "prem_exit_pct": float(prem.iloc[pos + lock_days]) * 100,
                     "gross_log_pct": gross * 100,
                     "net_log_pct": (gross - drag - legs - borrow) * 100})
    return pd.DataFrame(rows).set_index("created")


def lockup_arb_summary(cohorts: pd.DataFrame, blowup_from: str = "2020-09-01") -> dict:
    """Early cohorts vs the blow-up cohorts (created from ``blowup_from``, exiting into the
    2021 discount flip) — Welch t on the net cohort P&L between the two groups."""
    early = cohorts[cohorts.index < pd.Timestamp(blowup_from)]["net_log_pct"].to_numpy()
    late = cohorts[cohorts.index >= pd.Timestamp(blowup_from)]["net_log_pct"].to_numpy()
    return {
        "n_early": len(early), "early_mean_pct": float(early.mean()),
        "early_worst_pct": float(early.min()), "early_share_pos": float((early > 0).mean()),
        "n_late": len(late), "late_mean_pct": float(late.mean()) if len(late) else float("nan"),
        "late_worst_pct": float(late.min()) if len(late) else float("nan"),
        "welch_t": welch_t(early, late),
    }
