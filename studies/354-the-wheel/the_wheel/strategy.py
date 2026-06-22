"""Strategy + inference for Study 354 — The Wheel.

The retail pitch: sell a cash-secured **at-the-money put** each month; if the ETF closes
below the strike you are **assigned** (you now hold shares); then sell an at-the-money
**covered call** each month until the shares are **called away**; repeat "forever" for
income. We price the one-month ATM option with Black-Scholes using VIX/100 as the implied
vol — a transparent model, not a live chain.

The whole teardown rests on one identity. Holding cash-or-ETF with one short ATM option,
the Wheel's monthly P&L is **always**::

    wheel month return  =  (capped move on the side you are short)  +  premium kept

When short a put (you hold cash collateral): you keep the full **down** move below the strike
(you bought ETF that fell), and you keep **nothing** of the up move (the put expires, you stay
in cash) — plus the premium. When short a call (you hold the ETF): you keep the full **up**
move only to the strike, and you keep the full **down** move (you own the dropping ETF) — plus
the premium. Either leg is the textbook **short-volatility payoff**: truncated upside, full
downside, paid a premium for the trouble. So the Wheel's return ``=`` buy-and-hold ``minus``
the part of every up-move it gives away ``plus`` the premiums it banks. The question is only
whether the banked premium out-earns the surrendered upside — and against buy-and-hold on a
rising index, it does **not**: it is short vol in a costume.

This module: the Black-Scholes premium (:func:`bs_atm`), the Wheel engine (:func:`run_wheel`),
the upside/downside **capture** decomposition (:func:`capture`), and the Signal-axis inference
— a paired *t* of the Wheel's monthly excess return over buy-and-hold (:func:`paired_t`).
Pure numpy / pandas; stdlib ``erf`` for the normal CDF (no scipy required).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Black-Scholes ATM premium (the transparent option-pricing model)
# --------------------------------------------------------------------------- #
def _phi(x):
    """Standard-normal CDF (vectorised, stdlib erf — no scipy)."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def bs_atm(sigma_ann, t_years: float = 1.0 / 12.0, r: float = 0.0):
    """Black-Scholes price of an **at-the-money** (strike = spot) option, as a fraction of spot.

    For ``K = S`` and rate ``r`` the call and put prices per $1 of spot are::

        d1 = (r + sigma^2/2)*t / (sigma*sqrt(t)),   d2 = d1 - sigma*sqrt(t)
        call/S = Phi(d1) - exp(-r t) Phi(d2)
        put/S  = exp(-r t) Phi(-d2) - Phi(-d1)

    With ``r = 0`` (our default — a one-month horizon, near-zero carry) the ATM call and put
    are equal and reduce to ``2*Phi(sigma*sqrt(t)/2) - 1 ≈ 0.4 * sigma * sqrt(t)``. Vectorises
    over ``sigma_ann``. Returns ``(call_frac, put_frac)``.
    """
    sigma = np.asarray(sigma_ann, dtype=float)
    srt = sigma * math.sqrt(t_years)
    srt = np.where(srt <= 0, 1e-9, srt)
    d1 = (r * t_years + 0.5 * srt ** 2) / srt
    d2 = d1 - srt
    disc = math.exp(-r * t_years)
    call = _phi(d1) - disc * _phi(d2)
    put = disc * _phi(-d2) - _phi(-d1)
    return call, put


# --------------------------------------------------------------------------- #
# The Wheel engine
# --------------------------------------------------------------------------- #
def run_wheel(m: pd.DataFrame, r: float = 0.0, cost_frac: float = 0.0,
              cash_yield_ann: float = 0.0) -> pd.DataFrame:
    """Simulate the Wheel month by month over the month-end observation frame ``m``.

    ``m`` carries ``spy_open``, ``spy_close``, ``vix_open`` (implied vol), ``spy_ret``. State
    starts in **cash** selling ATM puts. Each month, given the open spot ``S`` (= strike, ATM),
    the realised SPY return ``g = spy_close/spy_open - 1`` and the priced premium ``c`` (call)
    / ``p`` (put):

    * **Short put (in cash).** Collateral is cash. If ``g >= 0`` the put expires worthless and
      we stay in cash: month return ``= p + cash_yield`` (premium + collateral yield). If
      ``g < 0`` we are assigned the ETF at the strike: month return ``= g + p`` (we ate the
      full drop, kept the premium) and **next month we hold the ETF** selling calls.
    * **Short call (holding ETF).** If ``g <= 0`` the call expires, we keep the ETF: month
      return ``= g + c`` (full drop, kept premium), stay in the call state. If ``g > 0`` the
      shares are called away at the strike: month return ``= min(g, 0) + c`` capped at the
      strike, i.e. ``= c`` for the price move (we only kept up to the strike) — formally
      ``min(g, 0) + c`` since any gain above the strike is surrendered; we then **return to
      cash** selling puts.

    ``cost_frac`` is a one-way transaction cost charged on each option written (a round-trip
    when assigned/called costs the underlying turnover too — charged as ``cost_frac`` on the
    notional). Returns a frame indexed like ``m`` with columns ``wheel_ret`` (the Wheel's
    monthly total return), ``state`` (``'put'`` or ``'call'`` entering the month), and the
    ``premium`` banked that month.
    """
    cash_m = cash_yield_ann / MONTHS_PER_YEAR
    call_frac, put_frac = bs_atm(m["vix_open"].to_numpy(), r=r)
    g = m["spy_ret"].to_numpy()
    out_ret = np.empty(len(m))
    out_state = np.empty(len(m), dtype=object)
    out_prem = np.empty(len(m))
    holding_etf = False  # start in cash, selling puts
    for i in range(len(m)):
        gi = g[i]
        if not holding_etf:
            prem = put_frac[i]
            out_state[i] = "put"
            if gi >= 0.0:                       # put expires; stay in cash
                ret = prem + cash_m
            else:                               # assigned; eat the drop, keep premium
                ret = gi + prem
                holding_etf = True
        else:
            prem = call_frac[i]
            out_state[i] = "call"
            if gi <= 0.0:                        # call expires; keep falling ETF + premium
                ret = gi + prem
            else:                               # called away at strike; upside capped
                ret = min(gi, 0.0) + prem       # = prem; gain above strike surrendered
                holding_etf = False
        out_ret[i] = ret - cost_frac            # one-way cost on the written option
        out_prem[i] = prem
    res = pd.DataFrame({"wheel_ret": out_ret, "state": out_state, "premium": out_prem},
                       index=m.index)
    return res


def equity_curve(ret: np.ndarray | pd.Series, start: float = 1.0) -> np.ndarray:
    """Compounded NAV path from a monthly-return series."""
    return start * np.cumprod(1.0 + np.asarray(ret, dtype=float))


# --------------------------------------------------------------------------- #
# Decomposition + summary stats
# --------------------------------------------------------------------------- #
def capture(m: pd.DataFrame, w: pd.DataFrame) -> dict:
    """Upside / downside **capture** of the Wheel vs buy-and-hold SPY.

    The signature of a short-vol / covered-call wrapper is keeping a *small* fraction of up
    months and a *large* fraction of down months. We measure, over up months (SPY return > 0)
    and down months (< 0) separately, the ratio of the Wheel's mean return to SPY's mean
    return. ``up_capture`` well below 1 with ``down_capture`` near (or above) 1 is the
    truncated-upside / full-downside payoff named.
    """
    g = m["spy_ret"].to_numpy()
    wr = w["wheel_ret"].to_numpy()
    up = g > 0
    dn = g < 0
    up_cap = float(wr[up].mean() / g[up].mean()) if up.any() and g[up].mean() != 0 else float("nan")
    dn_cap = float(wr[dn].mean() / g[dn].mean()) if dn.any() and g[dn].mean() != 0 else float("nan")
    return {
        "up_capture": up_cap, "dn_capture": dn_cap,
        "n_up": int(up.sum()), "n_dn": int(dn.sum()),
        "spy_up_mean": float(g[up].mean()) if up.any() else float("nan"),
        "wheel_up_mean": float(wr[up].mean()) if up.any() else float("nan"),
        "spy_dn_mean": float(g[dn].mean()) if dn.any() else float("nan"),
        "wheel_dn_mean": float(wr[dn].mean()) if dn.any() else float("nan"),
    }


def summary(ret: np.ndarray | pd.Series, rf_ann: float = 0.0) -> dict:
    """CAGR, annualised vol, Sharpe (excess of ``rf_ann``), max drawdown, skew of a monthly series."""
    r = np.asarray(ret, dtype=float)
    n = len(r)
    nav = np.cumprod(1.0 + r)
    cagr = float(nav[-1] ** (MONTHS_PER_YEAR / n) - 1.0)
    vol = float(r.std(ddof=1) * math.sqrt(MONTHS_PER_YEAR))
    ex = r - rf_ann / MONTHS_PER_YEAR
    sharpe = float(ex.mean() / r.std(ddof=1) * math.sqrt(MONTHS_PER_YEAR)) if r.std(ddof=1) > 0 else float("nan")
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1.0).min())
    mu = r.mean(); sd = r.std(ddof=1)
    skew = float(np.mean(((r - mu) / sd) ** 3)) if sd > 0 else float("nan")
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "mdd": mdd, "skew": skew,
            "final": float(nav[-1]), "n": n}


# --------------------------------------------------------------------------- #
# Inference — the Signal axis
# --------------------------------------------------------------------------- #
def paired_t(wheel_ret: np.ndarray | pd.Series, spy_ret: np.ndarray | pd.Series) -> dict:
    """Paired *t* of the Wheel's monthly **excess** return over buy-and-hold SPY.

    The Signal axis asks: does the Wheel deliver a real return *advantage* over simply holding
    the index? We test the per-month difference ``d = wheel_ret - spy_ret`` against zero with a
    paired *t* (``t = mean(d) / (sd(d)/sqrt(n))``). A robust ``t >= 2`` for ``mean(d) > 0`` would
    earn Signal `REAL`; a significantly *negative* mean (the Wheel trailing the index) is the
    honest finding the prior expects — premium income that does not out-earn surrendered upside.
    Also reports the annualised mean drag and the monthly win-rate of the difference.
    """
    d = np.asarray(wheel_ret, dtype=float) - np.asarray(spy_ret, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("nan")
    return {
        "n": n, "mean_monthly": mean, "ann_drag": mean * MONTHS_PER_YEAR,
        "t": float(t), "win_rate": float((d > 0).mean()),
    }


def fair_premium_identity(sigma_ann, t_years: float = 1.0 / 12.0) -> dict:
    """Sanity identity for the synthetic control: at ``r = 0`` the ATM call equals the ATM put,
    and both equal ``2*Phi(sigma*sqrt(t)/2) - 1``. Returns the engine value and the closed form
    so the positive control can assert they match — the premium model is exact, not fitted."""
    call, put = bs_atm(sigma_ann, t_years=t_years, r=0.0)
    closed = 2.0 * _phi(np.asarray(sigma_ann) * math.sqrt(t_years) / 2.0) - 1.0
    return {"call": call, "put": put, "closed_form": closed}
