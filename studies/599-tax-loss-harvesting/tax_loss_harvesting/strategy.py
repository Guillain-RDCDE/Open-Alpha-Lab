"""Study 599 — Tax-Loss Harvesting: lot-level TLH engine + the desk statistics.

The claim under test: *systematically harvesting losses into a twin fund adds 0.3–1%/yr of
after-tax alpha*. The engine simulates a daily-check TLH programme on one total-return price
series with **lot-level accounting**:

* every purchase (initial lump sum, monthly contribution, harvest repurchase, reinvested tax
  saving) opens a lot with its own basis and date;
* each day the engine flags lots trading below basis by more than ``threshold``; flagged lots
  are sold at the **next** close (exactly ONE execution lag) if still under water, the realised
  loss is classified short-term (< 366 calendar days) or long-term, and the tax saving
  (loss × the applicable rate) is **reinvested immediately** alongside the sale proceeds into
  the twin fund — SPY↔IVV treated as economically identical exposure (justified by
  ``data.twin_check``), so the market return stream is unchanged and the *entire* delta is tax
  timing + trading costs;
* wash-sale safety: a lot must be held ≥ 31 calendar days before it can be harvested, and the
  two-twin routing (sell fund A, buy fund B; contributions go into the currently-held twin)
  means no substantially-identical repurchase ever lands within the 30-day window;
* costs are **one-way bps × traded NAV** on every leg (harvest sell + rebuy, terminal sale);
* at the horizon both books are liquidated and pay capital-gains tax per lot (LT rate if held
  ≥ 366 days, ST otherwise; terminal losses credit at the same rates) — unless ``step_up=True``
  (death/donation basis step-up: no terminal tax on either book).

After-tax alpha (lump-sum case) = annualised ratio of TLH vs buy-and-hold **after-tax**
terminal wealth. Assumptions stated in results.md: harvested losses are fully usable against
gains taxed at ``st_rate``/``lt_rate`` (no $3,000 cap binding), and tax savings are reinvested
the day they are realised.

Stats: HAC/Newey-West t across overlapping cohorts (lag = the overlap), Welch t for the
bear-vs-calm harvest-yield split, and a ≥ 20-seed synthetic control.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

LT_DAYS = 366           # calendar days for long-term treatment
MIN_HOLD = 31           # wash-sale-safe minimum holding before a lot may be harvested


# --------------------------------------------------------------------------- #
# The lot-level engine
# --------------------------------------------------------------------------- #
def simulate(prices: pd.Series, initial: float = 100_000.0, monthly_contrib: float = 0.0,
             st_rate: float = 0.35, lt_rate: float = 0.15, cost_bps: float = 2.0,
             threshold: float = 0.01, harvest: bool = True,
             step_up: bool = False) -> dict:
    """Run one TLH (or buy-and-hold) programme over ``prices`` (total-return closes).

    Returns after-tax terminal wealth plus harvest diagnostics. ``harvest=False`` is the
    buy-and-hold benchmark: same cash flows, no harvesting, same terminal tax treatment.
    """
    px = prices.to_numpy(dtype=float)
    dates = prices.index
    day_num = (dates - dates[0]).days.to_numpy()             # calendar-day offsets
    n = len(px)
    c = cost_bps / 1e4

    # each lot: [shares, basis_per_share, buy_day_index]
    lots: list[list[float]] = [[initial * (1 - c) / px[0], px[0], 0]]
    invested = initial
    pending: list[int] = []                                  # lot ids flagged yesterday
    losses_st = losses_lt = 0.0
    tax_saved = 0.0
    n_harvests = 0
    traded = 0.0                                             # harvest NAV traded (sell leg)
    harvested_by_year: dict[int, float] = {}                 # realised loss per calendar year
    nav_by_year: dict[int, list] = {}                        # daily NAVs per calendar year
    harvested_by_age: dict[int, float] = {}                  # realised loss per year-since-start

    for i in range(n):
        # monthly contribution at the first trading day of each month (skips day 0)
        if monthly_contrib > 0.0 and i > 0 and dates[i].month != dates[i - 1].month:
            lots.append([monthly_contrib * (1 - c) / px[i], px[i], i])
            invested += monthly_contrib

        # execute yesterday's harvest decision at TODAY's close (one execution lag)
        if harvest and pending:
            cash = 0.0
            executed = []
            for k in pending:
                sh, basis, ib = lots[k]
                if px[i] < basis:                            # still a loss at execution
                    value = sh * px[i]
                    loss = (basis - px[i]) * sh
                    held = day_num[i] - day_num[ib]
                    rate = st_rate if held < LT_DAYS else lt_rate
                    if held < LT_DAYS:
                        losses_st += loss
                    else:
                        losses_lt += loss
                    saving = loss * rate
                    tax_saved += saving
                    cash += value * (1 - c) + saving
                    traded += value
                    yr = int(dates[i].year)
                    harvested_by_year[yr] = harvested_by_year.get(yr, 0.0) + loss
                    age = int(day_num[i] // 365.25)
                    harvested_by_age[age] = harvested_by_age.get(age, 0.0) + loss
                    executed.append(k)
                    n_harvests += 1
            if executed:
                for k in sorted(executed, reverse=True):
                    del lots[k]
                # proceeds + reinvested tax saving buy ONE new twin-fund lot at today's close
                lots.append([cash * (1 - c) / px[i], px[i], i])
            pending = []

        # today's flags, executed tomorrow
        if harvest and i + 1 < n:
            pending = [k for k, (sh, basis, ib) in enumerate(lots)
                       if px[i] < basis * (1 - threshold)
                       and (day_num[i] - day_num[ib]) >= MIN_HOLD]

        nav = sum(sh for sh, _, _ in lots) * px[i]
        nav_by_year.setdefault(int(dates[i].year), []).append(nav)

    # terminal liquidation at the last close: per-lot capital-gains tax (unless step-up)
    gross = sum(sh for sh, _, _ in lots) * px[-1]
    net_of_cost = gross * (1 - c)
    tax_term = 0.0
    if not step_up:
        for sh, basis, ib in lots:
            gain = (px[-1] - basis) * sh
            held = day_num[-1] - day_num[ib]
            rate = lt_rate if held >= LT_DAYS else st_rate
            tax_term += gain * rate                          # losses credit symmetrically
    w_after = net_of_cost - tax_term

    years = day_num[-1] / 365.25
    avg_nav_year = {y: float(np.mean(v)) for y, v in nav_by_year.items()}
    return {
        "w_after_tax": w_after, "w_pre_tax": net_of_cost, "years": years,
        "invested": invested, "n_harvests": n_harvests,
        "losses_st": losses_st, "losses_lt": losses_lt, "tax_saved": tax_saved,
        "traded_nav": traded, "terminal_tax": tax_term,
        "harvested_by_year": harvested_by_year, "avg_nav_by_year": avg_nav_year,
        "harvested_by_age": harvested_by_age,
    }


def after_tax_alpha(prices: pd.Series, **kw) -> dict:
    """TLH vs buy-and-hold, identical cash flows and terminal tax rules.

    Lump-sum case (no contributions): alpha = annualised after-tax wealth ratio, in bps/yr.
    With contributions the ratio-annualisation is money-weighted-approximate, so we also
    report the terminal after-tax wealth uplift in % (labelled as such wherever quoted).
    """
    h = simulate(prices, harvest=True, **kw)
    b = simulate(prices, harvest=False, **kw)
    years = h["years"]
    ratio = h["w_after_tax"] / b["w_after_tax"]
    alpha_bps = (ratio ** (1.0 / years) - 1.0) * 1e4
    uplift_pct = (ratio - 1.0) * 100.0
    return {"alpha_bps": alpha_bps, "uplift_pct": uplift_pct, "years": years,
            "h": h, "b": b}


# --------------------------------------------------------------------------- #
# Cohorts (path dependence + the HAC t the Signal axis needs)
# --------------------------------------------------------------------------- #
def cohort_alphas(prices: pd.Series, horizon_years: int = 10, step_months: int = 3,
                  **kw) -> pd.DataFrame:
    """Rolling lump-sum cohorts: invest at each quarter-start, run TLH for ``horizon_years``.

    Returns one row per cohort with the after-tax alpha (bps/yr) and the year-1 harvest yield.
    Overlapping windows — inference must use the HAC t with lag = the overlap.
    """
    starts = prices.resample("QS").first().index
    rows = []
    for t0 in starts:
        t1 = t0 + pd.DateOffset(years=horizon_years)
        if t1 > prices.index[-1]:
            break
        window = prices[(prices.index >= t0) & (prices.index <= t1)]
        if len(window) < horizon_years * 240:
            continue
        r = after_tax_alpha(window, **kw)
        hb = r["h"]
        y1_loss = hb["harvested_by_age"].get(0, 0.0) + hb["harvested_by_age"].get(1, 0.0)
        rows.append({"start": window.index[0], "alpha_bps": r["alpha_bps"],
                     "n_harvests": hb["n_harvests"],
                     "harvested_total": hb["losses_st"] + hb["losses_lt"],
                     "y01_loss": y1_loss})
    return pd.DataFrame(rows).set_index("start")


def nw_tstat(x: np.ndarray, lag: int) -> float:
    """Newey-West (HAC) t of mean(x) != 0 with ``lag`` Bartlett lags — for overlapping cohorts."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    e = x - x.mean()
    g0 = float(e @ e) / n
    s = g0
    for j in range(1, min(lag, n - 1) + 1):
        gj = float(e[j:] @ e[:-j]) / n
        s += 2.0 * (1.0 - j / (lag + 1.0)) * gj
    se = math.sqrt(max(s, 1e-300) / n)
    return float(x.mean() / se)


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch t (and dof) for mean(a) != mean(b) with unequal variances."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(t), float(dof)


# --------------------------------------------------------------------------- #
# Harvest-yield diagnostics
# --------------------------------------------------------------------------- #
BEAR_YEARS = (2000, 2001, 2002, 2008, 2009, 2020, 2022)


def harvest_yield_by_year(sim: dict) -> pd.Series:
    """Realised harvested loss as % of that year's average NAV, per calendar year."""
    out = {}
    for y, nav in sim["avg_nav_by_year"].items():
        out[y] = 100.0 * sim["harvested_by_year"].get(y, 0.0) / nav
    return pd.Series(out).sort_index()


def decay_profile(prices: pd.Series, horizon_years: int = 10, step_months: int = 3,
                  **kw) -> pd.Series:
    """Average harvested loss (% of initial stake) by YEAR-SINCE-INCEPTION across lump-sum
    cohorts — the basis-lock-up decay curve."""
    starts = prices.resample("QS").first().index
    acc = {k: [] for k in range(horizon_years)}
    for t0 in starts:
        t1 = t0 + pd.DateOffset(years=horizon_years)
        if t1 > prices.index[-1]:
            break
        window = prices[(prices.index >= t0) & (prices.index <= t1)]
        if len(window) < horizon_years * 240:
            continue
        sim = simulate(window, harvest=True, **kw)
        for k in range(horizon_years):
            acc[k].append(100.0 * sim["harvested_by_age"].get(k, 0.0) / kw.get("initial", 100_000.0))
    return pd.Series({k: float(np.mean(v)) for k, v in acc.items() if v}).sort_index()


# --------------------------------------------------------------------------- #
# Synthetic control (≥ 20 seeds averaged — never a single-seed baseline)
# --------------------------------------------------------------------------- #
def synthetic_control(sigmas=(0.02, 0.18, 0.35), n_seeds: int = 20,
                      st_rate: float = 0.35, lt_rate: float = 0.15,
                      cost_bps: float = 2.0) -> pd.DataFrame:
    """Machinery proof: TLH alpha must RISE with the planted volatility knob and must be
    ~zero at rates 0/0 (the null). Each cell averages ``n_seeds`` independent GBM paths."""
    from . import data

    rows = []
    for sigma in sigmas:
        a_tax, a_null = [], []
        for s in range(n_seeds):
            path = data.synthetic_path(n_years=10, sigma=sigma, seed=599 + s)
            a_tax.append(after_tax_alpha(path, st_rate=st_rate, lt_rate=lt_rate,
                                         cost_bps=cost_bps)["alpha_bps"])
            a_null.append(after_tax_alpha(path, st_rate=0.0, lt_rate=0.0,
                                          cost_bps=cost_bps)["alpha_bps"])
        rows.append({"sigma": sigma,
                     "alpha_bps_35_15": float(np.mean(a_tax)),
                     "alpha_sd": float(np.std(a_tax, ddof=1)),
                     "alpha_bps_rates0": float(np.mean(a_null))})
    return pd.DataFrame(rows).set_index("sigma")
