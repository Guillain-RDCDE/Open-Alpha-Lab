"""The Dogs-of-the-Dow engine and its honest controls — Study 88.

The folk claim (O'Higgins & Downes, *Beating the Dow*, 1991): *each January, buy the ten
highest-dividend-yield stocks in the Dow 30, equal-weight, hold a year, repeat — and you'll
beat the Dow itself.* We implement that literally and pin it against the **same-basis** Dow
benchmark (total return vs total return — no price-only-vs-total-return race), net of
rebalancing costs, and ask the two questions that separate a real edge from a value tilt:

- **Is the annual excess return statistically real?** — HAC *t* and a block bootstrap on
  the annual Dogs-minus-Dow excess return (an annual series is short, so the inference bar
  is honest about its own power).
- **Is the 'edge' just beta you were always paid for?** — an OLS of the Dogs' excess return
  on the Dow's excess return: a slope (beta) above 1 with an alpha indistinguishable from
  zero says "you took more market/value risk", not "you found alpha".

Selection convention (no look-ahead): a January rebalance ranks the **then-current** Dow
members on trailing-12-month dividend yield measured at the **prior December close**, buys
the top ten equal-weight, and holds for the next calendar year. The point-in-time universe
lives in :mod:`dogs_of_the_dow.data`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as _data


# ---------------------------------------------------------------------------
# Trailing dividend yield at a rebalance date
# ---------------------------------------------------------------------------
def trailing_yield(
    price: pd.Series, divs: pd.Series, asof: pd.Timestamp
) -> float:
    """Trailing-12-month dividends / price at ``asof`` (the December rebalance close).

    Sums per-share dividends with ex-dates in the 12 months ending at ``asof`` and divides
    by the close on/just before ``asof``. Returns NaN if there is no price near ``asof``.
    """
    asof = pd.Timestamp(asof)
    window_start = asof - pd.DateOffset(years=1)
    if divs is None or len(divs) == 0:
        ttm = 0.0
    else:
        mask = (divs.index > window_start) & (divs.index <= asof)
        ttm = float(divs[mask].sum())
    px = price.loc[:asof]
    if px.empty:
        return float("nan")
    p = float(px.iloc[-1])
    return ttm / p if p > 0 else float("nan")


# ---------------------------------------------------------------------------
# Annual Dogs portfolio engine (real panel)
# ---------------------------------------------------------------------------
def annual_returns_from_panel(
    panel: dict,
    first_year: int = 2000,
    last_year: int | None = None,
    cost_bps: float = 20.0,
) -> pd.DataFrame:
    """Build the annual Dogs portfolio and the same-basis Dow benchmark from a real panel.

    For each rebalance year *y*:

    1. Take the point-in-time Dow members as of the prior **December** (``data.members_asof``).
    2. Rank them by trailing-12-month dividend yield at the prior December close.
    3. Buy the top ten equal-weight; compound each name's **total return** over the calendar
       year *y*; equal-weight-average them → the Dogs' yearly total return.
    4. Charge ``cost_bps`` (one-way) on the turnover between last year's and this year's
       basket (a name held two years in a row pays nothing; a fully fresh basket pays
       ``cost_bps`` on the whole book).
    5. The benchmark return for year *y* is the Dow ETF (DIA) total return over the same
       calendar year — same total-return basis, so it is a fair race.

    Returns a DataFrame indexed by year with columns ``dogs``, ``dow``, ``excess``,
    ``n_names``, ``turnover``.
    """
    prices = panel["prices"]
    divs = panel["divs"]
    bench = panel["bench"]["close"]
    last_year = last_year or int(prices.index[-1].year) - 1

    rows = []
    prev_basket: set[str] = set()
    for y in range(first_year, last_year + 1):
        rebal = pd.Timestamp(f"{y - 1}-12-31")
        members = _data.members_asof(rebal)
        if not members:
            continue
        # Rank current members by trailing yield at the prior December close.
        yields = {}
        for t in members:
            if t not in prices.columns:
                continue
            px = prices[t].dropna()
            dv = divs[t].dropna() if t in divs.columns else pd.Series(dtype=float)
            yld = trailing_yield(px, dv, rebal)
            if np.isfinite(yld) and yld > 0:
                yields[t] = yld
        if len(yields) < 10:
            continue
        ranked = sorted(yields, key=lambda k: yields[k], reverse=True)
        basket = ranked[:10]

        # Compound each name's total return over calendar year y.
        yr_start = pd.Timestamp(f"{y}-01-01")
        yr_end = pd.Timestamp(f"{y}-12-31")
        names_ret = []
        for t in basket:
            seg = prices[t].loc[yr_start:yr_end].dropna()
            if len(seg) < 2:
                continue
            names_ret.append(float(seg.iloc[-1] / seg.iloc[0] - 1.0))
        if not names_ret:
            continue
        dogs_ret = float(np.mean(names_ret))

        # Turnover: fraction of the equal-weight book that changed names.
        cur = set(basket)
        turnover = len(cur - prev_basket) / len(cur) if prev_basket else 1.0
        dogs_ret_net = dogs_ret - turnover * cost_bps * 1e-4
        prev_basket = cur

        # Benchmark total return over the same calendar year.
        bseg = bench.loc[yr_start:yr_end].dropna()
        if len(bseg) < 2:
            continue
        dow_ret = float(bseg.iloc[-1] / bseg.iloc[0] - 1.0)

        rows.append({
            "year": y, "dogs": dogs_ret_net, "dow": dow_ret,
            "excess": dogs_ret_net - dow_ret, "n_names": len(names_ret),
            "turnover": turnover,
        })

    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# Annual Dogs engine on the synthetic panel (the spine's control)
# ---------------------------------------------------------------------------
def annual_returns_from_synthetic(panel: dict) -> pd.DataFrame:
    """Run the top-10-yield selection on a synthetic panel; return annual dogs/dow/excess.

    The 'Dow' benchmark here is the equal-weight all-names return each year (the synthetic
    universe is the index). The Dogs are the equal-weight top-ten-yield names. This is the
    same selection logic as the real engine, on a tape where the planted truth is known.
    """
    yields = panel["yields"]
    fwd = panel["fwd_returns"]
    rows = []
    for y in yields.index:
        order = yields.loc[y].sort_values(ascending=False)
        basket = list(order.index[:10])
        dogs_ret = float(fwd.loc[y, basket].mean())
        dow_ret = float(fwd.loc[y].mean())
        rows.append({"year": y, "dogs": dogs_ret, "dow": dow_ret,
                     "excess": dogs_ret - dow_ret})
    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# Summary stats on an annual table
# ---------------------------------------------------------------------------
def summarize(annual: pd.DataFrame) -> dict:
    """Headline stats from an annual dogs/dow/excess table.

    Compounds each leg into a CAGR over the sampled years, and reports the mean annual
    excess, win rate, and the count of years.
    """
    dogs = annual["dogs"].to_numpy()
    dow = annual["dow"].to_numpy()
    exc = annual["excess"].to_numpy()
    n = len(exc)
    dogs_cagr = float(np.prod(1 + dogs) ** (1 / n) - 1) if n else float("nan")
    dow_cagr = float(np.prod(1 + dow) ** (1 / n) - 1) if n else float("nan")
    return {
        "n_years": n,
        "dogs_cagr": dogs_cagr,
        "dow_cagr": dow_cagr,
        "cagr_gap": dogs_cagr - dow_cagr,
        "mean_excess": float(exc.mean()) if n else float("nan"),
        "win_rate": float((exc > 0).mean()) if n else float("nan"),
        "total_dogs": float(np.prod(1 + dogs)) if n else float("nan"),
        "total_dow": float(np.prod(1 + dow)) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# Inference: HAC t-stat and a block bootstrap on the annual excess
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x`` (local, no quantlab dep)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def block_bootstrap_ci(
    x: np.ndarray, block: int = 3, n_boot: int = 5000, seed: int = 88, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Circular block-bootstrap CI for the mean of an annual series.

    Resampling whole blocks preserves any year-to-year dependence the i.i.d. bootstrap
    would destroy. Returns ``(lo, hi, p_two_sided)`` where ``p_two_sided`` is the bootstrap
    two-sided p-value for mean == 0.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        means[b] = x[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    # Two-sided bootstrap p-value: how often the resampled mean crosses zero.
    frac_pos = float((means > 0).mean())
    p = 2.0 * min(frac_pos, 1.0 - frac_pos)
    return lo, hi, p


# ---------------------------------------------------------------------------
# Alpha vs beta — is the edge a value/high-yield tilt you were paid for?
# ---------------------------------------------------------------------------
def alpha_beta(annual: pd.DataFrame, rf: float = 0.0) -> dict:
    """OLS of Dogs excess-of-rf return on Dow excess-of-rf return.

    Slope = beta (market/value exposure); intercept = annual alpha. A beta > 1 with an
    alpha whose *t* is below 2 is the classic 'it's just more beta' verdict — the Dogs are
    a higher-yield, higher-beta slice of the Dow, not a free lunch.
    """
    y = annual["dogs"].to_numpy() - rf
    x = annual["dow"].to_numpy() - rf
    n = len(x)
    if n < 3:
        return {"alpha": float("nan"), "beta": float("nan"),
                "alpha_t": float("nan"), "r2": float("nan")}
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - X @ coef
    dof = n - 2
    sigma2 = float(resid @ resid) / dof if dof > 0 else float("nan")
    xtx_inv = np.linalg.inv(X.T @ X)
    se_alpha = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "alpha": alpha, "beta": beta,
        "alpha_t": alpha / se_alpha if se_alpha > 0 else float("nan"),
        "r2": r2,
    }
