"""Small Dogs of the Dow engine — Study 220.

The three strategies under the microscope:

- **Dogs** (top-10 by yield) — the Study 88 baseline, re-computed here from the same panel
  so all three are on an identical footing (no benchmark-period mismatches).
- **Small Dogs** (5 lowest-priced of the 10 Dogs) — the retail shortcut popularised in the
  mid-1990s as the "Small Dogs of the Dow" or "Low-5": same annual January rebalance, same
  total-return accounting, but only 5 names.
- **Foolish Four** (names #2-#5 by price ascending within the 10 Dogs — skip the cheapest
  one, keep the next four equal-weight) — the Motley Fool variant from 1996, since debunked
  as a pure data-mining artefact. The skip-the-cheapest quirk was "discovered" by sorting
  historical results, not by any theory.

All three race the Dow benchmark (DIA total return) and the full Dogs basket on the *same
total-return + 20 bps/turnover basis*. No look-ahead: selection uses only information
available at each January rebalance (trailing-12-month yield, absolute share price at the
prior December close).

The inference questions we ask for each subset:
1. Does it beat the Dow? — HAC t and block-bootstrap on the annual excess vs DIA.
2. Does it beat the Dogs? — same tests on subset minus Dogs excess return.
3. Is any edge real alpha or just beta? — alpha-vs-beta OLS vs the Dow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from dogs_of_the_dow import data as _dogs_data  # type: ignore[import]
    _members_asof = _dogs_data.members_asof
    _trailing_yield = None  # will import below
except ImportError:
    _members_asof = None

# Re-use trailing_yield from Study 88 strategy (same logic).
try:
    from dogs_of_the_dow.strategy import trailing_yield  # type: ignore[import]
except ImportError:
    def trailing_yield(price, divs, asof):  # type: ignore[misc]
        """Fallback trailing yield (used only when Study 88 is unavailable)."""
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

try:
    from . import data as _data
    _members_asof = _data.members_asof
except Exception:
    pass


# ---------------------------------------------------------------------------
# Annual portfolio engine (real panel)
# ---------------------------------------------------------------------------
def annual_returns_from_panel(
    panel: dict,
    first_year: int = 2000,
    last_year: int | None = None,
    cost_bps: float = 20.0,
) -> pd.DataFrame:
    """Build annual returns for Dogs, Small Dogs, Foolish Four, and the Dow benchmark.

    Selection (no look-ahead):
    - Dogs: rank the then-current 30 Dow members by trailing-12-month yield at the prior
      December close; take the top 10.
    - Small Dogs: of the Dogs 10, take the 5 with the *lowest absolute share price* at the
      same December close.
    - Foolish Four: of the Dogs 10, sort by price ascending, skip the #1 cheapest, take
      #2-#5 (4 names).

    Each portfolio is equal-weight. Turnover cost (``cost_bps`` one-way) is charged on the
    fraction of the prior year's basket that changes each January.

    Returns a DataFrame indexed by year with columns:
    ``dogs``, ``small_dogs``, ``foolish4``, ``dow``,
    ``excess_dogs``, ``excess_small``, ``excess_f4``,
    ``dogs_vs_small`` (small-dogs minus dogs), ``dogs_vs_f4``,
    ``n_dogs``, ``turnover_dogs``, ``turnover_small``, ``turnover_f4``.
    """
    from . import data as _d
    prices = panel["prices"]
    divs = panel["divs"]
    bench = panel["bench"]["close"]
    last_year = last_year or int(prices.index[-1].year) - 1

    rows = []
    prev_dogs: set[str] = set()
    prev_small: set[str] = set()
    prev_f4: set[str] = set()

    for y in range(first_year, last_year + 1):
        rebal = pd.Timestamp(f"{y - 1}-12-31")
        members = _d.members_asof(rebal)
        if not members:
            continue

        # --- Step 1: rank by trailing yield (Dogs selection) ---
        yields = {}
        px_at_rebal = {}
        for t in members:
            if t not in prices.columns:
                continue
            px = prices[t].dropna()
            dv = divs[t].dropna() if t in divs.columns else pd.Series(dtype=float)
            yld = trailing_yield(px, dv, rebal)
            if np.isfinite(yld) and yld > 0:
                yields[t] = yld
                px_slice = px.loc[:rebal]
                if not px_slice.empty:
                    px_at_rebal[t] = float(px_slice.iloc[-1])

        if len(yields) < 10:
            continue

        ranked_by_yield = sorted(yields, key=lambda k: yields[k], reverse=True)
        dogs_10 = ranked_by_yield[:10]

        # --- Step 2: sub-select by price ---
        # Filter to Dogs that have a price at rebalance.
        dogs_with_price = [t for t in dogs_10 if t in px_at_rebal]
        if len(dogs_with_price) < 5:
            continue

        ranked_by_price = sorted(dogs_with_price, key=lambda k: px_at_rebal[k])
        small_dogs_5 = ranked_by_price[:5]           # 5 cheapest of the 10 Dogs
        foolish_four = ranked_by_price[1:5]           # skip #1 cheapest, take #2-#5

        # --- Step 3: compute calendar-year total returns for each basket ---
        yr_start = pd.Timestamp(f"{y}-01-01")
        yr_end = pd.Timestamp(f"{y}-12-31")

        def basket_return(tickers):
            rets = []
            for t in tickers:
                seg = prices[t].loc[yr_start:yr_end].dropna()
                if len(seg) >= 2:
                    rets.append(float(seg.iloc[-1] / seg.iloc[0] - 1.0))
            return float(np.mean(rets)) if rets else float("nan")

        dogs_ret_gross = basket_return(dogs_10)
        small_ret_gross = basket_return(small_dogs_5)
        f4_ret_gross = basket_return(foolish_four)

        # --- Step 4: turnover costs ---
        cur_d = set(dogs_10)
        cur_s = set(small_dogs_5)
        cur_f4 = set(foolish_four)
        to_d = len(cur_d - prev_dogs) / len(cur_d) if prev_dogs else 1.0
        to_s = len(cur_s - prev_small) / len(cur_s) if prev_small else 1.0
        to_f4 = len(cur_f4 - prev_f4) / len(cur_f4) if prev_f4 else 1.0
        prev_dogs, prev_small, prev_f4 = cur_d, cur_s, cur_f4

        dogs_ret = dogs_ret_gross - to_d * cost_bps * 1e-4
        small_ret = small_ret_gross - to_s * cost_bps * 1e-4
        f4_ret = f4_ret_gross - to_f4 * cost_bps * 1e-4

        # --- Step 5: benchmark ---
        bseg = bench.loc[yr_start:yr_end].dropna()
        if len(bseg) < 2:
            continue
        dow_ret = float(bseg.iloc[-1] / bseg.iloc[0] - 1.0)

        rows.append({
            "year": y,
            "dogs": dogs_ret,
            "small_dogs": small_ret,
            "foolish4": f4_ret,
            "dow": dow_ret,
            "excess_dogs": dogs_ret - dow_ret,
            "excess_small": small_ret - dow_ret,
            "excess_f4": f4_ret - dow_ret,
            "small_vs_dogs": small_ret - dogs_ret,
            "f4_vs_dogs": f4_ret - dogs_ret,
            "n_dogs": len(dogs_10),
            "turnover_dogs": to_d,
            "turnover_small": to_s,
            "turnover_f4": to_f4,
        })

    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# Synthetic panel engine
# ---------------------------------------------------------------------------
def annual_returns_from_synthetic(panel: dict) -> pd.DataFrame:
    """Run all three strategies on a synthetic panel; return annual returns.

    The benchmark is the equal-weight all-names return each year.
    """
    yields = panel["yields"]
    prices_abs = panel["prices_abs"]
    fwd = panel["fwd_returns"]
    rows = []

    for y in yields.index:
        yld_row = yields.loc[y]
        px_row = prices_abs.loc[y]
        fwd_row = fwd.loc[y]

        # Dogs: top-10 by yield
        yield_order = yld_row.sort_values(ascending=False)
        dogs = list(yield_order.index[:10])

        # Small Dogs: 5 cheapest of the Dogs
        dogs_px = px_row[dogs].sort_values()
        small_dogs = list(dogs_px.index[:5])

        # Foolish Four: skip #1 cheapest, take #2-#5
        foolish_four = list(dogs_px.index[1:5])

        dogs_ret = float(fwd_row[dogs].mean())
        small_ret = float(fwd_row[small_dogs].mean())
        f4_ret = float(fwd_row[foolish_four].mean())
        bench_ret = float(fwd_row.mean())

        rows.append({
            "year": y,
            "dogs": dogs_ret,
            "small_dogs": small_ret,
            "foolish4": f4_ret,
            "dow": bench_ret,
            "excess_dogs": dogs_ret - bench_ret,
            "excess_small": small_ret - bench_ret,
            "excess_f4": f4_ret - bench_ret,
            "small_vs_dogs": small_ret - dogs_ret,
            "f4_vs_dogs": f4_ret - dogs_ret,
        })

    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
def summarize(annual: pd.DataFrame, col: str = "small_dogs", bench: str = "dow") -> dict:
    """Headline stats for one strategy column vs a benchmark column."""
    strat = annual[col].to_numpy()
    bmark = annual[bench].to_numpy()
    exc = strat - bmark
    n = len(exc)
    strat_cagr = float(np.prod(1 + strat) ** (1 / n) - 1) if n else float("nan")
    bench_cagr = float(np.prod(1 + bmark) ** (1 / n) - 1) if n else float("nan")
    return {
        "n_years": n,
        "strat_cagr": strat_cagr,
        "bench_cagr": bench_cagr,
        "cagr_gap": strat_cagr - bench_cagr,
        "mean_excess": float(exc.mean()) if n else float("nan"),
        "win_rate": float((exc > 0).mean()) if n else float("nan"),
        "total_strat": float(np.prod(1 + strat)) if n else float("nan"),
        "total_bench": float(np.prod(1 + bmark)) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# HAC t-stat and block bootstrap (local, no quantlab dep)
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x``."""
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
    x: np.ndarray, block: int = 3, n_boot: int = 5000, seed: int = 220, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Circular block-bootstrap 95% CI and p-value for the mean of an annual series."""
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
    frac_pos = float((means > 0).mean())
    p = 2.0 * min(frac_pos, 1.0 - frac_pos)
    return lo, hi, p


# ---------------------------------------------------------------------------
# Alpha-vs-beta
# ---------------------------------------------------------------------------
def alpha_beta(annual: pd.DataFrame, strat_col: str = "small_dogs",
               bench_col: str = "dow", rf: float = 0.0) -> dict:
    """OLS of strategy-minus-rf excess on benchmark-minus-rf excess."""
    y = annual[strat_col].to_numpy() - rf
    x = annual[bench_col].to_numpy() - rf
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
