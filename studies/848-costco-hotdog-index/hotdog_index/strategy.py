"""Strategy + inference for Study 848 — "Costco Hot-Dog Index" (COST vs CPI vs SPY).

The narrative: the $1.50 hot-dog-and-soda combo has been frozen since ~1985, a folk
"anti-inflation" icon, and Costco's pricing-power / membership model lets its stock
outrun inflation and the market regardless. We test three separable things:

1. **The real-price identity.** The combo is nominally constant, so its *real* price is
   ``1.50 · CPI_1985 / CPI_t`` — a mechanical erosion we *quantify* (not a t-test: it is
   arithmetic). The point is the *magnitude*: how much value has Costco given away.
2. **The total-return race.** COST total return vs CPI (a straight deflator) vs SPY, over
   the stamped window — CAGR, terminal wealth, real (CPI-deflated) multiple. COST winning
   is a real, descriptive fact; we name the hindsight/survivorship caveat.
3. **The *signal* — is COST a distinctive inflation hedge?** Does COST's monthly return
   have a beta to **inflation surprises** (ΔYoY inflation) that (a) is robustly non-zero
   (Newey-West |t| ≥ 2, holding across sub-eras) and (b) differs from a boring consumer-
   staples basket (XLP) and the market (SPY)? A *predictive* version asks whether the
   surprise forecasts COST's *next* month. This is the tradable, falsifiable claim.

A block-bootstrap placebo, a two-era cut, a costed inflation-timing overlay and a
seeded synthetic positive control round it out. All heavy paths are vectorised numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_YEAR = 12


# --------------------------------------------------------------------------- #
# Inference primitives (shared house kit — mirrors 803/823)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    """One-sample t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) − mean(b) (unequal variances)."""
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
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# OLS with a Newey-West slope t (vectorised)
# --------------------------------------------------------------------------- #
def hac_regression(y: np.ndarray, x: np.ndarray, lags: int = 6) -> dict:
    """OLS of ``y`` on a constant + ``x`` with a HAC (Newey-West) *t* on the slope.

    Returns ``slope``, ``alpha``, ``t_nw`` (HAC t on the slope), ``t_ols``
    (homoskedastic), ``r2`` and ``n``. Bartlett kernel, ``lags`` truncation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 10:
        return {"slope": float("nan"), "alpha": float("nan"), "t_nw": float("nan"),
                "t_ols": float("nan"), "r2": float("nan"), "n": n}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    ssr = float((resid ** 2).sum())
    r2 = 1.0 - ssr / sst if sst > 0 else float("nan")
    g = X * resid[:, None]
    S = g.T @ g
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        Gl = g[l:].T @ g[:-l]
        S += w * (Gl + Gl.T)
    V = XtX_inv @ S @ XtX_inv
    se_nw = float(np.sqrt(max(V[1, 1], 0.0)))
    sigma2 = ssr / (n - 2)
    se_ols = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    return {
        "slope": float(beta[1]), "alpha": float(beta[0]),
        "t_nw": float(beta[1] / se_nw) if se_nw > 0 else float("nan"),
        "t_ols": float(beta[1] / se_ols) if se_ols > 0 else float("nan"),
        "r2": r2, "n": n,
    }


# --------------------------------------------------------------------------- #
# 1 · The real-price identity — quantifying the erosion
# --------------------------------------------------------------------------- #
def real_price(cpi: pd.Series, nominal: float = 1.50, base_cpi: float = 107.6) -> pd.Series:
    """Real price of the frozen combo in constant base-year dollars: ``nominal·base/CPI``.

    The combo is nominally ``nominal`` every year; its purchasing-power cost in the base
    year's dollars is ``nominal · base_cpi / CPI_t``. A mechanical identity — the CPI
    CPI series only sets its *magnitude*.
    """
    return (nominal * base_cpi / cpi).rename("real_price")


def erosion_stats(cpi: pd.Series, nominal: float = 1.50, base_cpi: float = 107.6) -> dict:
    """Headline numbers for the real-price collapse of the $1.50 combo.

    ``real_now`` = today's combo in 1985 dollars; ``eroded_pct`` = 1 − real_now/nominal;
    ``inflation_adjusted`` = the price that would preserve 1985 real value (nominal·CPI/base);
    ``giveaway`` = that minus the frozen nominal price (what Costco forgoes per combo).
    """
    cpi_now = float(cpi.iloc[-1])
    real_now = nominal * base_cpi / cpi_now
    infl_adj = nominal * cpi_now / base_cpi
    return {
        "cpi_now": cpi_now,
        "cpi_multiple": cpi_now / base_cpi,             # how much the price level rose
        "real_now": real_now,                           # combo in 1985 dollars
        "eroded_pct": (1.0 - real_now / nominal) * 100.0,
        "inflation_adjusted": infl_adj,                 # price to hold 1985 real value
        "giveaway": infl_adj - nominal,                 # per-combo subsidy vs 1985 pricing
    }


# --------------------------------------------------------------------------- #
# 2 · The total-return race — COST vs CPI vs SPY
# --------------------------------------------------------------------------- #
def total_return_race(panel: pd.DataFrame) -> dict:
    """CAGR / terminal-wealth / real multiple for COST, SPY and CPI over the panel.

    Uses the total-return levels (``cost``, ``spy``) and the CPI level as a pure deflator.
    ``real_mult`` = the CPI-deflated growth multiple (nominal multiple ÷ CPI multiple):
    real purchasing-power growth. Descriptive — COST winning is a fact, not a stamp.
    """
    yrs = (panel.index[-1] - panel.index[0]).days / 365.25 if hasattr(panel.index, "days") \
        else len(panel) / MONTHS_YEAR
    out = {}
    cpi_mult = float(panel["cpi"].iloc[-1] / panel["cpi"].iloc[0])
    for name in ("cost", "spy"):
        mult = float(panel[name].iloc[-1] / panel[name].iloc[0])
        out[name] = {
            "mult": mult,
            "cagr": mult ** (1.0 / yrs) - 1.0,
            "real_mult": mult / cpi_mult,
        }
    out["cpi"] = {"mult": cpi_mult, "cagr": cpi_mult ** (1.0 / yrs) - 1.0, "real_mult": 1.0}
    out["years"] = yrs
    return out


# --------------------------------------------------------------------------- #
# 3 · The signal — is COST a distinctive inflation hedge?
# --------------------------------------------------------------------------- #
def inflation_beta(panel: pd.DataFrame, asset: str = "cost", lag: int = 0,
                   lags: int = 6) -> dict:
    """HAC regression of an asset's monthly return on the inflation surprise (ΔYoY).

    ``lag = 0`` is contemporaneous (does the asset move *with* an inflation surprise — the
    inflation beta); ``lag = 1`` is predictive (does *this* month's surprise forecast
    *next* month's return — the tradable version). Returns slope (the inflation beta),
    HAC/OLS t, R², n.
    """
    ret = panel[f"{asset}_ret"]
    surp = panel["surprise"]
    if lag > 0:
        y = ret.shift(-lag)
        d = pd.DataFrame({"y": y, "x": surp}).dropna()
    else:
        d = pd.DataFrame({"y": ret, "x": surp}).dropna()
    r = hac_regression(d["y"].to_numpy(), d["x"].to_numpy(), lags)
    r["asset"] = asset
    r["lag"] = lag
    return r


def beta_gap(panel: pd.DataFrame, lag: int = 0, lags: int = 6) -> dict:
    """Is COST's inflation-surprise beta different from staples (XLP)?

    Regress the *difference* series ``cost_ret − xlp_ret`` on the surprise: the slope is
    ``β_COST − β_XLP`` and its HAC t tests whether COST's pricing power gives it a
    distinctive inflation exposure over a boring staples basket. Also returns the
    COST-minus-SPY gap.
    """
    surp = panel["surprise"]
    out = {}
    for other in ("xlp", "spy"):
        diff = panel["cost_ret"] - panel[f"{other}_ret"]
        if lag > 0:
            d = pd.DataFrame({"y": diff.shift(-lag), "x": surp}).dropna()
        else:
            d = pd.DataFrame({"y": diff, "x": surp}).dropna()
        r = hac_regression(d["y"].to_numpy(), d["x"].to_numpy(), lags)
        out[f"cost_minus_{other}"] = {"slope": r["slope"], "t_nw": r["t_nw"], "n": r["n"]}
    return out


def signal_summary(panel: pd.DataFrame, lags: int = 6) -> dict:
    """The Signal-axis headline: contemporaneous & predictive inflation betas for all three
    assets, plus the COST-vs-staples beta gap."""
    return {
        "cost_contemp": inflation_beta(panel, "cost", lag=0, lags=lags),
        "spy_contemp": inflation_beta(panel, "spy", lag=0, lags=lags),
        "xlp_contemp": inflation_beta(panel, "xlp", lag=0, lags=lags),
        "cost_pred": inflation_beta(panel, "cost", lag=1, lags=lags),
        "gap": beta_gap(panel, lag=0, lags=lags),
    }


# --------------------------------------------------------------------------- #
# Placebo — is COST's inflation beta a lucky alignment?
# --------------------------------------------------------------------------- #
def placebo_pvalue(panel: pd.DataFrame, asset: str = "cost", n_perm: int = 3000,
                   block: int = 6, seed: int = 848) -> dict:
    """Block-rotation placebo for the contemporaneous inflation-beta slope.

    Circularly rotate the surprise series against the asset return in blocks of ``block``
    months (preserving the surprise's own autocorrelation), ``n_perm`` times. ``p`` = the
    two-sided share of rotations whose |slope| ≥ |observed|. Fully vectorised: one design
    row, ``n_perm`` rolled x-vectors.
    """
    d = pd.DataFrame({"y": panel[f"{asset}_ret"], "x": panel["surprise"]}).dropna()
    x = d["x"].to_numpy(dtype=float)
    y = d["y"].to_numpy(dtype=float)
    n = len(x)
    if n < 30:
        return {"obs_slope": float("nan"), "p_value": float("nan"), "n_perm": 0}
    yc = y - y.mean()
    obs = float((x - x.mean()) @ yc / (((x - x.mean()) ** 2).sum()))
    rng = np.random.default_rng(seed)
    shifts = rng.integers(block, n - block, size=n_perm)
    base = np.arange(n)
    idx = (base[None, :] - shifts[:, None]) % n
    Xr = x[idx]                                    # n_perm × n rolled surprises
    Xc = Xr - Xr.mean(axis=1, keepdims=True)
    slopes = (Xc @ yc) / (Xc ** 2).sum(axis=1)
    return {
        "obs_slope": obs,
        "placebo_mean": float(slopes.mean()),
        "placebo_sd": float(slopes.std(ddof=1)),
        "p_value": float((np.abs(slopes) >= abs(obs)).mean()),
        "n_perm": int(n_perm),
    }


# --------------------------------------------------------------------------- #
# Robustness — two-era cut
# --------------------------------------------------------------------------- #
def era_cut(panel: pd.DataFrame, split: str = "2013-01-01", lags: int = 6) -> pd.DataFrame:
    """COST's contemporaneous inflation-surprise beta within two eras split at ``split``."""
    rows = []
    for lab, lo, hi in [("early", "1900-01-01", split), ("late", split, "2100-01-01")]:
        mask = (panel.index >= _bound(panel.index, lo)) & (panel.index < _bound(panel.index, hi))
        sub = panel[mask]
        r = inflation_beta(sub, "cost", lag=0, lags=lags)
        rows.append((lab, r["slope"], r["t_nw"], r["r2"], r["n"]))
    return pd.DataFrame(rows, columns=["era", "slope", "t_nw", "r2", "n"]).set_index("era")


def _bound(index: pd.Index, s: str):
    """A comparable bound: a ``Period`` for a synthetic ``PeriodIndex``, else a ``Timestamp``."""
    if isinstance(index, pd.PeriodIndex):
        return pd.Period(s, freq=index.freq)
    return pd.Timestamp(s)


# --------------------------------------------------------------------------- #
# The costed inflation-timing overlay — can you get paid for the "hedge"?
# --------------------------------------------------------------------------- #
def timer_stats(panel: pd.DataFrame, cost_bps: float = 10.0, rf_annual: float = 0.02) -> dict:
    """Hold COST when inflation is accelerating (surprise > 0, 1-mo lag), cash otherwise.

    The folk "inflation hedge" made tradable: own COST when the inflation surprise known
    at ``t−1`` is positive, else sit in cash at ``rf_annual``. Charge ``cost_bps`` one-way
    × NAV per switch. Compares the timer's excess-of-cash Sharpe to buy-and-hold COST and
    to SPY. (Long-only directional overlay — no borrow leg.)
    """
    sig = (panel["surprise"].shift(1) > 0).astype(float)
    rf_m = rf_annual / MONTHS_YEAR
    d = pd.DataFrame({"sig": sig, "cost": panel["cost_ret"], "spy": panel["spy_ret"]}).dropna()
    if len(d) < 24:
        return {}
    switch = d["sig"].diff().abs().fillna(0.0)
    timer = d["sig"] * d["cost"] + (1.0 - d["sig"]) * rf_m - switch * cost_bps * 1e-4

    def sharpe(x):
        ex = x - rf_m
        s = ex.std(ddof=1)
        return float(ex.mean() / s * np.sqrt(MONTHS_YEAR)) if s > 0 else float("nan")

    return {
        "timer_sharpe": sharpe(timer),
        "cost_bh_sharpe": sharpe(d["cost"]),
        "spy_sharpe": sharpe(d["spy"]),
        "switches_per_yr": float(switch.sum() / len(d) * MONTHS_YEAR),
        "invested_frac": float(d["sig"].mean()),
        "n": int(len(d)),
    }


# --------------------------------------------------------------------------- #
# Synthetic positive control (seed-robust; the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: pd.DataFrame, lags: int = 6) -> dict:
    """Run the headline inflation-beta stats on a synthetic panel."""
    cb = inflation_beta(panel, "cost", lag=0, lags=lags)
    gap = beta_gap(panel, lag=0, lags=lags)
    return {"slope": cb["slope"], "t_nw": cb["t_nw"], "r2": cb["r2"], "n": cb["n"],
            "gap_xlp_t": gap["cost_minus_xlp"]["t_nw"]}


def synthetic_mean_t(data_mod, edge: float, n_seeds: int = 20, base_seed: int = 848,
                     lags: int = 6) -> dict:
    """Average COST's inflation-beta slope / HAC t over ``n_seeds`` synthetic worlds.

    House rule: any synthetic-dependent claim averages the statistic over ≥ 20 seeds so no
    single lucky RNG seed can manufacture significance.
    """
    slopes, ts, gaps = [], [], []
    for s in range(base_seed, base_seed + n_seeds):
        p = data_mod.synthetic_world(edge=edge, seed=s)
        out = synthetic_detect(p, lags=lags)
        slopes.append(out["slope"]); ts.append(out["t_nw"]); gaps.append(out["gap_xlp_t"])
    return {
        "edge": edge,
        "mean_slope": float(np.nanmean(slopes)),
        "mean_t": float(np.nanmean(ts)),
        "mean_gap_t": float(np.nanmean(gaps)),
        "fire_frac": float(np.mean(np.abs(ts) >= 2.0)),
        "n_seeds": n_seeds,
    }
