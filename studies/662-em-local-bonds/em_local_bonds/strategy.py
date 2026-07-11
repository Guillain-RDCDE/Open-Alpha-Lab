"""Strategy + inference for Study 662 — EM-Local-Bonds.

The claim: **emerging-market LOCAL-currency bonds pay a fat yield that compensates you for the
risk — a better carry than USD-denominated EM debt.** Local-currency sovereign debt (EBND /
LEMB) typically quotes a materially higher running yield than the hard-currency EMBI benchmark
(EMB), because the buyer is paid both a sovereign-credit premium *and* a local-rates premium.
The question is whether that extra promised yield shows up as extra **realized, risk-adjusted**
total return once the local currency actually moves — or whether FX depreciation quietly eats
it, leaving the local investor with more volatility for the same (or worse) payoff.

This is a **static buy-and-hold comparison, not a signal strategy** — there is no discretionary
entry/exit to lag. The desk's "one documented execution lag" convention here is: **none is
needed**, exactly like the calendar-known rules the house style exempts (turn-of-month windows)
— we hold each fund continuously and compare month-end-to-month-end total returns. Costs are
charged once, at entry (one-way × NAV, 5/10 bps), not as a repeated turnover drag: this is a
buy-and-hold exposure comparison, and the one-time entry cost is reported explicitly rather than
assumed away.

Measurements:

* **Headline — the collected spread.** Monthly total-return-of-cash-excess for a "Local" basket
  (simple average of EBND and LEMB, a monthly-rebalanced 50/50 blend across two different index
  families) minus the same for EMB (USD EM). Paired one-sample *t*, a Newey-West (HAC) *t* at
  three lag choices (3/6/12 months) as the serial-correlation-robust primary, and a circular
  block-bootstrap 95% CI on the annualized mean gap.
* **Sharpe race** — excess-of-cash (BIL) Sharpe for Local vs EMB vs AGG.
* **The FX-drag regression** — each leg's monthly return regressed on UUP (the dollar-strength
  proxy), Newey-West beta + *t* + correlation; then the *incremental* channel, isolated by
  regressing the Local-minus-EMB *difference itself* on UUP (removes the common EM-credit-cycle
  component both legs share).
* **Named crisis windows** — cumulative return of Local / EMB / AGG / the dollar inside the
  2013 taper tantrum, the 2015 EM-FX selloff and the 2022 strong-dollar episode.
* **Hit rate + max drawdown** — how often Local beat EMB month-to-month (Wilson interval), and
  each leg's worst peak-to-trough drawdown (with date).

The decisive number is the Newey-West *t* of the collected Local-minus-EMB spread on the REAL
tape; everything else explains *why* it comes out the way it does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 12  # monthly data


# --------------------------------------------------------------------------- #
# Returns plumbing
# --------------------------------------------------------------------------- #
def monthly_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Month-end total-return simple returns from a daily adjusted-close price panel."""
    m = px.resample("ME").last()
    return m.pct_change()


def local_basket(ret: pd.DataFrame) -> pd.Series:
    """Local-currency EM basket: simple average of EBND and LEMB monthly returns."""
    return ret[["EBND", "LEMB"]].mean(axis=1)


def apply_entry_cost(ann_ret: float, n_months: int, cost_bps: float) -> float:
    """Amortize a ONE-TIME one-way entry cost (bps of NAV) over ``n_months`` of holding.

    Buy-and-hold exposure comparison: costs are paid once at entry, not on every bar. Reported
    explicitly (gross vs net) even though the annualized drag is necessarily tiny over a
    multi-year hold — see the house rule that costs must always be shown, not assumed away.
    """
    years = n_months / ANN
    drag_ann = (cost_bps / 1e4) / years
    return ann_ret - drag_ann


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def paired_t(diff: np.ndarray) -> tuple[float, float, int]:
    """One-sample t of a paired difference series: (t, mean, n)."""
    x = np.asarray(diff, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    t = float(x.mean() / se) if se > 0 else float("nan")
    return t, float(x.mean()), n


def newey_west_mean_t(x: np.ndarray, lags: int = 6) -> tuple[float, float]:
    """HAC (Newey-West, Bartlett kernel) t of the sample mean of ``x``. Returns (t, se)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    mu = x.mean()
    u = x - mu
    s = (u ** 2).sum()
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        s += 2.0 * w * (u[l:] * u[:-l]).sum()
    var_mean = s / (n * n)
    se = float(np.sqrt(var_mean)) if var_mean > 0 else float("nan")
    return (float(mu / se) if se > 0 else float("nan")), se


def newey_west_ols(y: np.ndarray, x: np.ndarray, lags: int = 6) -> dict:
    """HAC (Newey-West, Bartlett kernel) OLS of y = a + b*x. Returns beta, t, se, corr."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    keep = ~(np.isnan(y) | np.isnan(x))
    y, x = y[keep], x[keep]
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    s = X * u[:, None]
    S = s.T @ s
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = s[l:].T @ s[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return {
        "alpha": float(beta[0]), "beta": float(beta[1]),
        "t_beta": float(beta[1] / se[1]) if se[1] > 0 else float("nan"),
        "se_beta": float(se[1]),
        "corr": float(np.corrcoef(x, y)[0, 1]),
        "n": n,
    }


def circular_block_bootstrap_ci(x: np.ndarray, block: int = 6, n_boot: int = 5_000,
                                 seed: int = 662, ann: int = ANN) -> dict:
    """Circular block-bootstrap 95% CI on the ANNUALIZED mean of ``x``.

    Blocks (not i.i.d. resampling) so the CI respects the series' own autocorrelation —
    the house convention (i.i.d. resampling of returns understates uncertainty when returns
    cluster serially, as monthly bond total returns do).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        for _ in range(n_blocks):
            s0 = rng.integers(0, n)
            idx.extend([(s0 + k) % n for k in range(block)])
        idx = idx[:n]
        means[b] = x[idx].mean() * ann
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "boot_mean": float(means.mean()),
            "point": float(x.mean() * ann), "n_boot": n_boot, "block": block}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def max_drawdown(x: pd.Series) -> tuple[float, pd.Timestamp]:
    """Peak-to-trough max drawdown of a return series and the date it bottomed."""
    cum = (1.0 + x).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return float(dd.min()), dd.idxmin()


def sharpe(x: pd.Series, ann: int = ANN) -> float:
    x = x.dropna()
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")


# --------------------------------------------------------------------------- #
# The headline split — collected Local-minus-EMB excess-of-cash spread
# --------------------------------------------------------------------------- #
def headline_spread(ret: pd.DataFrame, versus: str = "EMB") -> dict:
    """Local-basket minus ``versus`` excess-of-cash spread: paired t, NW t (3 lags), bootstrap."""
    cash = ret["BIL"]
    local = local_basket(ret)
    diff = (local - ret[versus]).dropna()

    t_paired, mean_diff, n = paired_t(diff.values)
    nw = {l: newey_west_mean_t(diff.values, lags=l)[0] for l in (3, 6, 12)}
    boot = circular_block_bootstrap_ci(diff.values, block=6)

    hits = int((diff > 0).sum())
    wlo, whi = wilson_interval(hits, n)

    excess = pd.DataFrame({
        "Local": local - cash, versus: ret[versus] - cash, "AGG": ret["AGG"] - cash,
    }).dropna()

    return {
        "n": n, "mean_diff_mo": mean_diff, "mean_diff_ann": mean_diff * ANN,
        "t_paired": t_paired, "nw_t_lag3": nw[3], "nw_t_lag6": nw[6], "nw_t_lag12": nw[12],
        "boot_lo": boot["lo"], "boot_hi": boot["hi"], "boot_mean": boot["boot_mean"],
        "hits": hits, "n_pairs": n, "hit_rate": hits / n, "wilson_lo": wlo, "wilson_hi": whi,
        "sharpe_local": sharpe(excess["Local"]), "sharpe_versus": sharpe(excess[versus]),
        "sharpe_agg": sharpe(excess["AGG"]),
    }


# --------------------------------------------------------------------------- #
# FX-drag regressions
# --------------------------------------------------------------------------- #
def fx_beta_table(ret: pd.DataFrame, lags: int = 6) -> dict:
    """Each leg's monthly return regressed on UUP (dollar); plus the isolated diff-vs-dollar."""
    dollar = ret["UUP"]
    local = local_basket(ret)
    out = {}
    for name, series in (("Local", local), ("EMB", ret["EMB"]), ("AGG", ret["AGG"])):
        aligned = pd.concat([series, dollar], axis=1).dropna()
        out[name] = newey_west_ols(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values, lags=lags)
    diff = (local - ret["EMB"]).dropna()
    aligned = pd.concat([diff, dollar], axis=1).dropna()
    out["Local-minus-EMB"] = newey_west_ols(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values,
                                             lags=lags)
    return out


# --------------------------------------------------------------------------- #
# Named crisis windows
# --------------------------------------------------------------------------- #
def crisis_window_table(ret: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Cumulative return of Local / EMB / AGG / UUP inside each named window."""
    local = local_basket(ret)
    rows = []
    for label, (lo, hi) in windows.items():
        m = (ret.index >= lo) & (ret.index <= hi)
        rows.append({
            "window": label, "n_months": int(m.sum()),
            "local": float((1 + local[m]).prod() - 1),
            "emb": float((1 + ret["EMB"][m]).prod() - 1),
            "agg": float((1 + ret["AGG"][m]).prod() - 1),
            "dollar": float((1 + ret["UUP"][m]).prod() - 1),
        })
    return pd.DataFrame(rows).set_index("window")


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(local: pd.Series, usd: pd.Series) -> dict:
    """Run the headline paired split (t, NW t) on a synthetic (local, usd) pair."""
    diff = (local - usd).dropna()
    t_paired, mean_diff, n = paired_t(diff.values)
    nw_t, _ = newey_west_mean_t(diff.values, lags=6)
    return {"n": n, "mean_diff_ann": mean_diff * ANN, "t_paired": t_paired, "nw_t": nw_t}
