"""Strategy + inference for Study 849 — Dry January / Veganuary.

The claim: the January cultural waves move the stocks. Concretely, in the **abnormal**
return (group total-return minus the ``SPY`` market benchmark, so we strip the market's own
January seasonality out):

* the **alcohol basket** (equal-weight ``BUD, STZ, TAP, DEO, SAM``) under-performs in
  January (Dry-January demand drop) and maybe bounces in February;
* **plant-based** ``BYND`` over-performs in January (Veganuary demand pulse);
* **staples** ``XLP`` is a flat control;
* the pure market-neutral expression of the combined thesis is the **plant − alcohol**
  spread, which should be positive in January.

This is a **monthly calendar-seasonality** test, distinct from the market-wide
"sell-in-May / Halloween" seasonals and from single-name event run-ups:

* [55-summer-lull](../../55-summer-lull/) — the *summer* volume/return lull, a different
  season and a market-wide claim;
* [95-holiday-cheer](../../95-holiday-cheer/) — the late-December "Santa-rally" window;
* [641-sell-in-may](../../641-sell-in-may/) — the May→October market-wide seasonal;
* [723-guacamole-bowl](../../723-guacamole-bowl/) — a Super-Bowl / avocado single-event
  demand pulse, not a whole-month calendar effect;
* [775-halloween-candy](../../775-halloween-candy/) — a fixed-date (Oct-31) single-name
  run-up window, not a calendar-*month* seasonal across a themed basket.

Method:

* **Monthly build.** Resample each daily total-return close to month-end and take the
  month's simple return. The alcohol basket is the equal-weight mean of the available
  alcohol names each month. The abnormal return of a group is ``group − SPY``.
* **January seasonality.** Across all Januaries in the sample, a one-sample *t* of the
  abnormal return (the correct unit — one independent observation per year), a Wilson
  hit-rate, and — the desk-standard robust read — a **Newey-West** *t* on the coefficient of
  a January dummy in a monthly regression of the abnormal return.
* **February "hangover."** The same for February.
* **Placebo across the other eleven months.** Compute each calendar month's mean abnormal
  return; where does January rank? A calendar permutation that asks whether January is
  special or just one of twelve draws.
* **Costed calendar timer.** A once-a-year long-plant / short-alcohol January trade, net of
  round-trip friction and short borrow — is there anything to bank?

All heavy paths are vectorised numpy — no per-date pandas ``.loc`` loops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

MONTHS_YEAR = 12
COST_BPS = 5.0          # one-way, per leg
BORROW_BPS_YR = 50.0    # annual borrow on the short leg


# --------------------------------------------------------------------------- #
# Monthly build: daily closes -> monthly returns -> abnormal returns
# --------------------------------------------------------------------------- #
def monthly_return(close: pd.Series) -> pd.Series:
    """Month-end simple return of a daily total-return close series (index = month-end)."""
    me = close.sort_index().resample("ME").last()
    return me.pct_change().rename(close.name)


def basket_monthly(closes: dict[str, pd.Series], names: list[str]) -> pd.Series:
    """Equal-weight monthly return of the available ``names`` (mean across names per month).

    Each name contributes its own month-end simple return; months where a name has no data
    are simply skipped for that name (equal-weight over whoever is live), so the basket is
    defined from the first month any constituent trades. Vectorised: one wide frame, a
    row-wise ``nanmean``.
    """
    cols = {n: monthly_return(closes[n]) for n in names if n in closes}
    if not cols:
        return pd.Series(dtype=float)
    wide = pd.DataFrame(cols).sort_index()
    return wide.mean(axis=1, skipna=True).rename("basket")


def abnormal_monthly(group_ret: pd.Series, bench_ret: pd.Series) -> pd.Series:
    """Group monthly return minus the benchmark monthly return, on the shared month-ends."""
    d = pd.DataFrame({"g": group_ret, "b": bench_ret}).dropna()
    return (d["g"] - d["b"]).rename("ar")


def build_abnormals(closes: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """The four headline abnormal-return series (monthly), all vs ``SPY``.

    * ``alcohol`` — equal-weight alcohol basket − SPY
    * ``plant``   — BYND − SPY
    * ``staples`` — XLP − SPY
    * ``plant_minus_alcohol`` — BYND − alcohol basket (market-neutral pure thesis)
    """
    bench = monthly_return(closes[dt.BENCH])
    alc = basket_monthly(closes, dt.ALCOHOL)
    plant = basket_monthly(closes, dt.PLANT)
    staples = basket_monthly(closes, dt.STAPLES)
    out = {
        "alcohol": abnormal_monthly(alc, bench),
        "plant": abnormal_monthly(plant, bench),
        "staples": abnormal_monthly(staples, bench),
    }
    pa = pd.DataFrame({"p": plant, "a": alc}).dropna()
    out["plant_minus_alcohol"] = (pa["p"] - pa["a"]).rename("ar")
    return out


# --------------------------------------------------------------------------- #
# Inference primitives (shared house kit)
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


def newey_west_t(x: np.ndarray, lags: int = 3) -> float:
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


def hit_rate(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    k = int((x > 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


# --------------------------------------------------------------------------- #
# The dummy-regression Newey-West seasonality t (the robust, desk-standard read)
# --------------------------------------------------------------------------- #
def dummy_regression_nw(ar: pd.Series, month: int = 1, lags: int = 3) -> dict:
    """Regress the monthly abnormal return on a ``month`` dummy; HAC *t* on the coefficient.

    ``ar_t = alpha + beta · 1[month(t) == month] + eps`` — ``beta`` is the mean abnormal
    return in the target month over-and-above the other-month baseline ``alpha``. A
    Newey-West (Bartlett) standard error guards against any residual serial correlation.
    Vectorised OLS + HAC sandwich. Returns ``beta`` (target-month premium), its HAC *t*, the
    homoskedastic OLS *t*, the baseline ``alpha``, R² and n.
    """
    s = ar.dropna()
    y = s.to_numpy(dtype=float)
    d = (s.index.month == month).astype(float)
    n = len(y)
    if n < 12:
        return {"beta": float("nan"), "alpha": float("nan"), "t_nw": float("nan"),
                "t_ols": float("nan"), "r2": float("nan"), "n": n}
    X = np.column_stack([np.ones(n), d])
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
        "beta": float(beta[1]),
        "alpha": float(beta[0]),
        "t_nw": float(beta[1] / se_nw) if se_nw > 0 else float("nan"),
        "t_ols": float(beta[1] / se_ols) if se_ols > 0 else float("nan"),
        "r2": r2,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# Per-month seasonality table + the January/February headline
# --------------------------------------------------------------------------- #
def month_table(ar: pd.Series) -> pd.DataFrame:
    """Mean, one-sample *t* and n of the abnormal return for each calendar month 1..12."""
    s = ar.dropna()
    rows = []
    for m in range(1, 13):
        x = s[s.index.month == m].to_numpy(dtype=float)
        rows.append((m, float(np.mean(x)) if len(x) else float("nan"),
                     one_sample_t(x), int(len(x))))
    return pd.DataFrame(rows, columns=["month", "mean", "t", "n"]).set_index("month")


def month_stats(ar: pd.Series, month: int = 1) -> dict:
    """Headline seasonality stats for one calendar ``month`` of an abnormal-return series."""
    s = ar.dropna()
    x = s[s.index.month == month].to_numpy(dtype=float)
    reg = dummy_regression_nw(ar, month=month)
    hr = hit_rate(x)
    return {
        "month": month,
        "n": int(len(x)),
        "mean_pct": float(np.mean(x) * 100.0) if len(x) else float("nan"),
        "t_1s": one_sample_t(x),
        "beta_pct": reg["beta"] * 100.0,        # premium vs the other-month baseline
        "t_nw": reg["t_nw"],
        "t_ols": reg["t_ols"],
        "hit_k": hr["k"], "hit_n": hr["n"], "hit_rate": hr["rate"],
        "hit_lo": hr["lo"], "hit_hi": hr["hi"],
    }


# --------------------------------------------------------------------------- #
# Placebo across the other eleven months — is January special or one-of-twelve?
# --------------------------------------------------------------------------- #
def month_placebo(ar: pd.Series, target: int = 1, tail: str = "two") -> dict:
    """Rank the target month's mean abnormal return among all twelve calendar months.

    The natural calendar placebo: if January carried a real demand shift, its mean abnormal
    return should stand out from the eleven other monthly means. ``tail``:

    * ``"left"``  — claim of a NEGATIVE target (e.g. alcohol Dry-January drag): p = share of
      months whose mean is ≤ the target's (target among the most-negative).
    * ``"right"`` — claim of a POSITIVE target (e.g. plant Veganuary lift): p = share ≥ target.
    * ``"two"``   — magnitude: p = share of months at least as far from 0 as the target.

    Returns the target's mean, its rank (1 = most extreme in the claimed direction), the
    twelve monthly means, and p.
    """
    tab = month_table(ar)
    means = tab["mean"].to_numpy(dtype=float)
    tgt = float(tab.loc[target, "mean"])
    if tail == "left":
        p = float(np.mean(means <= tgt))
        rank = int((means <= tgt).sum())          # 1 = most negative
    elif tail == "right":
        p = float(np.mean(means >= tgt))
        rank = int((means >= tgt).sum())          # 1 = most positive
    else:
        p = float(np.mean(np.abs(means) >= abs(tgt)))
        rank = int((np.abs(means) >= abs(tgt)).sum())
    return {"target": target, "target_mean_pct": tgt * 100.0, "rank": rank,
            "n_months": 12, "p_value": p, "means_pct": (means * 100.0).tolist()}


# --------------------------------------------------------------------------- #
# The costed calendar timer — long plant / short alcohol every January
# --------------------------------------------------------------------------- #
def timer_stats(closes: dict[str, pd.Series], cost_bps: float = COST_BPS,
                borrow_bps_yr: float = BORROW_BPS_YR) -> dict:
    """Once-a-year January long-plant / short-alcohol calendar trade, net of frictions.

    Enter at the December month-end close, hold the plant-minus-alcohol book through January,
    exit at the January month-end close (calendar-known, zero look-ahead). Charge a
    round-trip (2 × one-way) on each of the two legs plus one month of short borrow on the
    alcohol leg. Reports gross/net mean January return, the one-sample *t*, and the hit rate.
    """
    abn = build_abnormals(closes)
    pa = abn["plant_minus_alcohol"]
    jan = pa[pa.index.month == 1].to_numpy(dtype=float)
    jan = jan[~np.isnan(jan)]
    rt = 2.0 * (2.0 * cost_bps / 1e4)              # round-trip on TWO legs
    borrow = (borrow_bps_yr / 1e4) / 12.0          # one month of borrow on the short leg
    net = jan - rt - borrow
    n = len(jan)
    hr = hit_rate(jan)
    return {
        "n_years": n,
        "gross_pct": float(np.mean(jan) * 100.0) if n else float("nan"),
        "net_pct": float(np.mean(net) * 100.0) if n else float("nan"),
        "cost_pct": (rt + borrow) * 100.0,
        "t_gross": one_sample_t(jan),
        "t_net": one_sample_t(net),
        "hit_k": hr["k"], "hit_n": hr["n"], "hit_rate": hr["rate"],
    }


# --------------------------------------------------------------------------- #
# Synthetic positive control (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.Series], month: int = 1) -> dict:
    """Run the January-seasonality detector on a synthetic (NAME, SPY) panel."""
    name = monthly_return(panel["NAME"])
    bench = monthly_return(panel[dt.BENCH])
    ar = abnormal_monthly(name, bench)
    return month_stats(ar, month=month)


def synthetic_mean_t(data_mod, edge: float, n_seeds: int = 20, base_seed: int = 849,
                     month: int = 1) -> dict:
    """Average the January dummy HAC *t* over ``n_seeds`` synthetic worlds (house rule ≥20).

    No single lucky RNG seed can manufacture significance: the null (``edge = 0``) must
    average |t| well under 2, and a planted ``edge > 0`` must lift it monotonically.
    """
    ts, betas, t1s = [], [], []
    for s in range(base_seed, base_seed + n_seeds):
        panel = data_mod.synthetic_panel(edge=edge, seed=s)
        out = synthetic_detect(panel, month=month)
        ts.append(out["t_nw"]); betas.append(out["beta_pct"]); t1s.append(out["t_1s"])
    ts = np.asarray(ts, dtype=float)
    return {
        "edge": edge,
        "mean_t_nw": float(np.nanmean(ts)),
        "mean_t_1s": float(np.nanmean(t1s)),
        "mean_beta_pct": float(np.nanmean(betas)),
        "fire_frac": float(np.mean(np.abs(ts) >= 2.0)),
        "n_seeds": n_seeds,
    }
