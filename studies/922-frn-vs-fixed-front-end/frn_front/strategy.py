"""Race and inference for Study 922 — Floating-Rate Front End.

The question is a portfolio question, not a timing question: for the cash sleeve of a
portfolio, does a **Treasury floating-rate-note fund** (USFR, TFLO — notes whose coupon
resets weekly off the 13-week bill auction) pay more than **fixed** front-end paper —
the T-bill fund (BIL) and the 1-3 year fixed-coupon fund (SHY) — and does the answer
flip with the direction of the short rate?

The mechanics that make it a real question:

- A floater has (essentially) **no duration**: its coupon reprices with the bill rate,
  so its price barely moves when rates move. It earns the bill rate plus a small
  *discount margin*.
- A **bill ladder** reprices only as bills mature, so it lags the short rate by weeks.
- A **1-3y fixed** fund earns a term premium but carries ~1.85 years of duration: it
  loses ~1.85% of price per 1 pp of yield rise, and gains it back when yields fall.

So theory says: floaters win the hiking cycle, fixed wins the cutting cycle, and the
plateau is decided by the term premium. This study tests that on the tape.

**No strategy signal, one execution lag.** The four funds are held, not traded — the
race is buy-and-hold vs buy-and-hold, so there is no signal to lag. The single lag in
the study is applied to every ``^IRX``-derived object: the regime label and the cash
accrual rate are formed from data through day ``t`` and applied to day ``t+1``'s
returns. That lag is implemented once, in :func:`irx_regime` and :func:`cash_leg`.

**Costs.** Buy-and-hold sleeves pay a spread on the way in and on the way out only, so
friction is charged as ``2 x cost_bps x NAV`` amortised over the holding horizon
(:func:`cost_sweep`). Expressed as a *long/short pair* — long the floater, short the
fixed leg — the short side also pays borrow, swept in :func:`borrow_sweep`. Expense
ratios are already inside the total-return closes and are never added on top.

**Excess-of-cash.** Sharpe comparisons are excess-of-cash on both sides, with cash the
``^IRX`` accrual proxy (see ``data`` for the declared convention) or, as a robustness
alternative, BIL's own total return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# ASSUMPTION (declared, not derived from the tape): a hardcoded reading of Fed history.
# The mechanical alternative is `irx_regime`, which uses only the ^IRX tape.
CYCLE_WINDOWS = [
    ("ZIRP + first hikes", "2014-02-04", "2021-12-31"),
    ("hiking", "2022-03-01", "2023-07-31"),
    ("plateau", "2023-08-01", "2024-08-31"),
    ("cutting", "2024-09-01", "2026-06-30"),
]

# The liquidity-era split. USFR and TFLO launched in Feb 2014 and traded by appointment
# for their first few years; their daily closes in 2014-2017 carry quote artefacts an
# order of magnitude larger than the economics under test (see docs/results.md).
LIQUIDITY_SPLIT = "2018-01-01"


# --------------------------------------------------------------------------- #
# Returns, cash leg, excess frames
# --------------------------------------------------------------------------- #
def daily_returns(prices: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Simple daily total returns for ``cols`` (default: every column bar ``IRX``)."""
    if cols is None:
        cols = [c for c in prices.columns if c != "IRX"]
    return prices[list(cols)].pct_change().dropna()


def cash_leg(irx: pd.Series, basis: int = TRADING_DAYS) -> pd.Series:
    """Daily cash accrual implied by the ``^IRX`` 13-week bill rate (a PROXY).

    ``^IRX`` is quoted as an annualised *discount* rate in percent on an act/360 basis;
    we accrue ``(irx_{t-1} / 100) / basis`` per trading day. Two approximations are made
    explicit by the ``basis`` argument: the discount-vs-investment-rate convention and
    the 252-vs-360 day count. Both are swept in :func:`cash_proxy_sweep`; neither moves
    a conclusion, because they scale *every* leg's excess return identically.

    The one-day shift is the study's single execution lag: the rate printed at the close
    of day ``t`` is what a holder earns over day ``t+1``.
    """
    return (irx.shift(1) / 100.0) / float(basis)


def excess_frame(returns: pd.DataFrame, cash: pd.Series) -> pd.DataFrame:
    """Subtract the cash accrual from every column, on the common index."""
    idx = returns.index.intersection(cash.dropna().index)
    return returns.loc[idx].sub(cash.loc[idx], axis=0)


# --------------------------------------------------------------------------- #
# Inference primitives (mirror of Studies 803 / 912)
# --------------------------------------------------------------------------- #
def _hac_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = _hac_lags(n)
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def hac_ols(y: np.ndarray, X: np.ndarray, lags: int | None = None,
            return_vcov: bool = False):
    """OLS with Newey-West standard errors. Returns ``(beta, se, tstat)`` arrays.

    With ``return_vcov=True`` the full HAC covariance matrix ``V`` is appended, which is
    what :func:`hac_contrast` needs: the *t* of a linear combination of coefficients
    depends on their covariance, not only on the diagonal.

    Used for the regime interaction: regress a daily return *difference* on rising- and
    falling-rate dummies and read the coefficients off with HAC inference, so serial
    correlation in the difference does not manufacture significance.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(y)
    if lags is None:
        lags = _hac_lags(n)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        A = Xu[l:].T @ Xu[:-l] / n
        S += w * (A + A.T)
    V = n * XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    if return_vcov:
        return beta, se, t, V
    return beta, se, t


def hac_contrast(y: np.ndarray, X: np.ndarray, c: np.ndarray,
                 lags: int | None = None) -> tuple[float, float, float]:
    """HAC inference on a linear combination ``c'beta`` of OLS coefficients.

    Returns ``(value, se, tstat)`` with ``se = sqrt(c' V c)`` from the full Newey-West
    covariance. Reading a difference of two coefficients off their individual standard
    errors — or re-estimating it from a *restricted* regression — is not the same test;
    this is the one that answers "is the gap between these two coefficients zero?".
    """
    c = np.asarray(c, dtype=float)
    beta, _, _, V = hac_ols(y, X, lags=lags, return_vcov=True)
    value = float(c @ beta)
    var = float(c @ V @ c)
    if not np.isfinite(var) or var <= 0:
        return value, float("nan"), float("nan")
    se = float(np.sqrt(var))
    return value, se, float(value / se)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Performance summary
# --------------------------------------------------------------------------- #
def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    if len(r) == 0:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series.

    Pass an *excess-of-cash* series to read the excess Sharpe. ``ann_return`` is the
    arithmetic annualised mean (the quantity the HAC *t* tests); ``cagr`` is the
    geometric one an investor actually compounds.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n == 0:
        return {"n_days": 0, "ann_return": float("nan"), "cagr": float("nan"),
                "sharpe": float("nan"), "vol_ann": float("nan"),
                "max_drawdown": float("nan"), "tstat": float("nan")}
    mu = r.mean()
    std = r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    return {
        "n_days": int(n),
        "ann_return": float(mu * periods_per_year),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth.iloc[-1] > 0 else float("nan"),
        "sharpe": float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan"),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "tstat": newey_west_t(r.to_numpy()),
    }


# --------------------------------------------------------------------------- #
# The pairwise race (the headline)
# --------------------------------------------------------------------------- #
def pair_race(
    returns: pd.DataFrame,
    a: str,
    b: str,
    cost_bps: float = 0.0,
    borrow_bps: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """Race leg ``a`` against leg ``b`` on their daily total-return difference.

    ``cost_bps`` is a one-way spread charged **twice** (in and out) on each leg and
    amortised over the sample — the honest friction for a buy-and-hold sleeve.
    ``borrow_bps`` is an annual borrow fee charged on leg ``b``, applicable only if the
    difference is implemented as a long/short *pair* rather than as a choice between two
    sleeves; it is zero for the sleeve-choice reading and swept for the pair reading.

    Returns the annualised difference, its HAC *t*, the annualised tracking vol, and the
    hit rate of ``a`` over ``b``.
    """
    d = (returns[a] - returns[b]).dropna()
    n = len(d)
    if n < 3:
        return {"pair": f"{a}-{b}", "n_days": n, "ann_diff": float("nan"),
                "tstat": float("nan"), "vol_ann": float("nan"), "hit_rate": float("nan")}
    friction = 0.0
    if cost_bps > 0:
        friction += 4.0 * cost_bps * 1e-4 / (n / periods_per_year)   # in+out on both legs
    if borrow_bps > 0:
        friction += borrow_bps * 1e-4
    ann = float(d.mean() * periods_per_year) - friction
    return {
        "pair": f"{a}-{b}",
        "n_days": int(n),
        "ann_diff": ann,
        "ann_diff_gross": float(d.mean() * periods_per_year),
        "tstat": newey_west_t(d.to_numpy()),
        "vol_ann": float(d.std(ddof=1) * np.sqrt(periods_per_year)),
        "hit_rate": float((d > 0).mean()),
        "friction_ann": friction,
    }


def race_table(returns: pd.DataFrame, pairs=None, **kwargs) -> pd.DataFrame:
    """Run :func:`pair_race` over a list of ``(a, b)`` pairs and tabulate."""
    if pairs is None:
        pairs = [("USFR", "BIL"), ("TFLO", "BIL"), ("USFR", "SHY"),
                 ("TFLO", "SHY"), ("BIL", "SHY"), ("USFR", "TFLO")]
    rows = [pair_race(returns, a, b, **kwargs) for a, b in pairs]
    return pd.DataFrame(rows).set_index("pair")


def level_table(returns: pd.DataFrame, cash: pd.Series | None = None) -> pd.DataFrame:
    """Per-fund summary: total return, excess-of-cash Sharpe, vol and drawdown."""
    rows = {}
    exc = excess_frame(returns, cash) if cash is not None else None
    for c in returns.columns:
        s = summary(returns[c])
        row = {"ann_return": s["ann_return"], "vol_ann": s["vol_ann"],
               "max_drawdown": s["max_drawdown"], "n_days": s["n_days"]}
        if exc is not None:
            e = summary(exc[c])
            row["excess_ann"] = e["ann_return"]
            row["excess_sharpe"] = e["sharpe"]
            row["excess_t"] = e["tstat"]
        rows[c] = row
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Regimes — mechanical (^IRX direction) and calendar (declared assumption)
# --------------------------------------------------------------------------- #
def irx_regime(irx: pd.Series, window: int = 63, thresh: float = 0.25) -> pd.Series:
    """Label each day ``rising`` / ``falling`` / ``flat`` from the ``^IRX`` direction.

    The label on day ``t`` is the sign of the ``window``-day change in the 13-week bill
    rate through ``t``, with a dead band of ``thresh`` percentage points — and it is
    shifted one day, so it describes the world a holder could already see. Both the
    window and the threshold are swept in :func:`regime_param_sweep`.
    """
    chg = irx.diff(window)
    lab = pd.Series(
        np.where(chg > thresh, "rising", np.where(chg < -thresh, "falling", "flat")),
        index=irx.index, name="regime",
    )
    lab[chg.isna()] = np.nan
    return lab.shift(1)


def regime_table(returns: pd.DataFrame, regime: pd.Series, pairs=None) -> pd.DataFrame:
    """Annualised pairwise differences and HAC *t*, split by ``^IRX`` regime."""
    if pairs is None:
        pairs = [("USFR", "BIL"), ("USFR", "SHY"), ("BIL", "SHY")]
    df = returns.join(regime.rename("regime"), how="inner").dropna(subset=["regime"])
    rows = []
    for reg, sub in df.groupby("regime"):
        row = {"regime": reg, "n_days": len(sub)}
        for c in returns.columns:
            row[f"ret_{c}"] = float(sub[c].mean() * TRADING_DAYS * 100)
        for a, b in pairs:
            d = (sub[a] - sub[b]).dropna()
            row[f"{a}-{b}"] = float(d.mean() * TRADING_DAYS * 100)
            row[f"t_{a}-{b}"] = newey_west_t(d.to_numpy())
        rows.append(row)
    return pd.DataFrame(rows).set_index("regime")


def regime_contrast(
    returns: pd.DataFrame,
    a: str,
    b: str,
    regime: pd.Series,
) -> dict:
    """HAC-OLS test of *"does the rate direction flip the ranking of ``a`` and ``b``?"*.

    Regresses the daily difference ``a - b`` (in annualised percentage points) on rising-
    and falling-rate dummies, then reports the **contrast** — the extra annualised
    advantage of ``a`` in a rising-rate world over its advantage in a falling-rate one.
    This is the study's single headline test, formed on one series so it is not a
    fishing expedition across sub-samples.

    The contrast is the linear combination ``rising_extra - falling_extra`` of the
    *unrestricted* three-dummy fit, with its HAC standard error taken from the full
    covariance matrix (:func:`hac_contrast`) — so the number in the ``contrast`` column
    is exactly the difference of the two columns beside it. An earlier version read it off
    a *restricted* regression on ``rising - falling``, which silently borrows the flat
    days to pin the midpoint and, on an unbalanced split (626 rising vs 390 falling days
    here), reported a contrast ~20% larger than the gap it claimed to measure.
    """
    df = returns[[a, b]].join(regime.rename("regime"), how="inner").dropna()
    y = ((df[a] - df[b]) * TRADING_DAYS * 100).to_numpy()
    rise = (df["regime"] == "rising").to_numpy(dtype=float)
    fall = (df["regime"] == "falling").to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), rise, fall])
    beta, se, t = hac_ols(y, X)
    contrast, _, contrast_t = hac_contrast(y, X, np.array([0.0, 1.0, -1.0]))
    return {
        "pair": f"{a}-{b}",
        "n_days": int(len(y)),
        "n_rising": int(rise.sum()),
        "n_falling": int(fall.sum()),
        "flat_diff": float(beta[0]), "flat_t": float(t[0]),
        "rising_extra": float(beta[1]), "rising_t": float(t[1]),
        "falling_extra": float(beta[2]), "falling_t": float(t[2]),
        "contrast": float(contrast),          # rising_extra - falling_extra, exactly
        "contrast_t": float(contrast_t),
    }


def cycle_table(returns: pd.DataFrame, windows=None) -> pd.DataFrame:
    """Per-cycle-window annualised returns, vol, drawdown and pairwise differences.

    The windows are the declared ASSUMPTION ``CYCLE_WINDOWS`` — a hardcoded reading of
    Fed history, reported next to the mechanical ``^IRX``-direction cut so a reader can
    see whether the story survives without the hand-drawn calendar.
    """
    if windows is None:
        windows = CYCLE_WINDOWS
    rows = []
    for label, lo, hi in windows:
        sub = returns.loc[lo:hi].dropna()
        if len(sub) < 20:
            continue
        row = {"window": label, "start": lo, "end": hi, "n_days": len(sub)}
        for c in sub.columns:
            row[f"ret_{c}"] = float(sub[c].mean() * TRADING_DAYS * 100)
            row[f"dd_{c}"] = max_drawdown(sub[c]) * 100
        for a, b in [("USFR", "BIL"), ("USFR", "SHY"), ("BIL", "SHY")]:
            if a in sub.columns and b in sub.columns:
                d = (sub[a] - sub[b]).dropna()
                row[f"{a}-{b}"] = float(d.mean() * TRADING_DAYS * 100)
                row[f"t_{a}-{b}"] = newey_west_t(d.to_numpy())
        rows.append(row)
    return pd.DataFrame(rows).set_index("window")


def drawdown_table(returns: pd.DataFrame, split: str = LIQUIDITY_SPLIT) -> pd.DataFrame:
    """Max drawdown and vol per fund, whole sample and post-``split`` (the liquid era)."""
    rows = {}
    for c in returns.columns:
        full = returns[c].dropna()
        late = full[full.index >= pd.Timestamp(split)]
        rows[c] = {
            "dd_full": max_drawdown(full) * 100,
            "vol_full": float(full.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100),
            "dd_liquid": max_drawdown(late) * 100,
            "vol_liquid": float(late.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100),
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Bootstrap, sweeps
# --------------------------------------------------------------------------- #
def block_bootstrap_ci(
    diff: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 922,
    periods_per_year: int = TRADING_DAYS,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the *annualised mean* of a return difference.

    Blocks of ``block`` consecutive days preserve the strong serial correlation of
    front-end carry differences (a yield gap persists for months at a time), which an
    i.i.d. bootstrap would wish away.
    """
    r = np.asarray(pd.Series(diff).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    point = float(r.mean() * periods_per_year)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[i] = r[idx].mean() * periods_per_year
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean()), "n_obs": int(n),
            "n_boot": int(n_boot), "block": int(block)}


def cost_sweep(returns: pd.DataFrame, a: str, b: str,
               cost_grid=(0.0, 1.0, 2.0, 5.0), horizons_years=(1.0, 3.0)) -> pd.DataFrame:
    """Amortised round-trip spread cost for a *held* sleeve, at several horizons.

    A sleeve-switch pays the spread twice on each leg, once. Over a one-year hold that
    friction is charged in full against the annualised difference; over three years it
    is a third as large. Nothing here is traded daily, so a per-day cost model would be
    a caricature.
    """
    gross = pair_race(returns, a, b)["ann_diff_gross"]
    rows = []
    for c in cost_grid:
        for h in horizons_years:
            rows.append({"cost_bps": c, "horizon_years": h,
                         "friction_ann_pct": 4.0 * c * 1e-4 * 100 / h,
                         "net_ann_pct": (gross - 4.0 * c * 1e-4 / h) * 100})
    return pd.DataFrame(rows)


def borrow_sweep(returns: pd.DataFrame, a: str, b: str,
                 borrow_grid=(0.0, 25.0, 50.0, 100.0)) -> pd.DataFrame:
    """The same difference read as a long/short *pair*, with the short leg paying borrow.

    Treasury ETFs are general collateral and cheap to borrow, but not free; the grid
    spans plausible fees. A difference that only survives at zero borrow is not a trade.
    """
    rows = []
    for bp in borrow_grid:
        res = pair_race(returns, a, b, borrow_bps=bp)
        rows.append({"borrow_bps": bp, "net_ann_pct": res["ann_diff"] * 100,
                     "tstat_gross": res["tstat"]})
    return pd.DataFrame(rows)


def cash_proxy_sweep(returns: pd.DataFrame, irx: pd.Series,
                     bil: pd.Series | None = None) -> pd.DataFrame:
    """Excess-of-cash Sharpe of every fund under alternative cash proxies.

    Three definitions of "cash": ``^IRX`` on a 252-day accrual, ``^IRX`` on 360, and
    (optionally) BIL's own total return — the only *investable* one. A ranking that
    changes with the proxy would be an artefact of the convention.
    """
    rows = []
    variants = [("IRX/252", cash_leg(irx, 252)), ("IRX/360", cash_leg(irx, 360))]
    if bil is not None:
        variants.append(("BIL total return", bil))
    for label, cash in variants:
        exc = excess_frame(returns, cash)
        row = {"cash_proxy": label}
        for c in exc.columns:
            row[f"sharpe_{c}"] = summary(exc[c])["sharpe"]
            row[f"ann_{c}"] = summary(exc[c])["ann_return"] * 100
        rows.append(row)
    return pd.DataFrame(rows).set_index("cash_proxy")


def regime_param_sweep(returns: pd.DataFrame, irx: pd.Series, a: str, b: str,
                       windows=(21, 42, 63, 126), threshs=(0.10, 0.25, 0.50)) -> pd.DataFrame:
    """Sweep the regime classifier's lookback window and dead band.

    The rising-minus-falling contrast should be stable in sign and rough magnitude if it
    is economics rather than a lucky choice of classifier.
    """
    rows = []
    for w in windows:
        for th in threshs:
            reg = irx_regime(irx, window=w, thresh=th)
            res = regime_contrast(returns, a, b, reg)
            rows.append({"window": w, "thresh": th, "n_rising": res["n_rising"],
                         "n_falling": res["n_falling"], "contrast": res["contrast"],
                         "contrast_t": res["contrast_t"]})
    return pd.DataFrame(rows)


def duration_attribution(returns: pd.DataFrame, irx: pd.Series, floater: str = "USFR",
                         fixed: str = "SHY", durations=(1.60, 1.85, 2.10)) -> pd.DataFrame:
    """Is the floater-minus-fixed gap simply *duration x the move in rates*?

    Within each cycle window we compare the realised cumulative gap
    ``floater - fixed`` with the textbook prediction ``D x d(yield)``, where ``D`` is
    the fixed fund's effective duration — an ASSUMPTION (~1.85 years, from the fund's
    published statistics, not from this tape), so it is swept 1.60-2.10.

    The prediction uses ``^IRX``, the *13-week* rate, as a stand-in for the move in the
    fixed fund's own 1-3 year yield. That is deliberately crude and biased: in 2022-2023
    the curve inverted, so the bill rate moved far more than the 1-3y yield and the
    prediction over-states the gap. Reading the two columns side by side is the point —
    the sign and order of magnitude come from duration, the residual is the curve.
    """
    rows = []
    for label, lo, hi in CYCLE_WINDOWS:
        sub = returns.loc[lo:hi].dropna()
        irx_sub = irx.loc[lo:hi].dropna()
        if len(sub) < 20 or len(irx_sub) < 20:
            continue
        d_yield = float(irx_sub.iloc[-1] - irx_sub.iloc[0]) / 100.0
        cum_floater = float((1.0 + sub[floater]).prod() - 1.0)
        cum_fixed = float((1.0 + sub[fixed]).prod() - 1.0)
        row = {"window": label, "d_irx_pp": d_yield * 100,
               f"cum_{floater}_pct": cum_floater * 100,
               f"cum_{fixed}_pct": cum_fixed * 100,
               "realised_gap_pct": (cum_floater - cum_fixed) * 100}
        for D in durations:
            row[f"predicted_D{D:.2f}"] = D * d_yield * 100
        rows.append(row)
    return pd.DataFrame(rows).set_index("window")


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, window: int = 63, thresh: float = 0.25) -> dict:
    """Run the regime contrast on a synthetic front-end panel.

    On the planted world (``signal_strength=1``, the fixed leg has 1.85 years of
    duration) the floater-minus-fixed contrast must be large and positive: the floater
    wins the hikes and loses the cuts. On the null (``signal_strength=0``, the fixed leg
    is a floater in disguise) the same estimator must return ~0 even though rates move
    exactly as much. Proves the machinery is unbiased — it never supports a real-tape
    stamp.
    """
    rets = daily_returns(prices, cols=["frn", "bills", "fixed"])
    reg = irx_regime(prices["IRX"], window=window, thresh=thresh)
    con = regime_contrast(rets, "frn", "fixed", reg)
    race = pair_race(rets, "frn", "fixed")
    return {
        "contrast": con["contrast"],
        "contrast_t": con["contrast_t"],
        "rising_extra": con["rising_extra"],
        "falling_extra": con["falling_extra"],
        "n_rising": con["n_rising"],
        "n_falling": con["n_falling"],
        "ann_diff": race["ann_diff"] * 100,
        "ann_diff_t": race["tstat"],
        "n_days": race["n_days"],
    }
