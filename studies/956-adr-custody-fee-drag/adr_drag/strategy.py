"""Estimator + inference for Study 956 — the ADR custody (depositary) fee drag.

The estimand
------------
An ADR and its home line are claims on the same cash flows, so in dollars their
**total-return** indices should track each other up to a stationary arbitrage band. They
do not: the depositary bank deducts a pass-through custody fee, and the foreign tax
authority withholds tax, before either reaches the ADR holder. Both leaks are a *drift*
in the log ratio

    x_t = log(ADR_total_return_t) - log(home_line_total_return_t x FX_t)

so the estimand is the **slope of x_t in time**, reported as a positive annual cost:

    drag = - d x_t / dt        (per year, fraction of NAV)

Why a trend regression and not a mean of daily differences: the daily difference is
dominated by non-synchronous closes (the home market shut hours before New York), and
that noise is *stationary* — it reverses. Its cumulative contribution therefore does not
grow with the sample, while the fee's does. A trend fit on the level exploits exactly
that; a mean of daily returns throws it away and its standard error is two orders of
magnitude too wide to see a 20 bp/yr fee.

Three pieces the same regression gives us, and one identification trick that FAILS
----------------------------------------------------------------------------------
* ``drag_total``   — the slope on the **total-return** ratio: the whole leak.
* ``price_drift``  — the same slope on the **price-only** ratio. This is a *placebo*: an
  ADS is a fixed number of shares, so the price-only ratio must be flat. A non-zero
  price drift means the ruler is broken (an undetected ADS-ratio change, an FX mismatch),
  not that a fee was found.
* ``income_gap``   — ``drag_total - price_drift``, identically the gap between the home
  line's realised distribution yield and the ADR's. This is the leak proper, and it is
  measured, not assumed. It is a *combined* shortfall: it does not say what the leak is.

Splitting ``income_gap`` into withholding and custody needs one number that is not on the
tape: the withholding rate. That is a labelled **ASSUMPTION** and it is swept.

The identification trick was meant to be the **UK** names — the UK levies no dividend
withholding tax, so for a UK pair the assumption is 0% by law and the income gap would be
the custody fee with nothing else in it. **It does not survive contact with the data.**
All five UK pairs are London listings whose vendor "adjusted close" is split-adjusted only
(no dividends at all), so every one of them is thrown out by :func:`coverage_screen`
before the decomposition runs. Nothing in the surviving ten-name panel identifies the
split, and the sweep in :func:`withholding_sweep` shows why: at treaty rates the tax alone
exceeds the whole measured gap, so the residual "custody" term is negative for every name.
The honest reading of ``income_gap`` is an **upper bound** on the wrapper's income leak,
attributed to nothing in particular.

Breaks
------
ADS ratios change (a 1:2 becomes a 1:1; a home line splits without the ADS following).
Such an event is a one-off *step* in the log ratio, which would masquerade as an enormous
drift. ``segment_index`` cuts the sample at any daily jump larger than ``break_thresh``
and the regression carries a **separate intercept per segment** with time centred inside
it, so a step contributes nothing to the slope. The break threshold is swept.

Execution lag
-------------
The measurement has no traded leg and therefore no lag. The single traded comparison —
``switch_race``, "own the home line instead of the ADR" — is a buy-and-hold switch whose
entry is decided at the close of day ``t`` and executed at day ``t+1``, charged one-way
x NAV, with no short leg (hence no borrow), raced **excess-of-cash** against BIL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Default HAC lag for the trend regression. The log ratio is a highly persistent
# (near-AR(1) with phi close to one) series, so the automatic 4*(n/100)^(2/9) rule is far
# too short; one trading year of lags is the honest default.
HAC_LAGS = 252


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def nw_ols(X: np.ndarray, y: np.ndarray, lags: int = HAC_LAGS) -> tuple[np.ndarray, np.ndarray]:
    """OLS with Newey-West (Bartlett) standard errors. Returns ``(beta, se)``."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (X.T @ y)
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Xu[l:].T @ Xu[:-l]
        S = S + w * (G + G.T)
    V = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return beta, se


def block_bootstrap_ci(values: np.ndarray, n_boot: int, block: int, seed: int,
                       alpha: float = 0.05) -> tuple[float, float]:
    """Circular block-bootstrap percentile CI for the *mean* of ``values``."""
    r = np.asarray(values, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < block + 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        draws[b] = r[idx].mean()
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Log ratios and break segmentation
# --------------------------------------------------------------------------- #
def log_ratio(df: pd.DataFrame, leg: str = "tr") -> pd.Series:
    """``log(ADR) - log(home line x FX)`` for the total-return (``tr``) or price (``px``) leg."""
    if leg == "tr":
        num, den = df["adr_tr"], df["loc_tr_usd"]
    elif leg == "px":
        num, den = df["adr_px"], df["loc_px_usd"]
    else:
        raise ValueError("leg must be 'tr' or 'px'")
    return np.log(num.astype(float)) - np.log(den.astype(float))


def level_shift(x: pd.Series, window: int = 10) -> pd.Series:
    """Median of the next ``window`` days minus median of the previous ``window`` days.

    A *permanent* level shift (an ADS-ratio change) moves this statistic; a one-day
    non-synchronous gap — the ADR closing six hours after Tokyo on a violent day — does
    not, because it reverts inside both medians. Testing the level shift rather than the
    daily change is what stops the detector shattering a fat-tailed but perfectly healthy
    ratio into stubs.
    """
    v = pd.Series(x).astype(float)
    before = v.rolling(window, min_periods=window // 2).median()
    after = v[::-1].rolling(window, min_periods=window // 2).median()[::-1]
    return (after - before.shift(1)).rename("level_shift")


def break_points(x: pd.Series, break_thresh: float = 0.10, window: int = 10) -> list[int]:
    """Positions of detected permanent level shifts, one per cluster (non-max suppressed)."""
    s = level_shift(x, window=window).to_numpy()
    cand = np.where(np.isfinite(s) & (np.abs(s) > break_thresh))[0]
    out: list[int] = []
    for i in cand:
        if out and i - out[-1] <= window:
            if abs(s[i]) > abs(s[out[-1]]):
                out[-1] = i
            continue
        out.append(int(i))
    return out


def segment_index(x: pd.Series, break_thresh: float = 0.10, min_days: int = 250,
                  window: int = 10) -> np.ndarray:
    """Label each observation with a segment id, cutting at every detected level shift.

    An ADS-ratio change, a spin-off, or a home-line split the depositary mirrored on a
    different date all show up as a permanent step in the log ratio, which would
    masquerade as an enormous drift. Runs shorter than ``min_days`` are labelled ``-1``
    and dropped from the regression rather than being fitted on a stub of data.
    """
    seg = np.zeros(len(x), dtype=int)
    for i in break_points(x, break_thresh=break_thresh, window=window):
        seg[i:] += 1
    out = seg.copy()
    for s in np.unique(seg):
        if (seg == s).sum() < min_days:
            out[seg == s] = -1
    return out


def n_breaks(x: pd.Series, break_thresh: float = 0.10, window: int = 10) -> int:
    """Count of detected permanent level shifts in a log-ratio series."""
    return len(break_points(x, break_thresh=break_thresh, window=window))


# --------------------------------------------------------------------------- #
# The trend estimator
# --------------------------------------------------------------------------- #
def trend_drag(
    x: pd.Series,
    break_thresh: float = 0.10,
    min_days: int = 250,
    hac_lags: int = HAC_LAGS,
) -> dict:
    """Annualised drag = minus the common time slope of ``x``, with segment fixed effects.

    Regresses ``x`` on time-in-years (centred *within* each break segment) plus one dummy
    per segment. The slope is common across segments, the level is not, so a one-off ADS
    ratio step contributes nothing. A **positive** returned ``drag`` means the ADR loses
    ground to its home line, i.e. a cost to the holder.
    """
    x = pd.Series(x).dropna()
    if len(x) < min_days + 10:
        return {"drag": float("nan"), "se": float("nan"), "t": float("nan"),
                "n_days": len(x), "n_segments": 0, "years": float("nan")}
    seg = segment_index(x, break_thresh=break_thresh, min_days=min_days)
    keep = seg >= 0
    xs = x[keep]
    segs = seg[keep]
    if len(xs) < min_days + 10:
        return {"drag": float("nan"), "se": float("nan"), "t": float("nan"),
                "n_days": len(xs), "n_segments": 0, "years": float("nan")}

    labels = np.unique(segs)
    t_years = np.zeros(len(xs))
    D = np.zeros((len(xs), len(labels)))
    for j, s in enumerate(labels):
        m = segs == s
        D[m, j] = 1.0
        tt = np.arange(m.sum(), dtype=float) / TRADING_DAYS
        t_years[m] = tt - tt.mean()
    X = np.column_stack([D, t_years])
    beta, se = nw_ols(X, xs.to_numpy(dtype=float), lags=hac_lags)
    slope, slope_se = float(beta[-1]), float(se[-1])
    return {
        "drag": -slope,
        "se": slope_se,
        "t": float(-slope / slope_se) if slope_se > 0 else float("nan"),
        "n_days": int(len(xs)),
        "n_segments": int(len(labels)),
        "years": float(len(xs) / TRADING_DAYS),
    }


def bootstrap_drag_ci(
    x: pd.Series,
    break_thresh: float = 0.10,
    min_days: int = 250,
    n_boot: int = 600,
    block: int = 63,
    seed: int = 956,
    alpha: float = 0.05,
) -> dict:
    """Circular block bootstrap of the trend drag, resampling *within* each segment.

    Blocks of ``block`` consecutive daily changes are resampled and re-cumulated inside
    each break segment (so the segmentation is preserved), and the trend regression is
    refitted on the rebuilt path. Blocks keep the arbitrage band's persistence, which is
    what makes the naive standard error optimistic.
    """
    x = pd.Series(x).dropna()
    seg = segment_index(x, break_thresh=break_thresh, min_days=min_days)
    keep = seg >= 0
    xs, segs = x[keep], seg[keep]
    if len(xs) < min_days + 10:
        return {"drag": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_le_zero": float("nan"), "n_boot": 0}
    point = trend_drag(x, break_thresh, min_days)["drag"]
    labels = np.unique(segs)
    parts = []
    for s in labels:
        v = xs[segs == s].to_numpy(dtype=float)
        parts.append(np.diff(v))
    rng = np.random.default_rng(seed)
    offsets = np.arange(block)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        rebuilt, seg_ids = [], []
        for j, d in enumerate(parts):
            n = d.size
            if n < block + 2:
                continue
            nb = int(np.ceil(n / block))
            starts = rng.integers(0, n, nb)
            idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
            rebuilt.append(np.concatenate([[0.0], np.cumsum(d[idx])]))
            seg_ids.append(np.full(n + 1, j))
        if not rebuilt:
            continue
        y = np.concatenate(rebuilt)
        sid = np.concatenate(seg_ids)
        labs = np.unique(sid)
        tt = np.zeros(len(y))
        D = np.zeros((len(y), len(labs)))
        for j, s in enumerate(labs):
            m = sid == s
            D[m, j] = 1.0
            u = np.arange(m.sum(), dtype=float) / TRADING_DAYS
            tt[m] = u - u.mean()
        X = np.column_stack([D, tt])
        bhat = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        draws[b] = -float(bhat[-1])
    valid = draws[np.isfinite(draws)]
    if valid.size == 0:
        return {"drag": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_le_zero": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"drag": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_le_zero": float((valid <= 0).mean()), "n_boot": int(valid.size),
            "block": block}


# --------------------------------------------------------------------------- #
# Per-name decomposition
# --------------------------------------------------------------------------- #
def income_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """The two cumulative distribution legs and their difference, all in log points.

    ``log(TR) - log(price-only)`` is the cumulative reinvested distribution of a line;
    its slope is that line's realised yield. FX drops out of both legs (it multiplies the
    total-return and the price-only close identically), so the income comparison is
    **currency-free** — a real advantage over comparing the price ratios.
    """
    gross = np.log(df["loc_tr"].astype(float)) - np.log(df["loc_px"].astype(float))
    net = np.log(df["adr_tr"].astype(float)) - np.log(df["adr_px"].astype(float))
    return gross, net, (net - gross).rename("income_gap")


def coverage_screen(df: pd.DataFrame, min_local_yield: float = 0.005) -> dict:
    """Data-integrity gate: does the home line's adjusted close carry dividends at all?

    Yahoo's adjusted close is dividend-adjusted for most venues but **split-only for the
    London Stock Exchange**, so an LSE home leg silently reports a ~0 %/yr distribution
    yield against an ADR paying 4-5 %. Comparing the two manufactures a 5 %/yr "custody
    fee" — a hundred times any real depositary charge.

    The gate depends only on the *home* line's own realised yield, never on the ADR's or
    on the gap between them, so it cannot select for a small or a large answer. Names that
    fail are reported, not silently dropped.
    """
    gross, net, _ = income_series(df)
    yrs = max(len(df) / TRADING_DAYS, 1e-9)
    gy = float(gross.iloc[-1] - gross.iloc[0]) / yrs
    ny = float(net.iloc[-1] - net.iloc[0]) / yrs
    return {
        "local_yield": gy, "adr_yield": ny,
        "yield_ratio": float(ny / gy) if gy > 0 else float("nan"),
        "pass": bool(gy >= min_local_yield),
        "min_local_yield": min_local_yield,
    }


def screen_frames(frames: dict, min_local_yield: float = 0.005) -> tuple[dict, pd.DataFrame]:
    """Apply :func:`coverage_screen` to a dict of frames. Returns ``(kept, report)``."""
    rows = {tk: coverage_screen(df, min_local_yield) for tk, df in frames.items()}
    report = pd.DataFrame(rows).T
    kept = {tk: frames[tk] for tk in frames if rows[tk]["pass"]}
    return kept, report


def decompose_pair(
    df: pd.DataFrame,
    wht: float,
    break_thresh: float = 0.10,
    min_days: int = 250,
    hac_lags: int = HAC_LAGS,
) -> dict:
    """Full decomposition for one ADR / home-line pair.

    Returns annualised, NAV-fraction quantities:

    ``drag_total``   total-return ratio slope (the whole leak, positive = ADR loses);
    ``price_drift``  price-only ratio slope (the **placebo**: must be ~0);
    ``income_gap``   ``drag_total - price_drift`` — the measured gross-minus-net yield gap;
    ``gross_yield``  the home line's realised distribution yield (TR minus price-only);
    ``net_yield``    the ADR's realised distribution yield (TR minus price-only);
    ``wht_cost``     ``wht x gross_yield`` — the **ASSUMED** withholding share;
    ``custody``      ``income_gap - wht_cost`` — the residual custody-fee estimate;
    ``custody_cents`` the same, expressed in **cents per ADS per year** at the mean ADR
    price, which is the unit the depositary publishes its fee schedule in.

    A warning on ``custody`` and ``custody_cents``: on the real tape they come out
    **negative for every name** at any positive ``wht``, because the vendor's per-ADS
    dividend is the gross declared amount and the withholding is simply not in the series.
    Read them as a falsification of the decomposition, not as a fee. The defensible
    quantity is ``income_gap`` itself, and even that is an *upper bound* on the depositary
    fee rather than a measurement of it — a fee billed as a separate DTC/broker line item
    (which is how much of the schedule is collected) never touches this tape either.
    """
    x_tr = log_ratio(df, "tr")
    x_px = log_ratio(df, "px")
    tot = trend_drag(x_tr, break_thresh, min_days, hac_lags)
    pxd = trend_drag(x_px, break_thresh, min_days, hac_lags)

    gross_s, net_s, gap_s = income_series(df)
    # The income legs step on ex-dates; a step is the signal here, not a break, so the
    # break detector is disabled on them (threshold 1.0 log point is never reached).
    gross = trend_drag(gross_s, break_thresh=1.0, min_days=min_days, hac_lags=hac_lags)
    net = trend_drag(net_s, break_thresh=1.0, min_days=min_days, hac_lags=hac_lags)
    gap = trend_drag(gap_s, break_thresh=1.0, min_days=min_days, hac_lags=hac_lags)
    gross_yield = -gross["drag"]   # a yield is a *rise* of TR over price, so flip the sign
    net_yield = -net["drag"]

    income_gap = gap["drag"]       # positive = the ADR's realised yield is the smaller one
    wht_cost = wht * gross_yield
    custody = income_gap - wht_cost
    mean_px = float(df["adr_px"].astype(float).mean())

    return {
        "drag_total": tot["drag"], "drag_se": tot["se"], "drag_t": tot["t"],
        "n_days": tot["n_days"], "years": tot["years"], "n_segments": tot["n_segments"],
        "n_breaks": n_breaks(x_tr, break_thresh),
        "price_drift": pxd["drag"], "price_drift_t": pxd["t"],
        "income_gap": income_gap, "income_gap_t": gap["t"], "income_gap_se": gap["se"],
        "gross_yield": gross_yield, "net_yield": net_yield,
        "wht_assumed": wht, "wht_cost": wht_cost,
        "custody": custody,
        "income_gap_cents": income_gap * mean_px * 100.0,
        "custody_cents": custody * mean_px * 100.0,
        "mean_adr_price": mean_px,
        "start": df.index[0], "end": df.index[-1],
    }


def panel_table(frames: dict, whts: dict, break_thresh: float = 0.10,
                min_days: int = 250, hac_lags: int = HAC_LAGS) -> pd.DataFrame:
    """Run :func:`decompose_pair` over a dict of ``{ticker: frame}`` into one table."""
    rows = {}
    for tk, df in frames.items():
        rows[tk] = decompose_pair(df, whts.get(tk, 0.0), break_thresh, min_days, hac_lags)
    out = pd.DataFrame(rows).T
    for c in out.columns:
        if c not in ("start", "end"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def pooled(table: pd.DataFrame, col: str = "drag_total", subset=None) -> dict:
    """Cross-name mean of ``col`` with a plain cross-sectional t (names as observations).

    Names are treated as independent draws — defensible here because the estimation error
    is dominated by each pair's own idiosyncratic arbitrage band, but the *fee* itself is
    a common institutional parameter, so the cross-name t is a test of "is the average
    leak non-zero", not of any one name.
    """
    v = table[col] if subset is None else table.loc[list(subset), col]
    v = pd.to_numeric(v, errors="coerce").dropna().to_numpy(dtype=float)
    n = v.size
    if n < 2:
        return {"mean": float("nan"), "t": float("nan"), "n": n,
                "sd": float("nan"), "share_positive": float("nan")}
    sd = v.std(ddof=1)
    return {"mean": float(v.mean()), "sd": float(sd), "n": n,
            "t": float(v.mean() / (sd / np.sqrt(n))) if sd > 0 else float("nan"),
            "share_positive": float((v > 0).mean()),
            "median": float(np.median(v))}


def name_bootstrap(table: pd.DataFrame, col: str = "income_gap", n_boot: int = 5000,
                   seed: int = 956, alpha: float = 0.05) -> dict:
    """Resample *names* with replacement for a CI on the cross-name mean.

    The per-name estimates are each a 20-year trend fit; what a reader wants to know is
    whether the *average* leak across issuers is above zero, so the resampling unit is the
    issuer. A one-sided sign test is reported alongside, because it survives the one or
    two names whose spin-off history makes their point estimate an outlier.
    """
    v = pd.to_numeric(table[col], errors="coerce").dropna().to_numpy(dtype=float)
    n = v.size
    if n < 3:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n": n, "n_positive": 0, "sign_p": float("nan")}
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    k = int((v > 0).sum())
    # Exact one-sided binomial tail under p = 1/2.
    tail = sum(_choose(n, j) for j in range(k, n + 1)) / (2.0 ** n)
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "ci_low": float(lo), "ci_high": float(hi), "n": n,
            "n_positive": k, "sign_p": float(tail),
            "frac_le_zero": float((draws <= 0).mean())}


def _choose(n: int, k: int) -> float:
    out = 1.0
    for i in range(k):
        out = out * (n - i) / (i + 1)
    return out


def leave_one_out(table: pd.DataFrame, col: str = "income_gap") -> pd.DataFrame:
    """Recompute the cross-name mean and t dropping each name in turn."""
    rows = []
    for tk in table.index:
        sub = table.drop(index=tk)
        p = pooled(sub, col)
        rows.append({"dropped": tk, "mean": p["mean"], "t": p["t"],
                     "median": p["median"], "n": p["n"]})
    return pd.DataFrame(rows).set_index("dropped")


# --------------------------------------------------------------------------- #
# Robustness: assumption sweeps and an era cut
# --------------------------------------------------------------------------- #
def withholding_sweep(frames: dict, whts: dict, scales=(0.0, 0.5, 1.0, 1.5),
                      break_thresh: float = 0.10) -> list[dict]:
    """Sweep the **assumed** withholding rate (the only non-tape input) by a scale factor.

    ``scale=0`` credits nothing to tax (every cent of the income gap becomes "custody" —
    the most generous reading of the fee); ``scale=1`` uses the treaty rates in
    ``data.PAIRS``; ``scale=1.5`` over-credits tax (roughly the statutory, pre-treaty
    rates for Germany and Denmark).

    The UK names would have been the anchor (0% by law at every scale), but all five fail
    the coverage screen and never reach this function — see the module docstring. On the
    surviving panel the sweep is therefore not a decomposition but a **falsification**:
    every scale above zero drives the residual negative, which is how the study learns
    that the withholding is not in the vendor's total-return series at all.
    """
    out = []
    for s in scales:
        scaled = {k: v * s for k, v in whts.items()}
        tbl = panel_table(frames, scaled, break_thresh=break_thresh)
        p = pooled(tbl, "custody")
        out.append({"scale": s, "custody_mean": p["mean"], "custody_t": p["t"],
                    "custody_median": p["median"], "share_positive": p["share_positive"]})
    return out


def break_threshold_sweep(frames: dict, whts: dict,
                          thresholds=(0.06, 0.10, 0.15, 0.25)) -> list[dict]:
    """Sweep the break-detection threshold: does the answer depend on where we cut?"""
    out = []
    for th in thresholds:
        tbl = panel_table(frames, whts, break_thresh=th)
        pt = pooled(tbl, "drag_total")
        pp = pooled(tbl, "price_drift")
        out.append({"thresh": th, "drag_mean": pt["mean"], "drag_t": pt["t"],
                    "price_drift_mean": pp["mean"],
                    "n_segments": float(pd.to_numeric(tbl["n_segments"]).mean())})
    return out


def era_cut(frames: dict, whts: dict, split: str = "2016-01-01",
            break_thresh: float = 0.10, min_days: int = 250) -> dict:
    """Re-run the panel on each half of the sample."""
    out = {}
    for tag, sl in (("early", slice(None, split)), ("late", slice(split, None))):
        sub = {k: v.loc[sl] for k, v in frames.items()}
        sub = {k: v for k, v in sub.items() if len(v) >= min_days + 10}
        if not sub:
            out[tag] = None
            continue
        tbl = panel_table(sub, whts, break_thresh=break_thresh, min_days=min_days)
        out[tag] = {
            "n_names": int(len(tbl)),
            "start": min(v.index[0] for v in sub.values()),
            "end": max(v.index[-1] for v in sub.values()),
            "drag": pooled(tbl, "drag_total"),
            "income_gap": pooled(tbl, "income_gap"),
            "custody": pooled(tbl, "custody"),
            "price_drift": pooled(tbl, "price_drift"),
        }
    return out


# --------------------------------------------------------------------------- #
# The only traded leg — "own the home line instead"
# --------------------------------------------------------------------------- #
def switch_race(
    frames: dict,
    cash: pd.Series,
    fx_cost_bps: float = 30.0,
    foreign_custody_bps_per_year: float = 0.0,
    lag: int = 1,
) -> dict:
    """Race an equal-weight ADR basket against the same basket bought on its home markets.

    Both baskets are daily-rebalanced equal weight, measured **excess-of-cash** (minus
    BIL's total return), long only (no short leg, hence no borrow). The home-line basket
    pays a **one-way** FX-conversion cost x NAV at entry — the switch is decided at the
    close of day ``t`` and executed at ``t + lag`` — plus an optional ongoing foreign
    custody / safekeeping charge, both **ASSUMPTIONS** and both swept.

    Returns the two excess-of-cash Sharpes, the advantage, and the HAC *t* on the daily
    return difference.

    TWO CAVEATS, both against this leg being a live backtest:

    * **It inherits a look-ahead filter.** ``frames`` normally arrive from
      :func:`data.load_pair`, whose bad-print screen uses a *centred* rolling median and
      therefore peeks a few days forward. That is harmless for the headline estimand (the
      income gap is identical with the filter off) but it is *not* harmless here: the
      filter removes ~0.03 % of rows and moves this race by hundreds of basis points,
      because a handful of vendor bad prints in the home/FX legs dominate an arithmetic
      mean of daily returns. Read the result as a *measurement* of the wrapper's cost,
      not as a strategy you could have run.
    * **Rebalancing turnover is charged on neither leg** — only the single entry. A
      genuinely daily-rebalanced home-line basket would pay an FX conversion on every
      trade, which the ADR basket would not. The home leg's advantage here is therefore
      an upper bound.
    """
    idx = None
    for df in frames.values():
        i = df.index
        idx = i if idx is None else idx.intersection(i)
    idx = idx.intersection(cash.dropna().index).sort_values()

    r_adr, r_loc = [], []
    for df in frames.values():
        d = df.reindex(idx)
        r_adr.append(d["adr_tr"].astype(float).pct_change())
        r_loc.append(d["loc_tr_usd"].astype(float).pct_change())
    a = pd.concat(r_adr, axis=1).mean(axis=1)
    l = pd.concat(r_loc, axis=1).mean(axis=1)
    c = cash.reindex(idx).astype(float).pct_change()

    # One-way entry cost charged on the executed day (signal at t, fill at t+lag).
    entry = pd.Series(0.0, index=idx)
    if len(idx) > lag:
        entry.iloc[lag] = fx_cost_bps * 1e-4
    ongoing = foreign_custody_bps_per_year * 1e-4 / TRADING_DAYS
    l_net = l - entry - ongoing

    e_adr = (a - c).dropna()
    e_loc = (l_net - c).dropna()
    common = e_adr.index.intersection(e_loc.index)
    e_adr, e_loc = e_adr.loc[common], e_loc.loc[common]

    def sharpe(r):
        sd = r.std(ddof=1)
        return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")

    diff = (e_loc - e_adr).dropna()
    lags = int(np.floor(4.0 * (len(diff) / 100.0) ** (2.0 / 9.0)))
    return {
        "n_days": int(len(common)),
        "sharpe_adr": sharpe(e_adr), "sharpe_local": sharpe(e_loc),
        "sharpe_adv": sharpe(e_loc) - sharpe(e_adr),
        "ann_diff": float(diff.mean() * TRADING_DAYS),
        "t_diff": newey_west_t(diff.to_numpy(), lags=lags),
        "fx_cost_bps": fx_cost_bps,
        "foreign_custody_bps_per_year": foreign_custody_bps_per_year,
    }


def switch_cost_sweep(frames: dict, cash: pd.Series,
                      grid=((0.0, 0.0), (30.0, 0.0), (30.0, 15.0), (50.0, 30.0))) -> list[dict]:
    """Sweep the two assumed frictions of owning the home line instead of the ADR."""
    return [switch_race(frames, cash, fx_cost_bps=f, foreign_custody_bps_per_year=k)
            for f, k in grid]


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(frames: dict, whts: dict, break_thresh: float = 0.10) -> dict:
    """Run the whole estimator on a synthetic panel with a **known** planted drag."""
    tbl = panel_table(frames, whts, break_thresh=break_thresh)
    return {
        "drag": pooled(tbl, "drag_total"),
        "income_gap": pooled(tbl, "income_gap"),
        "custody": pooled(tbl, "custody"),
        "price_drift": pooled(tbl, "price_drift"),
        "table": tbl,
    }
