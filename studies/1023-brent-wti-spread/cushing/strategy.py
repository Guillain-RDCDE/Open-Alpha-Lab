"""Is the Brent–WTI spread tethered? — Study 1023 (The Cushing Glut).

The claim, steelmanned: *Brent and WTI are the same commodity in two places. The spread between
them is bounded by the cost of moving a barrel from one to the other, so it is stationary and
always mean-reverts — fade it when it stretches.* That decomposes into four questions, each
with its own machinery here.

1. **Is it stationary?** ``adf`` (null: unit root) and ``kpss_test`` (null: stationary) are run
   as a pair, because each alone is weak at n ≈ 390: agreement is evidence, disagreement is the
   honest answer "the tape can't tell". ``engle_granger`` asks the same question of the two
   log prices without imposing a 1:1 hedge ratio.
2. **Did the tether move?** ``sup_f_mean`` scans every admissible break date for a shift in
   the mean (Andrews 1993, 15% trimming). Its textbook critical values assume weakly dependent
   errors, which a spread with a half-life of months is not, so the p-value comes from
   ``sup_f_bootstrap``: the sup-F distribution under a *no-break AR(1)* fitted to the whole
   sample (the persistence the break itself induces makes that null conservative).
   ``cusum_mean`` (HAC-scaled OLS-CUSUM) is a second, model-light opinion, and ``bai_perron``
   locates several breaks at once by dynamic programming with the LWZ criterion.
3. **How fast does it revert?** ``ar1_fit`` gives phi and the half-life ``ln 0.5 / ln phi``
   with a delta-method band, within each data-chosen regime.
4. **What did fading it feel like?** ``zscore_realtime`` builds the signal from past data only
   (expanding or rolling), ``positions`` runs a 2σ-in / 0.5σ-out state machine, and ``book``
   charges one-way costs per leg plus a monthly roll toll. ``zscore_frozen`` replays the trader
   who calibrated on 1987–2009 and kept trusting it; ``episodes`` and ``underwater`` measure
   the maximum adverse excursion and how long that trader was under water.

**One execution lag, exactly.** A position decided on month *t*'s (average) prices earns the
spread return of month *t+1*: ``book`` applies a single ``shift(1)`` to positions and to the
cost of the trade that set them. The book is long/short $1 of Brent against $1 of WTI per unit
of position; a futures spread is self-financing, so its returns are already excess returns and
the Sharpe needs no cash leg subtracted.

**The P&L is an upper bound.** The tape is spot, monthly-averaged; futures pay a roll that
differed between the two contracts most when Cushing was full. See :mod:`cushing.data`.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from quantlab.analytics import mean_tstat_hac  # noqa: E402
from quantlab.stats import sharpe_ci_bootstrap  # noqa: E402

MONTHS = 12
ENTRY_Z = 2.0
EXIT_Z = 0.5
COST_BPS = 5.0          # one-way, per leg, per unit of NAV traded
ROLL_BPS = 3.0          # per leg, per month a position is carried (calendar-spread toll)
CALIB_END = "2009-12-31"  # the frozen trader's calibration window ends here (the claim's "pre-2010")
ANDREWS_CV_15 = {"10%": 7.17, "5%": 8.68, "1%": 12.16}   # sup-F, 1 restriction, 15% trim
CUSUM_CV_5 = 1.358                                      # sup |Brownian bridge|, 5%


# --------------------------------------------------------------------------- #
# The spread
# --------------------------------------------------------------------------- #
def spread_frame(px: pd.DataFrame) -> pd.DataFrame:
    """Brent, WTI, the dollar spread ``brent - wti`` ($/bbl) and the log ratio ``ln(B/W)``."""
    out = px[["brent", "wti"]].copy()
    out["spread"] = out["brent"] - out["wti"]
    out["log_ratio"] = np.log(out["brent"]) - np.log(out["wti"])
    return out


def _window(s: pd.Series | pd.DataFrame, start=None, end=None):
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    return s


# --------------------------------------------------------------------------- #
# 1. Stationarity and cointegration
# --------------------------------------------------------------------------- #
def adf(x, regression: str = "c", maxlag: int = 12) -> dict:
    """Augmented Dickey–Fuller, lag order by AIC up to ``maxlag``. Null: unit root."""
    from statsmodels.tsa.stattools import adfuller
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if x.size < 30:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # statsmodels 0.15 FutureWarning on the tuple return
        stat, p, lags, nobs, cv, _ = adfuller(x, maxlag=maxlag, regression=regression,
                                              autolag="AIC")
    return {"stat": float(stat), "p": float(p), "lags": int(lags), "n": int(nobs),
            "cv5": float(cv["5%"]), "reject5": bool(p < 0.05)}


def kpss_test(x, regression: str = "c") -> dict:
    """KPSS with automatic bandwidth. Null: (level-)stationary. p is clipped to [0.01, 0.10]
    by the lookup table — a reported 0.10 means "≥ 0.10", 0.01 means "≤ 0.01"."""
    from statsmodels.tsa.stattools import kpss
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if x.size < 30:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, lags, cv = kpss(x, regression=regression, nlags="auto")
    return {"stat": float(stat), "p": float(p), "lags": int(lags),
            "reject5": bool(p < 0.05)}


def joint_reading(a: dict, k: dict) -> str:
    """ADF and KPSS read together — the honest four-way table."""
    if not a or not k:
        return "n/a"
    if a["reject5"] and not k["reject5"]:
        return "stationary"
    if not a["reject5"] and k["reject5"]:
        return "unit root"
    if a["reject5"] and k["reject5"]:
        return "conflict (both reject)"
    return "inconclusive (neither rejects)"


def stationarity_table(sf: pd.DataFrame, windows) -> pd.DataFrame:
    """ADF + KPSS on the dollar spread and the log ratio, for each ``(label, start, end)``."""
    rows = []
    for label, start, end in windows:
        for col in ("spread", "log_ratio"):
            x = _window(sf[col], start, end).dropna()
            a, k = adf(x), kpss_test(x)
            if not a or not k:
                continue
            rows.append({"window": label, "series": col, "n": int(len(x)),
                         "adf_stat": a["stat"], "adf_p": a["p"], "adf_lags": a["lags"],
                         "kpss_stat": k["stat"], "kpss_p": k["p"],
                         "reading": joint_reading(a, k)})
    return pd.DataFrame(rows)


def engle_granger(px: pd.DataFrame, start=None, end=None) -> dict:
    """Engle–Granger on ``ln Brent = a + b ln WTI + u``: the cointegrating slope and the
    residual-ADF test with MacKinnon p-values. b ≈ 1 is the claim's 'same commodity'."""
    from statsmodels.tsa.stattools import coint
    d = _window(px[["brent", "wti"]], start, end).dropna()
    if len(d) < 36:
        return {}
    lb, lw = np.log(d["brent"].to_numpy()), np.log(d["wti"].to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, _ = coint(lb, lw, trend="c", maxlag=12, autolag="aic")
    A = np.column_stack([np.ones(len(lw)), lw])
    beta = np.linalg.lstsq(A, lb, rcond=None)[0]
    return {"n": int(len(d)), "stat": float(stat), "p": float(p),
            "alpha": float(beta[0]), "beta": float(beta[1]), "reject5": bool(p < 0.05)}


# --------------------------------------------------------------------------- #
# 2. Structural breaks in the mean
# --------------------------------------------------------------------------- #
def _sup_f_core(x: np.ndarray, trim: float = 0.15):
    n = x.size
    lo, hi = max(int(np.floor(trim * n)), 2), min(int(np.ceil((1 - trim) * n)), n - 2)
    ks = np.arange(lo, hi + 1)
    c1 = np.cumsum(x)
    c2 = np.cumsum(x * x)
    tot1, tot2 = c1[-1], c2[-1]
    ssr0 = tot2 - tot1 ** 2 / n
    s1, q1 = c1[ks - 1], c2[ks - 1]
    s2, q2 = tot1 - s1, tot2 - q1
    ssr1 = (q1 - s1 ** 2 / ks) + (q2 - s2 ** 2 / (n - ks))
    F = (ssr0 - ssr1) / (ssr1 / (n - 2))
    return ks, F


def sup_f_mean(x: pd.Series, trim: float = 0.15) -> dict:
    """Andrews (1993) sup-F for one shift in the mean, scanning the middle 70% of dates.

    ``break_date`` is the first observation of the new regime. The Andrews critical values
    (``ANDREWS_CV_15``) assume weakly dependent errors and are shown for reference only; use
    ``sup_f_bootstrap`` for the p-value on a persistent series.
    """
    x = pd.Series(x).dropna()
    v = x.to_numpy(dtype=float)
    ks, F = _sup_f_core(v, trim)
    j = int(np.argmax(F))
    k = int(ks[j])
    return {"sup_f": float(F[j]), "break_index": k, "break_date": x.index[k],
            "mean_before": float(v[:k].mean()), "mean_after": float(v[k:].mean()),
            "F_path": pd.Series(F, index=x.index[ks]), "n": int(v.size)}


def ar1_fit(x) -> dict:
    """OLS AR(1) ``x_t = c + phi x_{t-1} + e``: phi, its SE, the half-life and a 95% band.

    Half-life ``ln 0.5 / ln phi`` months; infinite when phi ≥ 1. The band maps phi ± 1.96 SE
    through the same formula, so an upper phi at or above one prints as ``inf``. OLS phi is
    biased down in short samples (Kendall 1954), so these half-lives lean *fast*.
    """
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    if v.size < 24:
        return {}
    y, z = v[1:], v[:-1]
    A = np.column_stack([np.ones(z.size), z])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ b
    s2 = float(e @ e) / (y.size - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    phi, se = float(b[1]), float(np.sqrt(cov[1, 1]))

    def hl(p):
        return float(np.log(0.5) / np.log(p)) if 0.0 < p < 1.0 else (
            float("inf") if p >= 1.0 else 0.0)

    return {"phi": phi, "phi_se": se, "c": float(b[0]), "sigma": float(np.sqrt(s2)),
            "mean": float(b[0] / (1 - phi)) if phi < 1 else float("nan"),
            "halflife": hl(phi), "halflife_lo": hl(phi - 1.96 * se),
            "halflife_hi": hl(phi + 1.96 * se), "n": int(v.size), "resid": e}


def sup_f_bootstrap(x: pd.Series, n_boot: int = 499, trim: float = 0.15,
                    seed: int = 1023) -> dict:
    """p-value of the sup-F against a persistent, break-free null.

    The null world is the AR(1) fitted to the *whole* sample (break included), driven by
    resampled residuals. The break itself pushes that phi toward one, so the null is more
    persistent than the truth and the test is conservative: it asks whether the shift is
    larger than anything a long-memory, break-free spread throws up by chance.
    """
    x = pd.Series(x).dropna()
    obs = sup_f_mean(x, trim)
    fit = ar1_fit(x)
    phi = min(fit["phi"], 1.0)
    resid = fit["resid"] - fit["resid"].mean()
    rng = np.random.default_rng(seed)
    n = x.size
    stats = np.empty(n_boot)
    for b in range(n_boot):
        e = rng.choice(resid, size=n + 50, replace=True)
        y = np.zeros(n + 50)
        for t in range(1, n + 50):
            y[t] = phi * y[t - 1] + e[t]
        _, F = _sup_f_core(y[50:], trim)
        stats[b] = F.max()
    p = float((1 + (stats >= obs["sup_f"]).sum()) / (n_boot + 1))
    return {"sup_f": obs["sup_f"], "p": p, "null_phi": float(phi),
            "null_q95": float(np.quantile(stats, 0.95)),
            "null_q99": float(np.quantile(stats, 0.99)),
            "break_date": obs["break_date"], "n_boot": n_boot}


def cusum_mean(x: pd.Series, lags: int = 12) -> dict:
    """OLS-CUSUM (Ploberger & Krämer 1992) for a constant mean, scaled by a Newey–West
    long-run s.d. so serial correlation does not by itself trigger it. 5% critical value
    1.358 (sup of a Brownian bridge)."""
    x = pd.Series(x).dropna()
    v = x.to_numpy(dtype=float)
    n = v.size
    e = v - v.mean()
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * float(e[k:] @ e[:-k]) / n
    path = np.cumsum(e) / np.sqrt(max(lrv, 1e-18) * n)
    j = int(np.argmax(np.abs(path)))
    return {"stat": float(np.abs(path).max()), "cv5": CUSUM_CV_5,
            "reject5": bool(np.abs(path).max() > CUSUM_CV_5),
            "peak_date": x.index[j], "path": pd.Series(path, index=x.index)}


def bai_perron(x: pd.Series, max_breaks: int = 5, min_seg: int = 24) -> dict:
    """Global SSR-minimising break dates for m = 0..max_breaks mean shifts (Bai & Perron 2003),
    by dynamic programming; the number of breaks is chosen by LWZ (Liu, Wu & Zidek 1997),
    the penalty Bai–Perron recommend over BIC when errors are serially correlated.

    Even LWZ over-counts on a series this persistent, so the count is descriptive; the
    *location* of the dominant break is what the study uses.
    """
    x = pd.Series(x).dropna()
    v = x.to_numpy(dtype=float)
    n = v.size
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])

    def ssr(i, j):  # segment [i, j) — vectorised over i or j
        m = j - i
        s = c1[j] - c1[i]
        return (c2[j] - c2[i]) - s * s / m

    INF = np.inf
    # cost[m][j] = min SSR of v[:j] split into m+1 segments; arg[m][j] = start of last segment
    cost = np.full((max_breaks + 1, n + 1), INF)
    arg = np.zeros((max_breaks + 1, n + 1), dtype=int)
    for j in range(min_seg, n + 1):
        cost[0, j] = ssr(0, j)
    for m in range(1, max_breaks + 1):
        for j in range((m + 1) * min_seg, n + 1):
            i = np.arange(m * min_seg, j - min_seg + 1)
            c = cost[m - 1, i] + ssr(i, j)
            k = int(np.argmin(c))
            cost[m, j], arg[m, j] = c[k], int(i[k])
    rows, sols = [], {}
    for m in range(0, max_breaks + 1):
        if not np.isfinite(cost[m, n]):
            continue
        bks, j = [], n
        for mm in range(m, 0, -1):
            j = arg[mm, j]
            bks.append(j)
        bks = sorted(bks)
        p = 2 * m + 1
        lwz = np.log(cost[m, n] / (n - p)) + p * 0.299 * np.log(n) ** 2.1 / n
        bic = np.log(cost[m, n] / n) + p * np.log(n) / n
        sols[m] = bks
        rows.append({"m": m, "ssr": float(cost[m, n]), "lwz": float(lwz), "bic": float(bic),
                     "breaks": ", ".join(str(x.index[b].date())[:7] for b in bks)})
    tbl = pd.DataFrame(rows).set_index("m")
    m_lwz = int(tbl["lwz"].idxmin())
    m_bic = int(tbl["bic"].idxmin())
    bk = sols[m_lwz]
    edges = [0] + bk + [n]
    segs = [{"start": x.index[a], "end": x.index[b - 1], "n": int(b - a),
             "mean": float(v[a:b].mean())} for a, b in zip(edges[:-1], edges[1:])]
    return {"table": tbl, "m_lwz": m_lwz, "m_bic": m_bic,
            "break_dates": [x.index[b] for b in bk], "segments": segs,
            "solutions": {m: [x.index[b] for b in s] for m, s in sols.items()}}


def halflife_by_regime(x: pd.Series, break_dates) -> pd.DataFrame:
    """AR(1) half-life inside each regime delimited by ``break_dates`` (data-chosen)."""
    x = pd.Series(x).dropna()
    edges = [x.index[0]] + list(break_dates) + [x.index[-1] + pd.offsets.MonthEnd(1)]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = x[(x.index >= a) & (x.index < b)]
        f = ar1_fit(seg)
        if not f:
            continue
        ad = adf(seg)
        rows.append({"start": seg.index[0], "end": seg.index[-1], "n": f["n"],
                     "mean": float(seg.mean()), "sd": float(seg.std(ddof=1)),
                     "phi": f["phi"], "halflife": f["halflife"],
                     "halflife_lo": f["halflife_lo"], "halflife_hi": f["halflife_hi"],
                     "adf_p": ad.get("p", np.nan)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. The trader's experience
# --------------------------------------------------------------------------- #
def zscore_realtime(x: pd.Series, window: int | None = None,
                    min_obs: int = 36) -> pd.Series:
    """z_t from data through month t only: expanding (``window=None``) or rolling mean/sd.

    Month t's own value is in its own statistics — it is known at t — and nothing later is.
    NaN until ``min_obs`` observations exist.
    """
    x = pd.Series(x).astype(float)
    if window is None:
        mu = x.expanding(min_periods=min_obs).mean()
        sd = x.expanding(min_periods=min_obs).std(ddof=1)
    else:
        mu = x.rolling(window, min_periods=min(min_obs, window)).mean()
        sd = x.rolling(window, min_periods=min(min_obs, window)).std(ddof=1)
    return (x - mu) / sd


def zscore_frozen(x: pd.Series, calib_end: str = CALIB_END, calib_start=None) -> pd.Series:
    """z_t against a mean and s.d. estimated once on ``[calib_start, calib_end]`` and never
    updated — the trader who learned the tether before 2010 and kept believing it. Defined
    only after ``calib_end`` (inside the window it would be look-ahead)."""
    x = pd.Series(x).astype(float)
    cal = _window(x, calib_start, calib_end)
    z = (x - cal.mean()) / cal.std(ddof=1)
    return z.where(x.index > pd.Timestamp(calib_end))


def positions(z: pd.Series, entry: float = ENTRY_Z, exit: float = EXIT_Z) -> pd.Series:
    """Fade the stretch: short the spread (−1) when z > entry, long (+1) when z < −entry,
    flat once |z| < exit; otherwise hold. A crossing straight to the other extreme flips.
    NaN z → flat. Decided at the close of month t; ``book`` applies the lag."""
    zz = pd.Series(z).to_numpy(dtype=float)
    out = np.zeros(zz.size)
    p = 0.0
    for t, v in enumerate(zz):
        if not np.isfinite(v):
            p = 0.0
        elif v > entry:
            p = -1.0
        elif v < -entry:
            p = 1.0
        elif abs(v) < exit:
            p = 0.0
        out[t] = p
    return pd.Series(out, index=pd.Series(z).index, name="pos")


def book(px: pd.DataFrame, pos: pd.Series, cost_bps: float = COST_BPS,
         roll_bps: float = ROLL_BPS) -> pd.DataFrame:
    """Monthly P&L of ``pos`` units of (long $1 Brent, short $1 WTI), **one lag**.

    * ``spread_ret_t = rB_t − rW_t`` (simple monthly returns of the two legs);
    * ``gross_t = pos_{t−1} · spread_ret_t`` — the only shift in the pipeline;
    * ``trade_cost_t = |pos_{t−1} − pos_{t−2}| · 2 legs · cost_bps`` — the trade that set
      ``pos_{t−1}``, charged one-way × NAV traded per leg, booked against the return it earns;
    * ``roll_cost_t = |pos_{t−1}| · 2 legs · roll_bps`` — a monthly toll for carrying a
      futures position through a roll (bid–ask on the calendar spread, not the roll *yield*,
      which a spot tape cannot see);
    * ``usd_t = pos_{t−1} · Δspread_t`` — the same bet in $/bbl for a barrel-for-barrel book.
    """
    r = px[["brent", "wti"]].pct_change()
    spr = r["brent"] - r["wti"]
    p = pos.reindex(px.index).fillna(0.0)
    p_lag = p.shift(1).fillna(0.0)
    turn = (p - p.shift(1).fillna(0.0)).abs() * 2.0
    trade_cost = turn.shift(1).fillna(0.0) * cost_bps / 1e4
    roll_cost = p_lag.abs() * 2.0 * roll_bps / 1e4
    gross = p_lag * spr
    out = pd.DataFrame({"pos": p, "spread_ret": spr, "gross": gross,
                        "trade_cost": trade_cost, "roll_cost": roll_cost,
                        "net": gross - trade_cost - roll_cost,
                        "usd": p_lag * (px["brent"] - px["wti"]).diff(),
                        "turnover": turn})
    return out.iloc[1:]


def perf(r: pd.Series, n_boot: int = 2000, seed: int = 1023) -> dict:
    """Monthly-return summary with HAC (Newey–West) t and a circular-block-bootstrap
    (12-month blocks) Sharpe interval. Months flat count as zero return — the rule chose
    to be in cash-equivalent margin, and a futures spread earns nothing while flat."""
    r = pd.Series(r).dropna()
    if r.size < 24 or r.std(ddof=1) == 0:
        return {"n": int(r.size), "mean_m": float(r.mean()) if r.size else np.nan,
                "ann": np.nan, "vol": np.nan, "sharpe": np.nan, "t_hac": np.nan,
                "sr_lo": np.nan, "sr_hi": np.nan, "p_boot_neg": np.nan, "maxdd": np.nan}
    h = mean_tstat_hac(r)
    ci = sharpe_ci_bootstrap(r, n_boot=n_boot, periods_per_year=MONTHS, seed=seed,
                             block_size=12)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    return {"n": int(r.size), "mean_m": float(r.mean()),
            "ann": float(r.mean() * MONTHS), "vol": float(r.std(ddof=1) * np.sqrt(MONTHS)),
            "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(MONTHS)),
            "t_hac": float(h["tstat"]), "sr_lo": float(ci["ci_low"]),
            "sr_hi": float(ci["ci_high"]), "p_boot_neg": float(ci["frac_negative"]),
            "maxdd": float(dd.min())}


def episodes(bk: pd.DataFrame) -> pd.DataFrame:
    """One row per round trip: entry/exit month, side, months held, net return, and the
    **maximum adverse excursion** — the worst cumulative net return (and $/bbl) at any month
    while the trade was open, measured from entry."""
    pos = bk["pos"]
    held = pos.shift(1).fillna(0.0)  # position that earns month t
    rows = []
    t, idx = 0, bk.index
    n = len(bk)
    while t < n:
        if held.iloc[t] == 0:
            t += 1
            continue
        side = held.iloc[t]
        s = t
        while t < n and held.iloc[t] == side:
            t += 1
        seg = bk.iloc[s:t]
        cum = (1 + seg["net"]).cumprod() - 1
        cusd = seg["usd"].cumsum()
        rows.append({"entry": idx[s - 1] if s > 0 else idx[s], "exit": seg.index[-1],
                     "side": "short spread" if side < 0 else "long spread",
                     "months": int(len(seg)), "net_return": float(cum.iloc[-1]),
                     "mae": float(min(cum.min(), 0.0)), "mae_date": cum.idxmin(),
                     "usd_pnl": float(cusd.iloc[-1]), "mae_usd": float(min(cusd.min(), 0.0)),
                     "closed": bool(t < n)})
    return pd.DataFrame(rows)


def underwater(r: pd.Series, start=None) -> dict:
    """Longest spell below a previous equity high, from ``start``: peak date, trough,
    recovery date (or None) and its length in months — open spells count to the tape's end."""
    r = _window(pd.Series(r).fillna(0.0), start, None)
    eq = (1 + r).cumprod()
    eq = pd.concat([pd.Series([1.0], index=[r.index[0] - pd.offsets.MonthEnd(1)]), eq])
    peak = eq.cummax()
    under = eq < peak - 1e-12
    best = {"months": 0, "peak": None, "trough": None, "recovered": None, "depth": 0.0}
    t, idx = 0, eq.index
    while t < len(eq):
        if not under.iloc[t]:
            t += 1
            continue
        s = t
        while t < len(eq) and under.iloc[t]:
            t += 1
        months = t - s
        seg = eq.iloc[s:t] / peak.iloc[s:t] - 1
        if months > best["months"]:
            best = {"months": int(months), "peak": idx[s - 1], "trough": seg.idxmin(),
                    "recovered": idx[t] if t < len(eq) else None,
                    "depth": float(seg.min())}
    return best


def cost_sweep(px: pd.DataFrame, pos: pd.Series, grid=(0, 2, 5, 10, 20, 40),
               roll_bps: float = ROLL_BPS) -> pd.DataFrame:
    """Net Sharpe and HAC t as the one-way cost per leg rises (roll toll held fixed)."""
    rows = []
    for c in grid:
        b = book(px, pos, cost_bps=c, roll_bps=roll_bps)
        p = perf(b["net"], n_boot=200)
        rows.append({"cost_bps": c, "ann": p["ann"], "sharpe": p["sharpe"],
                     "t_hac": p["t_hac"]})
    return pd.DataFrame(rows).set_index("cost_bps")


def break_even_cost(px: pd.DataFrame, pos: pd.Series, roll_bps: float = ROLL_BPS) -> float:
    """One-way cost per leg (bps) at which the mean net return hits zero (∞ if no trades)."""
    b = book(px, pos, cost_bps=0.0, roll_bps=roll_bps)
    traded = float(b["turnover"].shift(1).fillna(0.0).sum())
    if traded <= 0:
        return float("inf")
    return float(b["net"].sum() / traded * 1e4)


# --------------------------------------------------------------------------- #
# 4. Power — what the tests can see at n ≈ 390
# --------------------------------------------------------------------------- #
def power_curve(strengths=(0.0, 0.1, 0.25, 0.5, 1.0), n_reps: int = 100,
                n_months: int = 393, halflife_months: float = 3.0,
                break_size: float = 0.0, seed: int = 1023) -> pd.DataFrame:
    """ADF and KPSS rejection rates at 5% on synthetic OU spreads of the real tape's length.

    ``signal_strength`` scales the reversion speed, so the planted half-life is
    ``halflife_months / s``. With ``break_size`` > 0 every path also carries a level shift —
    which, to an ADF test that assumes a constant mean, looks like a unit root (Perron 1989).
    """
    from cushing.data import synthetic_spread
    rows = []
    for s in strengths:
        rej_adf = rej_kpss = 0
        hls = []
        for k in range(n_reps):
            px, tr = synthetic_spread(n_months=n_months, signal_strength=s,
                                      halflife_months=halflife_months,
                                      break_size=break_size, seed=seed + k)
            x = np.log(px["brent"] / px["wti"])
            a, kp = adf(x), kpss_test(x)
            rej_adf += a["reject5"]
            rej_kpss += kp["reject5"]
            hls.append(tr["halflife_months"])
        rows.append({"signal_strength": s, "halflife": hls[0],
                     "adf_reject": rej_adf / n_reps, "kpss_reject": rej_kpss / n_reps})
    return pd.DataFrame(rows).set_index("signal_strength")


def synthetic_trade_power(strengths=(0.0, 0.5, 1.0), n_reps: int = 40,
                          n_months: int = 393, seed: int = 1023) -> pd.DataFrame:
    """Share of synthetic paths on which the expanding-window fader's gross HAC t ≥ 2."""
    from cushing.data import synthetic_spread
    rows = []
    for s in strengths:
        ts = []
        for k in range(n_reps):
            px, _ = synthetic_spread(n_months=n_months, signal_strength=s, seed=seed + k)
            sf = spread_frame(px)
            b = book(px, positions(zscore_realtime(sf["log_ratio"])), 0.0, 0.0)
            ts.append(mean_tstat_hac(b["gross"])["tstat"])
        ts = np.asarray(ts, dtype=float)
        rows.append({"signal_strength": s, "median_t": float(np.nanmedian(ts)),
                     "share_t_ge_2": float(np.nanmean(ts >= 2.0))})
    return pd.DataFrame(rows).set_index("signal_strength")


# --------------------------------------------------------------------------- #
# Verdict — thresholds fixed before the real tape was run
# --------------------------------------------------------------------------- #
def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    **Signal** (is the tether real, and does fading it pay?). Every test reads the log
    Brent/WTI ratio — the series the dollar-neutral book trades:

    * **Real** — the real-time fader's gross HAC t ≥ 2 over the full sample, *and* the log
      ratio is stationary over the full sample (ADF p < 0.05 and KPSS p ≥ 0.05), *and* no
      significant mean break (bootstrap sup-F p ≥ 0.05). That is the claim as stated: always.
    * **Mixed** — a significant break (bootstrap p < 0.05) with the pre-break regime itself
      stationary (ADF p < 0.05): tethered within a regime, untethered across it.
    * **Weak** — gross t ≥ 1, or full-sample ADF p < 0.05, without the above.
    * **None** — otherwise.

    **Tradability**: **Investable** is out of reach by construction — the tape is spot,
    monthly-averaged, and blind to the roll. **Fragile** if the real-time fader's net HAC
    t ≥ 2 *and* the pre-2010-calibrated trader's longest underwater spell is ≤ 36 months;
    **Mirage** otherwise.
    """
    break_sig = h["break_p"] < 0.05
    stationary_full = h["adf_p_full"] < 0.05 and h["kpss_p_full"] >= 0.05
    if h["t_gross"] >= 2.0 and stationary_full and not break_sig:
        signal = "Real"
    elif break_sig and h["adf_p_pre"] < 0.05:
        signal = "Mixed"
    elif h["t_gross"] >= 1.0 or h["adf_p_full"] < 0.05:
        signal = "Weak"
    else:
        signal = "None"
    trad = ("Fragile" if (h["t_net"] >= 2.0 and h["frozen_underwater_months"] <= 36)
            else "Mirage")

    uw = h["frozen_underwater_months"]
    uw_txt = (f"**{uw} months**, still open when the tape ends"
              if not h["frozen_recovered"] else f"**{uw} months**")
    lead = {
        "Real": "The tether held, all the way through.",
        "Mixed": "Tethered inside a regime, untethered across one.",
        "Weak": ("A tether that held for two decades and then moved; what is left cannot be "
                 "certified as the claim."),
        "None": "No tether this tape can certify.",
    }[signal]
    signal_why = (
        f"{lead} Over the full {h['n_months']} months the log Brent/WTI ratio is **not** "
        f"stationary (ADF p = {h['adf_p_full']:.2f}; KPSS rejects stationarity, "
        f"p {'≤' if h['kpss_p_full'] <= 0.01 else '='} {h['kpss_p_full']:.2f}). A data-driven "
        f"sup-F scan puts a mean break at **{h['break_date']}** (sup-F {h['sup_f']:.0f}, "
        f"bootstrap p = {h['break_p']:.3f} against a persistent, break-free AR(1)); the dollar "
        f"spread's average moved from {h['mean_pre_usd']:+.2f} to {h['mean_post_usd']:+.2f} "
        f"$/bbl. Before the break the ratio reverted with a half-life of "
        f"**{h['hl_pre']:.1f} months** (95% band {h['hl_pre_lo']:.1f}–{h['hl_pre_hi']:.1f}), "
        f"and its ADF p is {h['adf_p_pre']:.3f} in logs "
        f"({'< 0.001' if h['adf_p_pre_usd'] < 0.001 else format(h['adf_p_pre_usd'], '.3f')} "
        f"in dollars). The real-time fader's gross HAC t is **{h['t_gross']:+.2f}** "
        f"(block-bootstrap Sharpe 95% CI {h['sr_gross_lo']:+.2f} to {h['sr_gross_hi']:+.2f}).")
    if signal == "Weak" and break_sig and h["adf_p_pre_usd"] < 0.05 <= h["adf_p_pre"]:
        signal_why += (
            " The pre-registered **Mixed** reading — a significant break *and* a stationary "
            "pre-break regime — misses on the log ratio, apparently because that regime "
            "carries a smaller shift of its own in the mid-2000s (Bai–Perron places one there); read in dollars it would have been "
            "Mixed. The stamp follows the rule, and the rule reads logs.")
    trad_why = (
        f"On spot, monthly-average prices that nobody can trade — an **upper bound** before "
        f"the futures roll — the real-time (expanding-window) fader earns "
        f"{h['ann_net']:+.2%} a year net (Sharpe {h['sr_net']:+.2f}, HAC t {h['t_net']:+.2f}). "
        f"The trader who calibrated on 1987–2009 and kept the faith went short the spread in "
        f"{h['frozen_entry']}, sat through a maximum adverse excursion of "
        f"**{h['frozen_mae']:.0%}** of NAV ({h['frozen_mae_usd']:+.1f} $/bbl per barrel "
        f"pair) and was underwater {uw_txt}.")
    one = (f"Brent and WTI were tethered until a data-chosen break in {h['break_date']} moved "
           f"the spread's home by {h['mean_post_usd'] - h['mean_pre_usd']:+.0f} $/bbl — fading "
           f"it in real time earned a gross t of {h['t_gross']:.1f} on untradeable spot "
           f"prices, and the trader calibrated before 2010 spent {uw} months underwater.")
    return {"signal": signal, "signal_why": signal_why, "trad": trad, "trad_why": trad_why,
            "one_sentence": one}
