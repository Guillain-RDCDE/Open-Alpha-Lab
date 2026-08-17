"""Ladder construction + inference for Study 921 — Bill Ladder vs ETF.

**The instrument.** A rolling 3-month Treasury-bill ladder: 13 equal rungs, one bought
every 7 calendar days, each held to maturity. Every rung's return is locked at purchase —
a bill bought at a bond-equivalent yield *y* accrues *y* on an actual/365 basis until it
matures, whatever happens to rates in between. The ladder's daily accrual is therefore the
equal-weighted mean of the 13 live rungs' locked yields.

**The rate.** ``^IRX`` is the CBOE index of the **discount** rate on the most recently
auctioned 13-week bill, quoted in percent on a 360-day basis. A discount rate is not a
yield, and the conversion is not a rounding detail — at a 3.7% discount it is worth about
+9 bps, which is the same order as the fee this study is trying to measure. So the
conversion is stated explicitly, and the whole race is re-run on the raw (unconverted)
quote as a robustness floor::

    P    = 1 - d * tenor / 360          (price per 1 of face)
    BEY  = (1 - P) / P * 365 / tenor    (bond-equivalent, actual/365)

**The lag.** Exactly one. The ``^IRX`` quote observed at the close of day ``t`` prices the
bill bought on day ``t+1``; the rung then accrues from ``t+1`` onward. Nothing else in the
pipeline looks forward.

**The race.** Cash is the numeraire, so there is nothing to be excess *of*: we race the
ladder's total return directly against each ETF's total return (``auto_adjust=True``, so
the monthly distributions are reinvested). The headline is the annualised return gap in
bps/yr with a Newey-West *t* on the daily difference, a circular block-bootstrap CI, an
era cut spanning the zero-rate and post-2022 regimes, and two friction sweeps.

**The frictions** (both PROXY / ASSUMPTION, both swept):

- ``cost_bps`` — per-auction friction charged on the rung being bought, i.e.
  ``cost_bps x (1 / n_rungs) x NAV`` on each roll. At TreasuryDirect a non-competitive
  auction bid costs nothing; through a broker's secondary-market bill desk a 1-3 bps
  round-trip is ordinary. 52 rolls a year of 1/13 of NAV means the annual drag is roughly
  ``4 x cost_bps``.
- ``idle_days`` — calendar days a maturing rung's proceeds sit uninvested before the
  replacement settles. With auto-reinvestment at TreasuryDirect this is 0; with a manual
  roll it is 1-5 days, and at 5% short rates each idle day costs real money.

**A warning that belongs in the arithmetic, not the footnotes.** The ladder is carried at
amortised cost — held to maturity, never marked — so its measured volatility is near zero
by *accounting convention*, not because it is safer than the ETF. Sharpe ratios computed
against that zero are meaningless, which is why this study reports return gaps and *t*
statistics on the difference and refuses to quote a Sharpe race.
"""

from __future__ import annotations

import collections

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.0
DEFAULT_TENOR = 91.0
DEFAULT_RUNGS = 13


# --------------------------------------------------------------------------- #
# Rate conversion
# --------------------------------------------------------------------------- #
def discount_to_bey(discount_pct: pd.Series, tenor: float = DEFAULT_TENOR) -> pd.Series:
    """Convert a bank-discount quote (percent, 360-day basis) to a bond-equivalent yield.

    ``P = 1 - d * tenor / 360`` then ``BEY = (1 - P) / P * 365 / tenor``. Returned as a
    decimal (0.05 = 5%), not percent. The two adjustments — the 360→365 day count and the
    discount→price base change — both push the yield *up*, by roughly 9 bps at a 3.7%
    quote and 2 bps at a 1% quote.
    """
    d = pd.Series(discount_pct).astype(float) / 100.0
    price = 1.0 - d * tenor / 360.0
    return ((1.0 - price) / price * DAYS_PER_YEAR / tenor).rename("bey")


def rate_to_yield(rate_pct: pd.Series, basis: str = "discount",
                  tenor: float = DEFAULT_TENOR) -> pd.Series:
    """Dispatch the rate convention: ``'discount'`` converts, ``'raw'`` takes the quote as-is.

    ``'raw'`` is the deliberately conservative reading — it assumes ^IRX is already a
    bond-equivalent yield, which understates the ladder. It exists so the verdict can be
    checked against the weaker of the two possible conventions.
    """
    if basis == "discount":
        return discount_to_bey(rate_pct, tenor=tenor)
    if basis == "raw":
        return (pd.Series(rate_pct).astype(float) / 100.0).rename("bey")
    raise ValueError(f"unknown basis {basis!r} (expected 'discount' or 'raw')")


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #
def ladder_returns(
    rate_pct: pd.Series,
    n_rungs: int = DEFAULT_RUNGS,
    tenor: float = DEFAULT_TENOR,
    basis: str = "discount",
    cost_bps: float = 0.0,
    idle_days: float = 0.0,
) -> pd.DataFrame:
    """Daily simple returns of a rolling held-to-maturity bill ladder.

    Parameters
    ----------
    rate_pct:
        The ^IRX quote in percent, indexed by date (the study's only rate input).
    n_rungs:
        Number of live rungs; the roll spacing is ``tenor / n_rungs`` calendar days
        rounded to whole days (13 rungs of a 91-day bill → a purchase every 7 days).
    basis, tenor:
        Rate convention and bill tenor — see ``rate_to_yield``.
    cost_bps, idle_days:
        The two PROXY frictions, charged on the rung being rolled: ``cost_bps`` as a
        one-way cost on ``1 / n_rungs`` of NAV, ``idle_days`` as forgone accrual on the
        same slice.

    Returns a frame with ``rate`` (the ladder's blended locked yield, decimal),
    ``roll`` (1 on purchase days), ``r_gross`` and ``r_ladder`` (net of both frictions).
    Rows before the ladder is fully built (the first ``tenor`` days) are dropped.
    """
    s = pd.Series(rate_pct).astype(float).dropna().sort_index()
    bey = rate_to_yield(s, basis=basis, tenor=tenor)
    # ONE lag: the quote seen at the close of t prices the bill bought at t+1.
    bey_lag = bey.shift(1)
    idx = s.index

    step = max(1, int(round(tenor / n_rungs)))
    live = collections.deque(maxlen=n_rungs)
    blended = np.full(len(idx), np.nan)
    roll = np.zeros(len(idx))
    last_buy = None
    for i, ts in enumerate(idx):
        y = bey_lag.iloc[i]
        if not np.isfinite(y):
            continue
        if last_buy is None or (ts - last_buy).days >= step:
            live.append(float(y))
            last_buy = ts
            roll[i] = 1.0
        if len(live) == n_rungs:
            blended[i] = float(np.mean(live))

    elapsed = pd.Series(idx).diff().dt.days.to_numpy(dtype=float)
    # Accrual is actual/365 over *calendar* days, so a Friday→Monday step pays three
    # days — the same clock the ETF's total-return close runs on.
    r_gross = blended / DAYS_PER_YEAR * elapsed
    idle_drag = roll * (1.0 / n_rungs) * blended * idle_days / DAYS_PER_YEAR
    fee_drag = roll * (1.0 / n_rungs) * cost_bps * 1e-4
    out = pd.DataFrame(
        {
            "rate": blended,
            "roll": roll,
            "r_gross": r_gross,
            "r_ladder": r_gross - idle_drag - fee_drag,
        },
        index=idx,
    )
    return out.dropna(subset=["r_ladder"])


# --------------------------------------------------------------------------- #
# Inference primitives (mirror of Study 912)
# --------------------------------------------------------------------------- #
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
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0; default bandwidth 4(n/100)^(2/9).

    The bandwidth matters here in an unusual direction. A bill ETF's daily close carries
    bid-offer bounce, so the return difference is *negatively* autocorrelated at lag 1 and
    the HAC standard error is **smaller** than the naive one — HAC raises this study's *t*
    rather than deflating it. Reported both ways in ``examples/verify.py``.

    Because HAC here *helps* the result, its bandwidth cannot be left unaudited: on this
    tape the *t* rises monotonically with ``lags`` (+1.44 at 1 lag, +2.75 at the automatic
    9, +9.26 at 252). The automatic rule is therefore the *conservative* end of the kernel
    family, not a favourable pick — and ``nonoverlap_t`` settles the question with no
    bandwidth at all. See ``hac_bandwidth_scan``.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * (float(u[l:] @ u[:-l]) / n)
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def acf1(x) -> float:
    """Lag-1 autocorrelation — the Roll (1984) bid-offer-bounce diagnostic.

    A negative value on the ladder-minus-ETF difference means the ETF's close bounces
    between bid and offer, which is *why* the i.i.d. standard error overstates the
    uncertainty here and HAC legitimately raises the *t*.
    """
    a = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(a) < 3:
        return float("nan")
    u = a - a.mean()
    den = float(u @ u)
    return float(u[1:] @ u[:-1] / den) if den > 0 else float("nan")


def nonoverlap_t(diff: pd.Series, freq: str = "M") -> dict:
    """Ordinary *t* on **non-overlapping** period sums — the tuning-knob-free arbiter.

    This study has an inference problem it must not paper over: the HAC *t* (+2.75) is more
    than double the naive daily *t* (+1.15), and HAC is the number the verdict leans on. A
    reader is entitled to ask whether the significance is a property of the tape or of a
    chosen bandwidth. Both existing answers have a knob — HAC has ``lags``, the block
    bootstrap has ``block`` — so neither can settle it alone.

    This one has none. Sum the daily difference into **non-overlapping** calendar periods:
    the ETF's bid-offer bounce is a first difference of a stationary pricing error, so it
    *telescopes* inside each period and only the endpoints survive, while the accrual gap
    accumulates. Consecutive period sums are then close to independent and the ordinary
    one-sample *t* is valid as it stands, with nothing to choose.

    On the real tape this vindicates HAC rather than the naive *t*: weekly +2.18, monthly
    +3.27, quarterly +3.54 against the naive daily +1.15. That is the evidence the Real
    stamp actually rests on.

    ``freq`` is a pandas **period** alias (``'W'``/``'M'``/``'Q'``), deliberately not a
    resample alias — the resample spellings changed between pandas 2 and 3, the period
    aliases did not.
    """
    d = pd.Series(diff).dropna()
    if len(d) < 3:
        return {"freq": freq, "n_periods": 0, "mean_bps": float("nan"),
                "t": float("nan")}
    g = d.groupby(pd.PeriodIndex(d.index, freq=freq)).sum()
    return {
        "freq": freq,
        "n_periods": int(len(g)),
        "mean_bps": float(g.mean() * 1e4),
        "t": one_sample_t(g.to_numpy()),
    }


def horizon_check(diff: pd.Series, freqs=("W", "M", "Q")) -> list[dict]:
    """``nonoverlap_t`` across several aggregation horizons (the bandwidth-free ladder)."""
    return [nonoverlap_t(diff, freq=f) for f in freqs]


def hac_bandwidth_scan(diff: pd.Series,
                       lag_grid=(1, 2, 5, 9, 21, 63, 252)) -> list[dict]:
    """HAC *t* across bandwidths, plus the naive *t*, so the knob is visible not hidden.

    Disclosed because HAC *raises* this study's *t*. The automatic bandwidth (9 lags on
    n≈4,800) sits near the bottom of this range, so the headline is not the flattering pick.
    """
    x = pd.Series(diff).dropna().to_numpy(dtype=float)
    rows = [{"lags": 0, "t": one_sample_t(x), "label": "naive (i.i.d.)"}]
    for L in lag_grid:
        rows.append({"lags": int(L), "t": newey_west_t(x, lags=int(L)), "label": "HAC"})
    return rows


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The race
# --------------------------------------------------------------------------- #
def _years(idx: pd.DatetimeIndex) -> float:
    return float((idx[-1] - idx[0]).days) / 365.25


def annualise_gap_bps(diff: pd.Series, obs_per_year: float | None = None) -> float:
    """Annualised mean daily return difference, in bps/yr.

    Scaled by the *realised* observations-per-year of the sample rather than a nominal
    252, so a tape with holidays or a short window is not silently mis-annualised. When
    the series is a non-contiguous *subset* of a parent sample (a rate-regime mask, say),
    pass the parent's ``obs_per_year`` — otherwise the gaps in the mask inflate the span
    and the annualisation is understated.
    """
    d = pd.Series(diff).dropna()
    if len(d) < 2:
        return float("nan")
    if obs_per_year is None:
        obs_per_year = len(d) / _years(d.index)
    return float(d.mean() * obs_per_year * 1e4)


def cagr(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    wealth = float((1.0 + r).prod())
    y = _years(r.index)
    return float(wealth ** (1.0 / y) - 1.0) if y > 0 and wealth > 0 else float("nan")


def ann_vol(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 3:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(len(r) / _years(r.index)))


def race(
    rate_pct: pd.Series,
    etf_close: pd.Series,
    n_rungs: int = DEFAULT_RUNGS,
    tenor: float = DEFAULT_TENOR,
    basis: str = "discount",
    cost_bps: float = 0.0,
    idle_days: float = 0.0,
) -> dict:
    """Race the ladder against one cash ETF's total return on their common dates.

    Cash is the numeraire — there is nothing to take excess *of* — so the comparison is
    total return vs total return, and the statistic of interest is the annualised gap
    ``ladder - ETF`` in bps/yr with a Newey-West *t* on the daily difference.

    Returns a dict of the two legs' CAGR and annualised vol, the gap, the HAC and naive
    *t*, the roll count, and the aligned ``diff`` series for downstream bootstrapping.
    """
    lad = ladder_returns(rate_pct, n_rungs=n_rungs, tenor=tenor, basis=basis,
                         cost_bps=cost_bps, idle_days=idle_days)
    e = pd.Series(etf_close).astype(float).dropna().sort_index()
    common = lad.index.intersection(e.index)
    lad = lad.loc[common]
    r_etf = e.loc[common].pct_change().rename("r_etf")
    df = pd.concat([lad["r_ladder"], r_etf, lad["roll"]], axis=1).dropna()
    diff = (df["r_ladder"] - df["r_etf"]).rename("diff")

    return {
        "n_days": int(len(df)),
        "start": df.index[0], "end": df.index[-1],
        "years": _years(df.index),
        "cagr_ladder": cagr(df["r_ladder"]),
        "cagr_etf": cagr(df["r_etf"]),
        "vol_ladder": ann_vol(df["r_ladder"]),
        "vol_etf": ann_vol(df["r_etf"]),
        "gap_bps": annualise_gap_bps(diff),
        "t_hac": newey_west_t(diff.to_numpy()),
        "t_naive": one_sample_t(diff.to_numpy()),
        # The lag-1 autocorrelation of the difference is the *evidence* for preferring HAC
        # over the naive t: a strongly negative value is the Roll (1984) bid-offer-bounce
        # signature that makes the i.i.d. standard error too big.
        "acf1": acf1(diff),
        "n_rolls": int(df["roll"].sum()),
        "mean_rate": float(lad["rate"].mean()),
        "diff": diff,
        "frame": df,
    }


# --------------------------------------------------------------------------- #
# Bootstrap CI on the annualised gap
# --------------------------------------------------------------------------- #
def bootstrap_gap_ci(
    diff: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 921,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the annualised return gap (bps/yr).

    Blocks of ``block`` consecutive days preserve the ETF's bid-offer bounce and the
    ladder's rate persistence. A Sharpe bootstrap would be meaningless here (the ladder's
    denominator is an accounting artefact), so the bootstrapped quantity is the *gap*.
    """
    d = pd.Series(diff).dropna()
    x = d.to_numpy(dtype=float)
    n = x.size
    if n < block + 2:
        return {"gap_bps": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    scale = len(d) / _years(d.index) * 1e4
    point = float(x.mean() * scale)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = x[idx].mean() * scale
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "gap_bps": point, "ci_low": float(lo), "ci_high": float(hi),
        "frac_negative": float((boots < 0).mean()),
        "n_obs": int(n), "n_boot": int(n_boot), "block": int(block),
    }


# --------------------------------------------------------------------------- #
# Cuts and sweeps
# --------------------------------------------------------------------------- #
def era_cut(rate_pct: pd.Series, etf_close: pd.Series,
            edges=("2016-01-01", "2022-01-01"), **kw) -> dict:
    """Slice the full-sample race into date slabs at ``edges``.

    The slabs are cut out of the *already-built* difference series rather than by re-running
    the ladder on a truncated tape, so no slab loses a quarter to ladder warmup and the
    three slabs partition the sample exactly.

    The default edges bracket the three short-rate regimes this tape contains: the crisis
    and its aftermath, the deep zero-rate years, and the post-2022 hiking regime. A
    fee-driven gap should be roughly *constant* across all three, because a fee is charged
    whatever the level of rates; a curve- or luck-driven gap should not be.
    """
    res = race(rate_pct, etf_close, **kw)
    df, diff = res["frame"], res["diff"]
    bounds = [None] + list(edges) + [None]
    out = {}
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        tag = f"{lo or 'start'}..{hi or 'end'}"
        # half-open [lo, hi) so the slabs partition the sample exactly (label slicing
        # would include the boundary date in both neighbours)
        m = np.ones(len(diff), dtype=bool)
        if lo is not None:
            m &= diff.index >= pd.Timestamp(lo)
        if hi is not None:
            m &= diff.index < pd.Timestamp(hi)
        d = diff[m]
        sub = df[m]
        if len(d) < 250:
            out[tag] = None
            continue
        lvl = pd.Series(rate_pct).reindex(d.index).ffill()
        out[tag] = {
            "n_days": int(len(d)),
            "start": d.index[0], "end": d.index[-1],
            "gap_bps": annualise_gap_bps(d),
            "t_hac": newey_west_t(d.to_numpy()),
            # knob-free within-era t, so "positive in all three eras" can be read with the
            # right amount of confidence rather than as three equal endorsements
            "t_month": nonoverlap_t(d, freq="M")["t"],
            "cagr_ladder": cagr(sub["r_ladder"]),
            "cagr_etf": cagr(sub["r_etf"]),
            "mean_rate_pct": float(lvl.mean()),
        }
    return out


def rate_regime_cut(rate_pct: pd.Series, etf_close: pd.Series,
                    threshold_pct: float = 1.0, **kw) -> dict:
    """Split *by rate level* rather than by date: quote below/above ``threshold_pct``.

    The zero-rate years are the cleanest test of a fee story. When the bill yield is 5 bps
    and the fund charges 13, the fund's net yield is negative and the ladder should win by
    the full fee; if instead the gap collapses with the rate level, the ladder is earning
    something rate-dependent (curve carry) rather than avoiding a fee. Both regimes are
    annualised on the *parent* sample's observations-per-year, since each mask is a
    non-contiguous subset.
    """
    res = race(rate_pct, etf_close, **kw)
    opy = res["n_days"] / res["years"]
    lvl = pd.Series(rate_pct).reindex(res["diff"].index).ffill()
    out = {}
    for tag, mask in [("zero_rate", lvl < threshold_pct), ("normal_rate", lvl >= threshold_pct)]:
        m = mask.fillna(False)
        d = res["diff"][m]
        if len(d) < 200:
            out[tag] = None
            continue
        out[tag] = {"n_days": int(len(d)),
                    "gap_bps": annualise_gap_bps(d, obs_per_year=opy),
                    "t_hac": newey_west_t(d.to_numpy()),
                    "mean_rate_pct": float(lvl[m].mean())}
    return out


def fee_attribution(rate_pct: pd.Series, etf_close: pd.Series,
                    expense_ratio_bps: float, **kw) -> dict:
    """Is the gap the fee? Compare the ladder's return to the ETF's return *before* fees.

    ``gross_etf = cagr_etf + expense_ratio``. If the ladder and the fund hold the same
    thing and the ladder's only advantage is not paying the manager, the residual
    ``ladder - gross_etf`` should be ~0. A positive residual means the ladder is also
    earning something the fund is not (curve, tenor); a negative residual means the fund
    is earning something the ladder is not — for a longer-maturity fund, duration.

    The expense ratio is a PROXY (a sponsor-published sticker, not tape), so the residual
    inherits that uncertainty; it is a decomposition, not a test.
    """
    res = race(rate_pct, etf_close, **kw)
    gross_etf = res["cagr_etf"] * 1e4 + expense_ratio_bps
    return {
        "cagr_ladder_bps": res["cagr_ladder"] * 1e4,
        "cagr_etf_bps": res["cagr_etf"] * 1e4,
        "expense_ratio_bps": float(expense_ratio_bps),
        "gross_etf_bps": float(gross_etf),
        "residual_bps": float(res["cagr_ladder"] * 1e4 - gross_etf),
        "gap_bps": res["gap_bps"], "t_hac": res["t_hac"], "n_days": res["n_days"],
    }


def friction_sweep(rate_pct: pd.Series, etf_close: pd.Series,
                   cost_grid=(0.0, 1.0, 2.0, 3.0, 5.0, 10.0), **kw) -> list[dict]:
    """Per-auction friction sweep: gap and *t* at several one-way costs per roll.

    52 rolls a year, each on 1/13 of NAV, so the annual drag is about ``4 x cost_bps``.
    """
    rows = []
    for c in cost_grid:
        res = race(rate_pct, etf_close, cost_bps=c, **kw)
        rows.append({"cost_bps": c, "gap_bps": res["gap_bps"], "t_hac": res["t_hac"],
                     "cagr_ladder": res["cagr_ladder"]})
    return rows


def idle_sweep(rate_pct: pd.Series, etf_close: pd.Series,
               idle_grid=(0.0, 1.0, 2.0, 3.0, 5.0), **kw) -> list[dict]:
    """Reinvestment-lag sweep: gap and *t* when maturing proceeds sit idle for N days."""
    rows = []
    for d in idle_grid:
        res = race(rate_pct, etf_close, idle_days=d, **kw)
        rows.append({"idle_days": d, "gap_bps": res["gap_bps"], "t_hac": res["t_hac"]})
    return rows


def basis_check(rate_pct: pd.Series, etf_close: pd.Series, **kw) -> dict:
    """Re-run the race under both rate conventions (the discount→BEY assumption sweep)."""
    return {b: {k: v for k, v in race(rate_pct, etf_close, basis=b, **kw).items()
                if k in ("gap_bps", "t_hac", "cagr_ladder")}
            for b in ("discount", "raw")}


def rung_check(rate_pct: pd.Series, etf_close: pd.Series,
               rung_grid=(4, 13, 26), **kw) -> list[dict]:
    """Vary the ladder's rung count (a monthly, weekly and twice-weekly roll schedule)."""
    rows = []
    for n in rung_grid:
        res = race(rate_pct, etf_close, n_rungs=n, **kw)
        rows.append({"n_rungs": n, "gap_bps": res["gap_bps"], "t_hac": res["t_hac"],
                     "n_rolls": res["n_rolls"]})
    return rows


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(frame: pd.DataFrame, **kw) -> dict:
    """Race the ladder against the synthetic ETF on a ``data.synthetic_daily`` frame.

    On the planted world the recovered gap should land near the planted ``fee_bps``; on the
    null (a free ETF) it should be flat zero. Proves the pipeline neither invents nor eats
    a fee — it never supports a real-tape stamp.
    """
    res = race(frame["irx"], frame["etf"], **kw)
    return {
        "gap_bps": res["gap_bps"], "t_hac": res["t_hac"], "n_days": res["n_days"],
        "cagr_ladder": res["cagr_ladder"], "cagr_etf": res["cagr_etf"],
        "vol_ladder": res["vol_ladder"], "vol_etf": res["vol_etf"],
        "mean_rate": res["mean_rate"],
    }
