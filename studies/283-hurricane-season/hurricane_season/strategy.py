"""Strategy and analysis layer for Study 283 (Hurricane-Season).

The claim: the Atlantic hurricane season (Jun 1 - Nov 30) drags on equities --
either the broad market or, more specifically, the property-and-casualty insurers
who carry the catastrophe risk. We test it three ways:

  - **Seasonal mean test.** Compare in-season vs off-season daily returns with a
    Welch t-test and a HAC (Newey-West) t-stat on the demeaned in-season indicator
    regression -- daily equity returns have mild autocorrelation and volatility
    clustering, so a naive i.i.d. t overstates significance. The HAC t on the REAL
    tape is the bar for a REAL signal.

  - **Permutation test.** Block-shuffle the in-season label across whole seasons
    and recompute the spread, to see where the real spread lands under the null.

  - **Event windows.** Average the cumulative abnormal return (vs the contemporaneous
    market) over [-5, +20] trading days around each major U.S. landfall -- the sharp
    test of whether *named storms* move insurers.

The honest baselines: equities drift up ~5-6 bps/day regardless; the season covers
half the calendar, so an "avoid the season" rule is really just a market-timing rule
that sits out half the year. We label gross vs net, price-only vs total-return, and
name survivorship on the insurer basket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# HAC (Newey-West) t-stat for the mean of a series
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """Newey-West HAC t-stat and standard error for the mean of ``x``.

    Tests H0: mean(x) == 0 with a heteroskedasticity-and-autocorrelation-consistent
    standard error (Bartlett kernel). For daily equity returns this is the honest
    standard error; the i.i.d. t overstates significance because of volatility
    clustering and mild autocorrelation.

    ``lags`` defaults to the Newey-West rule-of-thumb ``floor(4*(n/100)^(2/9))``.
    Returns ``(tstat, se)``.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    mu = x.mean()
    e = x - mu
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    gamma0 = (e @ e) / n
    var = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        cov = (e[k:] @ e[:-k]) / n
        var += 2.0 * w * cov
    se = np.sqrt(var / n)
    if se == 0:
        return float("nan"), float(se)
    return float(mu / se), float(se)


# ---------------------------------------------------------------------------
# Seasonal mean test -- in-season vs off-season
# ---------------------------------------------------------------------------
def season_stats(
    df: pd.DataFrame,
    n_permutations: int = 5_000,
    seed: int = 283,
) -> dict:
    """Compare in-season vs off-season daily returns.

    Expects ``df`` with columns ``ret`` (daily simple return) and ``in_season``
    (bool), indexed by date.

    Reports (all returns in basis points/day unless noted):
      - ``n``, ``n_in``, ``n_off``
      - ``mean_in``, ``mean_off``, ``spread`` (in - off), in bps/day
      - ``welch_t``, ``welch_p`` : Welch t-test on daily returns (i.i.d. -- optimistic)
      - ``hac_t`` : HAC t-stat on the in-season indicator (the honest bar)
      - ``ann_in``, ``ann_off`` : annualised mean (252 trading days), in %
      - ``perm_p`` : block-permutation p-value (two-sided) for the spread
    """
    d = df.dropna(subset=["ret"]).copy()
    ret = d["ret"].to_numpy(dtype=float)
    season = d["in_season"].to_numpy(dtype=bool)
    n = len(d)
    n_in = int(season.sum())
    n_off = int((~season).sum())

    in_ret = ret[season]
    off_ret = ret[~season]
    mean_in_bps = float(in_ret.mean() * 1e4)
    mean_off_bps = float(off_ret.mean() * 1e4)
    spread_bps = mean_in_bps - mean_off_bps

    if n_in >= 2 and n_off >= 2:
        welch_t, welch_p = scipy_stats.ttest_ind(in_ret, off_ret, equal_var=False)
    else:
        welch_t, welch_p = float("nan"), float("nan")

    # HAC t on the in-season indicator: regress ret on (in_season - mean), the
    # slope = spread; we use the per-observation contribution (centred indicator
    # times centred return) and take its HAC mean t. Simpler and robust: build the
    # "in-minus-off" daily contribution and HAC-t its mean.
    p_in = season.mean()
    contrib = (season.astype(float) - p_in) * ret / (p_in * (1.0 - p_in))
    hac_t, _ = hac_tstat(contrib)

    ann_in = mean_in_bps * 252 / 100.0
    ann_off = mean_off_bps * 252 / 100.0

    # Block permutation: shuffle whole-season labels (group by season-year) so the
    # autocorrelation structure within a season is preserved under the null.
    year = d.index.year.to_numpy()
    # A "hurricane-season-year" block: each calendar year is one block; we flip
    # which half of the year is labelled in-season by rotating the season mask
    # within each year is overkill -- instead permute the per-year *fraction*.
    rng = np.random.default_rng(seed)
    years = np.unique(year)
    # Per-year demeaned daily returns, then randomly assign each day's season label
    # by reshuffling the full in_season vector in contiguous yearly blocks.
    perm_spreads = np.empty(n_permutations, dtype=float)
    season_f = season.astype(float)
    for i in range(n_permutations):
        shuffled = season_f.copy()
        # rotate the season mask by a random whole number of days within each year
        for y in years:
            ymask = year == y
            block = shuffled[ymask]
            if len(block) > 1:
                shift = rng.integers(0, len(block))
                shuffled[ymask] = np.roll(block, shift)
        sm = shuffled.astype(bool)
        if sm.sum() == 0 or (~sm).sum() == 0:
            perm_spreads[i] = 0.0
            continue
        perm_spreads[i] = (ret[sm].mean() - ret[~sm].mean()) * 1e4
    perm_p = float((np.abs(perm_spreads) >= abs(spread_bps)).mean())

    return {
        "n": int(n),
        "n_in": n_in,
        "n_off": n_off,
        "mean_in": round(mean_in_bps, 3),
        "mean_off": round(mean_off_bps, 3),
        "spread": round(spread_bps, 3),
        "welch_t": round(float(welch_t), 3),
        "welch_p": round(float(welch_p), 4),
        "hac_t": round(float(hac_t), 3),
        "ann_in": round(ann_in, 2),
        "ann_off": round(ann_off, 2),
        "perm_p": round(perm_p, 4),
        "perm_spreads": perm_spreads,  # full distribution, for plots
    }


# ---------------------------------------------------------------------------
# Event-window study around major landfalls
# ---------------------------------------------------------------------------
def event_windows(
    asset: pd.DataFrame,
    market: pd.DataFrame,
    landfalls: pd.DataFrame,
    pre: int = 5,
    post: int = 20,
    lag: int = 1,
) -> dict:
    """Average abnormal return around major U.S. landfalls.

    For each landfall date L, find the nearest trading day on or after L, apply a
    one-day execution lag (``lag``), then take the asset's return minus the market
    return over trading days [-``pre``, +``post``] relative to the (lagged) event
    day. The abnormal return is ``asset_ret - market_ret`` (a crude one-factor
    market adjustment with beta fixed at 1 -- conservative and transparent).

    Expects ``asset`` and ``market`` each with a ``ret`` column indexed by date.
    Returns a dict with ``rel`` (the trading-day offsets), ``caar`` (cumulative
    average abnormal return, in %), ``n_events``, and ``car_end`` (mean CAR at the
    final offset, in %, with its i.i.d. t-stat ``car_t``).
    """
    a = asset["ret"].dropna()
    m = market["ret"].reindex(a.index).fillna(0.0)
    abn = (a - m)
    dates = abn.index
    rel = np.arange(-pre, post + 1)
    per_event = []
    for L in pd.to_datetime(landfalls["landfall"]):
        # first trading day on or after L
        pos = dates.searchsorted(L, side="left")
        if pos >= len(dates):
            continue
        ev = pos + lag  # execution lag
        lo, hi = ev - pre, ev + post
        if lo < 0 or hi >= len(dates):
            continue
        seg = abn.iloc[lo: hi + 1].to_numpy()
        if len(seg) == len(rel):
            per_event.append(seg)
    if not per_event:
        return {"rel": rel, "caar": np.zeros(len(rel)), "n_events": 0,
                "car_end": float("nan"), "car_t": float("nan")}
    mat = np.vstack(per_event)             # (n_events, window)
    aar = mat.mean(axis=0)                  # average abnormal return per offset
    caar = np.cumsum(aar) * 100.0           # cumulative, in %
    # CAR per event (sum of abnormal returns over the window), then t-test vs 0.
    car_per_event = mat.sum(axis=1) * 100.0
    car_end = float(car_per_event.mean())
    if len(car_per_event) >= 2 and car_per_event.std(ddof=1) > 0:
        car_t = float(car_end / (car_per_event.std(ddof=1) / np.sqrt(len(car_per_event))))
    else:
        car_t = float("nan")
    return {
        "rel": rel,
        "caar": caar,
        "n_events": len(per_event),
        "car_end": round(car_end, 3),
        "car_t": round(car_t, 3),
    }


# ---------------------------------------------------------------------------
# Avoid-the-season timing strategy vs buy-and-hold
# ---------------------------------------------------------------------------
def avoid_season_strategy(df: pd.DataFrame, cost_bps: float = 1.0) -> dict:
    """'Sit out the hurricane season' timing rule vs buy-and-hold.

    The rule: hold the asset off-season (Dec 1 - May 31), go to cash in-season
    (Jun 1 - Nov 30). One-way transaction cost ``cost_bps`` is charged on the two
    switches per year (in and out). Returns annualised gross and net figures for
    the strategy and for buy-and-hold, plus the round-trips-per-year turnover.
    """
    d = df.dropna(subset=["ret"]).copy()
    ret = d["ret"].to_numpy(dtype=float)
    season = d["in_season"].to_numpy(dtype=bool)
    n = len(d)
    n_years = n / 252.0

    # strategy: long off-season, flat in-season
    strat = np.where(season, 0.0, ret)
    # switches: a switch happens whenever in_season flips between consecutive days
    flips = int((season[1:] != season[:-1]).sum())
    cost = cost_bps * 1e-4
    # apply cost on flip days (turning the book on/off)
    strat_net = strat.copy()
    flip_idx = np.where(season[1:] != season[:-1])[0] + 1
    strat_net[flip_idx] -= cost

    def ann(r):
        gross = np.prod(1.0 + r)
        return float(gross ** (1.0 / n_years) - 1.0) if n_years > 0 else float("nan")

    return {
        "n_years": round(n_years, 2),
        "bah_ann": round(ann(ret) * 100, 2),
        "strat_gross_ann": round(ann(strat) * 100, 2),
        "strat_net_ann": round(ann(strat_net) * 100, 2),
        "flips_per_year": round(flips / n_years, 2),
    }


# ---------------------------------------------------------------------------
# Summarize -- compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(
    market: pd.DataFrame,
    insurers: pd.DataFrame | None = None,
    landfalls: pd.DataFrame | None = None,
    n_permutations: int = 5_000,
    seed: int = 283,
) -> dict:
    """One-stop summary dict mirroring the R = dict(...) in build_notebooks.py.

    ``market`` is required; ``insurers`` and ``landfalls`` are optional (event
    windows and the insurer arm are skipped if absent).
    """
    mkt = season_stats(market, n_permutations=n_permutations, seed=seed)
    out = {
        "n": mkt["n"],
        "n_in": mkt["n_in"],
        "n_off": mkt["n_off"],
        "mkt_mean_in": mkt["mean_in"],
        "mkt_mean_off": mkt["mean_off"],
        "mkt_spread": mkt["spread"],
        "mkt_welch_t": mkt["welch_t"],
        "mkt_hac_t": mkt["hac_t"],
        "mkt_perm_p": mkt["perm_p"],
        "mkt_ann_in": mkt["ann_in"],
        "mkt_ann_off": mkt["ann_off"],
    }
    if insurers is not None:
        ins = season_stats(insurers, n_permutations=n_permutations, seed=seed)
        out.update({
            "ins_mean_in": ins["mean_in"],
            "ins_mean_off": ins["mean_off"],
            "ins_spread": ins["spread"],
            "ins_welch_t": ins["welch_t"],
            "ins_hac_t": ins["hac_t"],
            "ins_perm_p": ins["perm_p"],
            "ins_ann_in": ins["ann_in"],
            "ins_ann_off": ins["ann_off"],
        })
        if landfalls is not None:
            ew = event_windows(insurers, market, landfalls)
            out.update({
                "ew_n": ew["n_events"],
                "ew_car_end": ew["car_end"],
                "ew_car_t": ew["car_t"],
            })
    return out
