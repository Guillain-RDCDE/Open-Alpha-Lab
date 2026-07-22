"""Strategy + inference for Study 788 — Overnight / Intraday Tug of War.

The claim (Lou, Polk & Skouras 2019): sort a cross-section of stocks on their
**trailing overnight return**; the high-overnight names keep earning positive
returns *overnight* (persistence) while giving them back *intraday* (reversal),
and the low-overnight names do the opposite. A high-minus-low portfolio should
therefore show a **positive overnight leg** and a **negative intraday leg** — the
two legs pulling in opposite directions, the "tug of war."

This is a **cross-sectional** claim on individual names, distinct from:

* [01-overnight-anomaly](../../01-overnight-anomaly/) — the *aggregate / index-level*
  overnight-vs-intraday split (does the market as a whole earn its return at night?),
  not a cross-sectional sort;
* [640-gold-overnight](../../640-gold-overnight/) — the same night/day split on a
  *single asset* (gold), no cross-section;
* [116-power-hour](../../116-power-hour/) — an *intraday clock* effect (the last
  trading hour), not the overnight-vs-intraday cross-sectional tug.

Method:

* **Decompose** each name's daily OHLC into overnight (prev-close -> open) and
  intraday (open -> close) legs, via ``quantlab.decompose.decompose`` (the exact
  identity, no free parameters).
* **Point-in-time sort.** On each day ``t`` we rank the cross-section by the
  trailing overnight return over a formation window ending at ``t-1`` (known at
  the close of ``t-1``) and hold day ``t``. One documented execution lag: signal
  through close ``t-1`` -> hold the overnight+intraday legs of ``t``. The
  overnight leg is capturable because you enter at the close of ``t-1``, the
  instant the signal is fully known.
* **Legs of the H-L spread.** For each day we form the equal-weight top-minus-
  bottom fractile spread in the overnight leg and (separately) in the intraday
  leg. Persistence = mean overnight spread > 0; reversal = mean intraday spread
  < 0. Newey-West (HAC) *t* on each; Welch *t* on the top-vs-bottom pooled legs.
* **Costed timer.** An overnight-capture long-short pays one-way cost x NAV on
  every leg it turns over (entry at close, exit at open = 2 one-way trips/day),
  and the short book pays borrow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.decompose import decompose

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Night/day panels
# --------------------------------------------------------------------------- #
def leg_panels(panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose every name; return (overnight, intraday) return DataFrames
    (index = date, columns = ticker), aligned on the union of dates."""
    on, idc = {}, {}
    for sym, ohlc in panel.items():
        dec = decompose(ohlc)
        on[sym] = dec["r_overnight"]
        idc[sym] = dec["r_intraday"]
    return pd.DataFrame(on).sort_index(), pd.DataFrame(idc).sort_index()


def trailing_overnight_signal(on: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Formation signal: trailing-``window`` mean overnight return, per name.

    Value on row ``t`` uses overnight legs through ``t`` (inclusive). The sort in
    :func:`tug_spreads` shifts this by one day so a day-``t`` position is formed
    on information known at the close of ``t-1`` — the single execution lag.
    """
    return on.rolling(window, min_periods=window).mean()


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> H-L leg spreads
# --------------------------------------------------------------------------- #
def tug_spreads(
    on: pd.DataFrame,
    idc: pd.DataFrame,
    window: int = 21,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """Daily equal-weight top-minus-bottom fractile spread, per leg.

    On each day ``t`` names are ranked by the trailing overnight signal known at
    the close of ``t-1`` (one ``shift``). The top ``frac`` and bottom ``frac`` of
    names form the long and short books; we take the equal-weight mean of each
    book's day-``t`` **overnight** leg and (separately) day-``t`` **intraday**
    leg. Days with fewer than ``min_names`` ranked names are dropped.

    Columns: ``on_spread`` (long-short overnight), ``id_spread`` (long-short
    intraday), ``on_top`` / ``on_bot`` / ``id_top`` / ``id_bot`` (book means),
    ``n`` (names ranked that day). One row per trading day.
    """
    sig = trailing_overnight_signal(on, window).shift(1)  # known at close t-1
    rows = {}
    dates = on.index
    for t in dates:
        s = sig.loc[t].dropna()
        if len(s) < min_names:
            continue
        k = max(1, int(np.floor(len(s) * frac)))
        order = s.sort_values()
        bot = order.index[:k]
        top = order.index[-k:]
        on_t = on.loc[t]
        id_t = idc.loc[t]
        on_top = float(on_t[top].mean())
        on_bot = float(on_t[bot].mean())
        id_top = float(id_t[top].mean())
        id_bot = float(id_t[bot].mean())
        rows[t] = {
            "on_spread": on_top - on_bot,
            "id_spread": id_top - id_bot,
            "on_top": on_top, "on_bot": on_bot,
            "id_top": id_top, "id_bot": id_bot,
            "n": len(s),
        }
    return pd.DataFrame(rows).T.sort_index()


# --------------------------------------------------------------------------- #
# Inference primitives
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
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0 — the serial-
    correlation-robust statistic for a daily overlapping-signal return series."""
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
# The headline tug statistics
# --------------------------------------------------------------------------- #
def tug_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
    """Persistence (overnight spread), reversal (intraday spread) and the net.

    * ``on_*`` — mean day-``t`` overnight leg of the H-L spread and its NW / one-
      sample *t*; positive ⇒ high-past-overnight names keep earning overnight.
    * ``id_*`` — same for the intraday leg; negative ⇒ they reverse intraday.
    * ``cc_*`` — the close-to-close spread (≈ on + id), whether the tug nets to
      anything before costs.
    """
    on = spreads["on_spread"].to_numpy(dtype=float)
    idc = spreads["id_spread"].to_numpy(dtype=float)
    cc = on + idc
    return {
        "n_days": int(len(spreads)),
        "on_mean_bps": float(np.nanmean(on) * 1e4),
        "id_mean_bps": float(np.nanmean(idc) * 1e4),
        "cc_mean_bps": float(np.nanmean(cc) * 1e4),
        "on_t_nw": newey_west_t(on, nw_lags),
        "id_t_nw": newey_west_t(idc, nw_lags),
        "cc_t_nw": newey_west_t(cc, nw_lags),
        "on_t_1s": one_sample_t(on),
        "id_t_1s": one_sample_t(idc),
        # The single "tug" headline: persistence minus reversal (on - id), the
        # magnitude of the opposite-signs pull. Positive & significant = a tug.
        "tug_mean_bps": float(np.nanmean(on - idc) * 1e4),
        "tug_t_nw": newey_west_t(on - idc, nw_lags),
    }


def leg_pooled_welch(spreads: pd.DataFrame) -> dict:
    """Welch t of top-vs-bottom book means, per leg (pooled across days).

    A complementary read to the spread NW *t*: is the top book's overnight leg
    higher than the bottom book's, and its intraday leg lower?"""
    return {
        "on_welch_t": welch_t(spreads["on_top"].to_numpy(), spreads["on_bot"].to_numpy()),
        "id_welch_t": welch_t(spreads["id_top"].to_numpy(), spreads["id_bot"].to_numpy()),
        "on_top_bps": float(spreads["on_top"].mean() * 1e4),
        "on_bot_bps": float(spreads["on_bot"].mean() * 1e4),
        "id_top_bps": float(spreads["id_top"].mean() * 1e4),
        "id_bot_bps": float(spreads["id_bot"].mean() * 1e4),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the tug real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    on: pd.DataFrame,
    idc: pd.DataFrame,
    window: int = 21,
    frac: float = 0.3,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 788,
) -> dict:
    """Permutation placebo: keep the trailing-overnight sort, but read each day's
    forward overnight legs from a **column-permuted** panel — i.e. scramble which
    name's forward outcome is attached to which ranked name, breaking the
    signal->outcome link while preserving each day's cross-sectional distribution.

    p = share of permuted worlds whose overnight-spread mean is >= observed
    (a right-tail test — the persistence claim). Averaged over ``n_seeds`` x
    ``n_draws_per_seed`` permutations so no single stream decides it.
    """
    cols = list(on.columns)
    ncol = len(cols)
    sig = trailing_overnight_signal(on, window).shift(1)   # sort key, fixed
    obs = float(tug_spreads(on, idc, window, frac, min_names)["on_spread"].mean())

    # Precompute, per valid day, the integer COLUMN POSITIONS of the top and
    # bottom books and the day's overnight-leg row (numpy) — so each permutation
    # is a fast fancy-index mean, not a per-day pandas .loc.
    on_mat = on.to_numpy(dtype=float)
    pos_of = {c: i for i, c in enumerate(cols)}
    rows_idx, tops, bots = [], [], []
    row_lookup = {t: r for r, t in enumerate(on.index)}
    for t in on.index:
        s = sig.loc[t].dropna()
        if len(s) < min_names:
            continue
        k = max(1, int(np.floor(len(s) * frac)))
        order = s.sort_values()
        rows_idx.append(row_lookup[t])
        bots.append(np.array([pos_of[c] for c in order.index[:k]]))
        tops.append(np.array([pos_of[c] for c in order.index[-k:]]))
    rows_idx = np.asarray(rows_idx)

    # Vectorised: pad each day's (possibly ragged) top/bottom book to the max
    # width with a masked sentinel, so every permutation is two fancy-index
    # gathers + a masked mean over (D, k_max) — no per-day python loop.
    means = []
    if len(rows_idx):
        M = on_mat[rows_idx]                          # (D, ncol) forward legs
        kt = max(len(a) for a in tops)
        kb = max(len(a) for a in bots)

        def _pad(books, kmax):
            P = np.zeros((len(books), kmax), dtype=int)
            V = np.zeros((len(books), kmax), dtype=bool)   # valid mask
            for j, a in enumerate(books):
                P[j, :len(a)] = a
                V[j, :len(a)] = True
            return P, V

        TOP, TOPv = _pad(tops, kt)
        BOT, BOTv = _pad(bots, kb)
        rows_ar = np.arange(len(rows_idx))[:, None]

        def _masked_mean(pos, valid, perm):
            vals = M[rows_ar, perm[pos]]                  # (D, kmax)
            vals = np.where(valid, vals, np.nan)
            return np.nanmean(vals, axis=1)

        for seed in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed)
            for _ in range(n_draws_per_seed):
                perm = rng.permutation(ncol)
                top_v = _masked_mean(TOP, TOPv, perm)
                bot_v = _masked_mean(BOT, BOTv, perm)
                means.append(np.nanmean(top_v - bot_v))
    means = np.asarray(means)
    return {
        "obs_bps": obs * 1e4,
        "placebo_mean_bps": float(means.mean() * 1e4),
        "placebo_sd_bps": float(means.std(ddof=1) * 1e4),
        "p_value": float((means >= obs).mean()),
        "n_draws": len(means),
        "draws_bps": means * 1e4,
    }


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(
    spreads: pd.DataFrame,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
    years: float | None = None,
) -> dict:
    """Cost the overnight-capture long-short.

    The tradable version of the persistence leg: each day enter the H-L book at
    the close of ``t-1`` and unwind at the open of ``t`` (capturing the overnight
    leg), which turns the whole book over **twice** (in and out) — 2 x one-way
    cost x NAV of gross exposure. Gross exposure is 2x NAV (1 long + 1 short), so
    a full round trip costs ``2 legs x 2 sides x cost_bps``. The short book pays
    borrow, charged per calendar day.

    Returns gross/net per-day bps, one-sample *t* on the net series, annualised
    Sharpe and the net annualised return.
    """
    on = spreads["on_spread"].to_numpy(dtype=float)
    on = on[~np.isnan(on)]
    n = len(on)
    if years is None:
        years = n / TRADING_DAYS
    # 2 legs (long+short) x 2 sides (enter+exit) x one-way cost, per day.
    round_trip_cost = 4.0 * cost_bps / 1e4
    borrow_daily = (borrow_bps_yr / 1e4) / 365.0   # short book, per calendar day
    net = on - round_trip_cost - borrow_daily
    gross_mean = float(on.mean())
    net_mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net_mean / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")
    return {
        "n_days": n,
        "gross_bps": gross_mean * 1e4,
        "net_bps": net_mean * 1e4,
        "cost_bps_per_day": (round_trip_cost + borrow_daily) * 1e4,
        "ann_net_pct": net_mean * TRADING_DAYS * 100,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.DataFrame], window: int = 21,
                     frac: float = 0.3) -> dict:
    """Run the headline tug stats on a synthetic panel."""
    on, idc = leg_panels(panel)
    sp = tug_spreads(on, idc, window, frac)
    ts = tug_stats(sp)
    return {"on_mean_bps": ts["on_mean_bps"], "id_mean_bps": ts["id_mean_bps"],
            "on_t_nw": ts["on_t_nw"], "id_t_nw": ts["id_t_nw"],
            "tug_t_nw": ts["tug_t_nw"], "n_days": ts["n_days"]}
