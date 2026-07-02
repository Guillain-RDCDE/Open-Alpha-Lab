"""The engine and its honest controls — Study 590 (Sharpe-Hacking).

The claim, at full strength: you can inflate a strategy's *reported* Sharpe ratio with pure
financial engineering — none of which adds a cent of genuine return — and three levers do most of
the work:

1. **Return smoothing (illiquidity / stale marks).** Report an AR(1)-smoothed version of your
   returns (as an illiquid book with stale marks mechanically does — Getmansky, Lo & Makarov 2004).
   Smoothing barely changes the mean but *slashes the measured standard deviation* and injects
   positive autocorrelation, so the naive annualised Sharpe (mean/std × √252) balloons. It is a
   **pure measurement artefact**: the honest, autocorrelation-corrected Sharpe is unchanged.

2. **Naive leverage.** Scale every return by a constant *L*. Folk wisdom says "more leverage → more
   Sharpe". It is **false**: mean and standard deviation both scale by *L*, so the Sharpe is
   *identical*. Leverage buys you volatility and drawdown, not risk-adjusted return. The demo proves
   the null result — the honest lever that does nothing to the metric.

3. **Volatility targeting.** Rescale each day's exposure to hit a constant target vol. On a series
   with time-varying volatility this *can* nudge the Sharpe (it trims the fat-vol days), but the
   gain is small, fragile, and — when you then annualise a now-autocorrelated, non-iid book naively
   — partly another **measurement** inflation rather than real edge.

The honest disciplines:

- **The honest Sharpe** deflates the naive Sharpe for the autocorrelation the transforms inject
  (the Lo 2002 / GLM 2004 correction). It is the number that *cannot* be gamed by smoothing — the
  yardstick the whole study is measured against.
- **The baseline is not zero.** The raw tape already has an honest Sharpe (``base_sharpe``); the
  demo is about *inflating* an honest number, so we report the **inflation** (naive-after-transform
  minus honest-baseline), never a bare Sharpe.
- **A bootstrap null** on the smoothing inflation: how big an inflation would you expect from the
  honest engine on resampled raw returns? The observed inflation sits far in the tail — it is a
  systematic artefact, not sampling noise.
- **Costs.** Vol-targeting and leverage both trade; we charge a turnover cost so any "free" Sharpe
  gain is shown net.
- **A synthetic positive control.** When a *genuine* edge is planted (``alpha > 0``), the honest
  Sharpe rises with it (the metric tracks real skill) while smoothing still inflates the *naive*
  Sharpe by the same mechanical amount on top — proving the correction isolates real edge from
  measurement games.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# The two Sharpe measurements — the honest one is the yardstick
# ---------------------------------------------------------------------------
def sharpe_naive(ret: np.ndarray | pd.Series, periods: int = TRADING_DAYS) -> float:
    """The reported Sharpe everyone quotes: mean/std of daily returns × √periods.

    This is the number that gets gamed — it assumes iid returns, so it over-states risk-adjusted
    performance the instant returns are serially correlated (as smoothing makes them).
    """
    r = np.asarray(ret, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return 0.0
    sd = r.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


def _autocorr(r: np.ndarray, lag: int) -> float:
    if r.size <= lag + 1:
        return 0.0
    a = r[:-lag] - r.mean()
    b = r[lag:] - r.mean()
    denom = np.sum((r - r.mean()) ** 2)
    if denom <= 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def sharpe_honest_annual(ret: np.ndarray | pd.Series, periods: int = TRADING_DAYS) -> float:
    """Autocorrelation-corrected annualised Sharpe (Lo 2002 / Getmansky-Lo-Makarov 2004).

    Naive annualisation multiplies the per-period (daily) Sharpe by √periods, which is only right for
    **iid** returns. When returns are serially correlated (smoothing injects exactly this), the
    correct annualisation factor is

        eta(q) = q / sqrt( q + 2 * sum_{k=1..q-1} (q-k) * rho_k ),   q = periods

    where ``rho_k`` is the lag-*k* autocorrelation of the daily returns. For iid returns eta(q) = √q
    and this reduces to the naive Sharpe; positive autocorrelation makes the denominator larger than
    √q, deflating the inflated Sharpe back toward the truth. This is the number that **cannot be
    gamed by smoothing** — the study's yardstick. (Lags are capped at ~n/4 for a stable estimate.)
    """
    r = np.asarray(ret, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return 0.0
    sd = r.std(ddof=1)
    if sd <= 0:
        return 0.0
    per_period_sr = r.mean() / sd
    q = periods
    # cap the number of usable lags at ~ n/4 for a stable estimate
    max_lag = min(q, max(2, r.size // 4))
    rhos = [_autocorr(r, k) for k in range(1, max_lag)]
    denom = q + 2.0 * sum((q - k) * rhos[k - 1] for k in range(1, max_lag))
    if denom <= 0:
        return float(per_period_sr * np.sqrt(periods))
    eta = q / np.sqrt(denom)
    return float(per_period_sr * eta)


# ---------------------------------------------------------------------------
# The three Sharpe-gaming transforms — none adds genuine return
# ---------------------------------------------------------------------------
def smooth_returns(ret: np.ndarray | pd.Series, theta: float = 0.5) -> np.ndarray:
    """Report an AR(1)-smoothed (stale-mark) version of the returns.

    An illiquid book marks slowly: the reported return is a moving average of the true recent
    returns,

        r_reported_t = (1 - theta) * r_true_t + theta * r_reported_{t-1}

    This is the Getmansky-Lo-Makarov illiquidity model. It preserves the long-run mean almost
    exactly (it is a normalised weighted average) but **crushes the measured standard deviation** and
    injects positive autocorrelation — inflating the naive Sharpe while adding **zero** genuine
    return. ``theta = 0`` is a no-op; higher ``theta`` = smoother marks = more inflation.
    """
    r = np.asarray(ret, dtype=float)
    out = np.empty_like(r)
    prev = r[0] if r.size else 0.0
    for i in range(r.size):
        prev = (1.0 - theta) * r[i] + theta * prev
        out[i] = prev
    return out


def lever_returns(ret: np.ndarray | pd.Series, leverage: float = 2.0) -> np.ndarray:
    """Scale every return by a constant — the leverage that does *nothing* to the Sharpe.

    ``r_t -> leverage * r_t``. Mean and standard deviation both scale by ``leverage``, so the Sharpe
    is **identical**. The demo's null lever: it buys volatility and drawdown, not risk-adjusted
    return.
    """
    return np.asarray(ret, dtype=float) * float(leverage)


def vol_target_returns(ret: np.ndarray | pd.Series, target_ann_vol: float = 0.10,
                       lookback: int = 20, cap: float = 4.0) -> np.ndarray:
    """Rescale each day's exposure to hit a constant target volatility (a real, tradable transform).

    Exposure on day *t* is ``min(cap, target_daily_vol / trailing_vol_{t-1})`` using a trailing
    ``lookback``-day realised vol known *before* the day (no look-ahead), so the book targets a
    constant vol. On a series with time-varying vol this trims the high-vol days — a small, genuine
    variance-timing effect — but it also produces an autocorrelated, non-iid book whose *naive*
    Sharpe is partly a measurement artefact. Returns the vol-targeted daily returns.
    """
    r = np.asarray(ret, dtype=float)
    n = r.size
    target_daily = target_ann_vol / np.sqrt(TRADING_DAYS)
    exposure = np.ones(n)
    out = np.empty(n)
    for t in range(n):
        if t >= lookback:
            trail = r[t - lookback:t].std(ddof=1)
            e = min(cap, target_daily / trail) if trail > 0 else 1.0
        else:
            e = 1.0
        exposure[t] = e
        out[t] = e * r[t]
    return out


# ---------------------------------------------------------------------------
# The headline: how much each lever inflates the REPORTED Sharpe
# ---------------------------------------------------------------------------
def gaming_table(ret: np.ndarray | pd.Series, theta: float = 0.5, leverage: float = 3.0,
                 target_ann_vol: float = 0.10) -> pd.DataFrame:
    """The centrepiece: naive vs honest Sharpe for the raw series and each gamed version.

    For each lever we report the **naive** (reported) Sharpe, the **honest**
    (autocorrelation-corrected) Sharpe, and the *inflation* = naive − honest-baseline. A lever that
    only games the *measurement* shows a big naive gain and ~0 honest gain.
    """
    r = np.asarray(ret, dtype=float)
    base_naive = sharpe_naive(r)
    base_honest = sharpe_honest_annual(r)

    rows = []

    def add(name, series):
        sn = sharpe_naive(series)
        sh = sharpe_honest_annual(series)
        rows.append({
            "transform": name,
            "naive_sharpe": sn,
            "honest_sharpe": sh,
            "naive_inflation": sn - base_naive,
            "honest_inflation": sh - base_honest,
        })

    add("raw (baseline)", r)
    add(f"smoothed (theta={theta})", smooth_returns(r, theta))
    add(f"levered (L={leverage:g})", lever_returns(r, leverage))
    add(f"vol-targeted (tgt={target_ann_vol:.0%})", vol_target_returns(r, target_ann_vol))
    return pd.DataFrame(rows).set_index("transform")


# ---------------------------------------------------------------------------
# Bootstrap null on the smoothing inflation — is it noise, or a systematic artefact?
# ---------------------------------------------------------------------------
def smoothing_inflation_null(ret: np.ndarray | pd.Series, theta: float = 0.5,
                             n_boot: int = 2000, block: int = 20, seed: int = 590) -> dict:
    """Circular-block-bootstrap confidence bands on the smoothing inflation, naive vs honest.

    Smoothing an iid-in-the-mean series inflates the *naive* Sharpe **mechanically** — resampling
    the returns and re-smoothing reproduces the same inflation, so the interesting statistical
    question is not "is the inflation different from zero" (it always is) but "which metric is the
    artefact". We answer it with a moving-block bootstrap (preserving short-range structure) and read
    two intervals:

    - the **naive** inflation (naive Sharpe of the smoothed series − naive Sharpe of the raw series):
      a wide-but-firmly-positive band whose 95% CI sits **far above 0** — the reported Sharpe is
      reliably, systematically gamed;
    - the **honest** inflation (same, but with the autocorrelation-corrected Sharpe): a band that
      **straddles 0** — the honest metric is *not* fooled by smoothing.

    Returns the point estimates and the 2.5/97.5 percentile bands for each, plus the fraction of
    bootstrap draws whose honest inflation exceeds the naive point estimate (≈ 0 — the honest metric
    never mimics the naive game).
    """
    r = np.asarray(ret, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    obs_naive = sharpe_naive(smooth_returns(r, theta)) - sharpe_naive(r)
    obs_honest = sharpe_honest_annual(smooth_returns(r, theta)) - sharpe_honest_annual(r)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    naive_b = np.empty(n_boot)
    honest_b = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        rs = r[idx]
        sm = smooth_returns(rs, theta)
        naive_b[i] = sharpe_naive(sm) - sharpe_naive(rs)
        honest_b[i] = sharpe_honest_annual(sm) - sharpe_honest_annual(rs)
    return {
        "naive_inflation": float(obs_naive),
        "naive_ci": (float(np.percentile(naive_b, 2.5)), float(np.percentile(naive_b, 97.5))),
        "honest_inflation": float(obs_honest),
        "honest_ci": (float(np.percentile(honest_b, 2.5)), float(np.percentile(honest_b, 97.5))),
        "frac_honest_ge_naive": float(np.mean(honest_b >= obs_naive)),
    }


# ---------------------------------------------------------------------------
# Costs — vol-targeting and leverage trade; charge for it
# ---------------------------------------------------------------------------
def net_vol_target(ret: np.ndarray | pd.Series, target_ann_vol: float = 0.10,
                   lookback: int = 20, cost_bps: float = 2.0) -> dict:
    """Vol-targeting Sharpe, gross and net of a turnover cost on the exposure changes.

    Each day the exposure shifts, you trade ``|Δexposure|`` of NAV and pay ``cost_bps`` one-way.
    Returns gross/net naive Sharpe, gross/net honest Sharpe, and the average daily turnover.
    """
    r = np.asarray(ret, dtype=float)
    n = r.size
    target_daily = target_ann_vol / np.sqrt(TRADING_DAYS)
    exposure = np.ones(n)
    gross = np.empty(n)
    for t in range(n):
        if t >= lookback:
            trail = r[t - lookback:t].std(ddof=1)
            e = min(4.0, target_daily / trail) if trail > 0 else 1.0
        else:
            e = 1.0
        exposure[t] = e
        gross[t] = e * r[t]
    turnover = np.abs(np.diff(exposure, prepend=exposure[0]))
    cost = turnover * cost_bps * 1e-4
    net = gross - cost
    return {
        "gross_naive_sharpe": sharpe_naive(gross),
        "net_naive_sharpe": sharpe_naive(net),
        "gross_honest_sharpe": sharpe_honest_annual(gross),
        "net_honest_sharpe": sharpe_honest_annual(net),
        "turnover_per_day": float(turnover.mean()),
    }


# ---------------------------------------------------------------------------
# Robustness — how the smoothing inflation scales with theta (the smoothing knob)
# ---------------------------------------------------------------------------
def theta_sweep(ret: np.ndarray | pd.Series, thetas=(0.0, 0.2, 0.4, 0.6, 0.8)) -> pd.DataFrame:
    """Naive vs honest Sharpe as the smoothing intensity ``theta`` grows.

    The naive Sharpe climbs steeply with ``theta`` (more staleness → more inflation) while the honest
    Sharpe stays flat — the signature of a pure measurement game.
    """
    r = np.asarray(ret, dtype=float)
    rows = []
    for th in thetas:
        s = smooth_returns(r, th) if th > 0 else r
        rows.append({
            "theta": th,
            "naive_sharpe": sharpe_naive(s),
            "honest_sharpe": sharpe_honest_annual(s),
            "lag1_autocorr": _autocorr(s[np.isfinite(s)], 1),
        })
    return pd.DataFrame(rows).set_index("theta")


# ---------------------------------------------------------------------------
# Synthetic positive control — the honest Sharpe tracks REAL edge, not the game
# ---------------------------------------------------------------------------
def seed_robust_control(data_mod, alpha: float, theta: float = 0.5, n_seeds: int = 25,
                        n_days: int = 3000, base_seed: int = 590) -> dict:
    """Average the key numbers over ``n_seeds`` synthetic worlds with a planted ``alpha``.

    The house rule: any synthetic-dependent claim averages over ≥ 20 seeds so no single lucky RNG
    seed can manufacture a result. For each seed we build the tape, then report:

    - ``mean_honest_raw`` — honest Sharpe of the raw series (should rise with ``alpha``: the honest
      metric tracks real edge),
    - ``mean_naive_inflation`` — naive Sharpe of the smoothed series − naive Sharpe of the raw
      series (the measurement game; should be *positive and roughly constant* regardless of
      ``alpha`` — smoothing inflates the report the same way whether or not there is real edge),
    - ``mean_honest_inflation`` — honest Sharpe of the smoothed series − honest Sharpe of the raw
      series (should be ≈ 0: the honest metric is *not* gamed by smoothing).

    Returns the across-seed means and the t-stat of the honest-inflation (which must NOT clear +2 —
    the correction is not itself a source of spurious edge).
    """
    honest_raw, naive_infl, honest_infl = [], [], []
    for s in range(base_seed, base_seed + n_seeds):
        r, _ = data_mod.synthetic_series(n_days=n_days, alpha=alpha, seed=s)
        r = r.to_numpy()
        sm = smooth_returns(r, theta)
        honest_raw.append(sharpe_honest_annual(r))
        naive_infl.append(sharpe_naive(sm) - sharpe_naive(r))
        honest_infl.append(sharpe_honest_annual(sm) - sharpe_honest_annual(r))
    hi = np.array(honest_infl)
    hi_t = float(hi.mean() / (hi.std(ddof=1) / np.sqrt(len(hi)))) if hi.std(ddof=1) > 0 else 0.0
    return {
        "alpha": alpha,
        "mean_honest_raw": float(np.mean(honest_raw)),
        "mean_naive_inflation": float(np.mean(naive_infl)),
        "mean_honest_inflation": float(np.mean(honest_infl)),
        "honest_inflation_t": hi_t,
    }
