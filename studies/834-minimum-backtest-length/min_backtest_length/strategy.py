"""The engine — Study 834 (Minimum Backtest Length / MinTRL).

The claim, at full strength (Bailey, Borwein, López de Prado & Zhu 2014, *Pseudo-Mathematics
and Financial Charlatanism*; the MinTRL formula from Bailey & López de Prado 2012, *The Sharpe
Ratio Efficient Frontier*): a Sharpe ratio estimated over a finite track record is a **noisy**
estimate, and the shorter the record the noisier it is. For a target annualised Sharpe ``SR``,
the **minimum track-record length** (MinTRL) needed to reject "true Sharpe <= 0" at confidence
``conf`` is, in observations,

    MinTRL_obs = 1 + [ 1 - g3 * sr + (g4 - 1)/4 * sr**2 ] * ( Z_conf / (sr - sr*) )**2

where ``sr`` is the **per-observation** Sharpe (``SR_ann / sqrt(freq)``), ``sr*`` the benchmark
per-observation Sharpe (0 here), ``g3``/``g4`` the skewness/kurtosis of the per-observation
returns, and ``Z_conf = Phi^-1(conf)``. In years, ``MinTRL_years = MinTRL_obs / freq``. For
Gaussian daily returns this collapses to the memorable rule of thumb

    MinTRL_years  ~  ( Z_conf / SR_ann )**2

so a Sharpe-1 idea needs ~2.7 years to clear a 95% bar, a Sharpe-0.5 idea ~10.8 years, and a
Sharpe-0.25 idea ~43 years. **Negative skew and fat tails lengthen it further.** Anything with a
backtest *shorter than its MinTRL* is statistically indistinguishable from luck.

The companion is the **Probabilistic Sharpe Ratio** (PSR): the probability that the true Sharpe
exceeds ``sr*`` given the observed Sharpe, moments and length. MinTRL is exactly the ``n`` that
makes ``PSR = conf``.

The disciplines:

* a **Monte-Carlo calibration** — on the null (true Sharpe 0) the PSR test fires at exactly its
  nominal false-positive rate ``1 - conf`` (the machinery is unbiased), while short backtests of
  that *worthless* world routinely post gaudy in-sample Sharpes by pure luck;
* a **positive control** — a genuinely high-Sharpe series, whose PSR detection rate climbs with
  the track length and only becomes reliable *past* the MinTRL (real edge is confirmable, but not
  from a short record);
* the standard desk **inference primitives** (one-sample / Welch / Newey-West *t*, Wilson
  interval) so every simulated rejection rate carries its own error bar.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Sharpe, PSR, MinTRL — the closed-form core
# --------------------------------------------------------------------------- #
def sharpe_ratio(ret: np.ndarray, freq: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of a per-observation return array (mean/sd * sqrt(freq))."""
    r = np.asarray(ret, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return float("nan")
    sd = r.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(freq))


def sample_moments(ret: np.ndarray) -> tuple[float, float]:
    """Sample (skewness, kurtosis) — population-moment convention (Gaussian kurtosis = 3)."""
    r = np.asarray(ret, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 4:
        return 0.0, 3.0
    m = r.mean()
    s = r.std(ddof=0)
    if s <= 0:
        return 0.0, 3.0
    z = (r - m) / s
    return float(np.mean(z ** 3)), float(np.mean(z ** 4))


def _z_conf(conf: float) -> float:
    return float(stats.norm.ppf(conf))


def probabilistic_sharpe_ratio(
    sr_ann: float,
    n_years: float,
    freq: int = TRADING_DAYS,
    sr_star_ann: float = 0.0,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Probability the true Sharpe exceeds ``sr_star_ann`` given the observed ``sr_ann``.

    PSR = Phi( (sr - sr*) * sqrt(n_obs - 1) / sqrt(1 - skew*sr + (kurt-1)/4 * sr**2) )
    with per-observation Sharpes ``sr = sr_ann/sqrt(freq)`` and ``n_obs = n_years*freq``.
    """
    n_obs = n_years * freq
    if n_obs < 2:
        return float("nan")
    sr = sr_ann / np.sqrt(freq)
    sr_star = sr_star_ann / np.sqrt(freq)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return float("nan")
    stat = (sr - sr_star) * np.sqrt(n_obs - 1.0) / np.sqrt(denom)
    return float(stats.norm.cdf(stat))


def min_trl_years(
    sr_ann: float,
    freq: int = TRADING_DAYS,
    sr_star_ann: float = 0.0,
    conf: float = 0.95,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Minimum track-record length **in years** to reject "true Sharpe <= sr_star" at ``conf``.

    MinTRL_obs = 1 + [1 - skew*sr + (kurt-1)/4 * sr**2] * (Z_conf/(sr - sr*))**2 ; /freq -> years.
    Returns +inf if the observed Sharpe does not beat the benchmark (no finite length suffices).
    """
    sr = sr_ann / np.sqrt(freq)
    sr_star = sr_star_ann / np.sqrt(freq)
    if sr <= sr_star:
        return float("inf")
    z = _z_conf(conf)
    adj = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    n_obs = 1.0 + adj * (z / (sr - sr_star)) ** 2
    return float(n_obs / freq)


def min_trl_curve(
    sr_grid,
    freq: int = TRADING_DAYS,
    conf: float = 0.95,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> np.ndarray:
    """Vectorised MinTRL (years) over a grid of annualised Sharpes."""
    sr = np.asarray(sr_grid, dtype=float) / np.sqrt(freq)
    z = _z_conf(conf)
    adj = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    n_obs = 1.0 + adj * (z / sr) ** 2
    out = n_obs / freq
    return np.where(np.asarray(sr_grid, dtype=float) > 0, out, np.inf)


def min_trl_for_power(
    sr_ann: float, freq: int = TRADING_DAYS, conf: float = 0.95, power: float = 0.95
) -> float:
    """Length (years) at which a *true*-``sr_ann`` strategy is **detected** with prob ``power``.

    MinTRL asks when an observed Sharpe *equal to the target* becomes significant (~50% of true
    draws clear it). Reliably *detecting* a true edge needs the extra power term: the one-sided
    Sharpe test rejects when ``SR_hat*sqrt(n_years) >= Z_conf``; with ``SR_hat ~ N(SR, 1/n_years)``
    the power is ``Phi(SR_ann*sqrt(n_years) - Z_conf)``, giving ``n_years = ((Z_conf +
    Z_power)/SR_ann)**2``. This is the honest, longer number the desk quotes alongside MinTRL.
    """
    if sr_ann <= 0:
        return float("inf")
    z = _z_conf(conf) + _z_conf(power)
    return float((z / sr_ann) ** 2)


# --------------------------------------------------------------------------- #
# Inference primitives (shared desk toolkit)
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


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
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
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Monte-Carlo over many backtests of a KNOWN world (vectorised)
# --------------------------------------------------------------------------- #
def _panel_sharpes(panel: np.ndarray, freq: int) -> np.ndarray:
    """Annualised Sharpe of every row (backtest) of a ``(n_sims, n_obs)`` panel — vectorised."""
    mean = panel.mean(axis=1)
    sd = panel.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mean / sd * np.sqrt(freq), np.nan)
    return sr


def _panel_moments(panel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (skewness, kurtosis) of a ``(n_sims, n_obs)`` panel — vectorised."""
    m = panel.mean(axis=1, keepdims=True)
    s = panel.std(axis=1, ddof=0, keepdims=True)
    z = np.where(s > 0, (panel - m) / s, 0.0)
    skew = np.mean(z ** 3, axis=1)
    kurt = np.mean(z ** 4, axis=1)
    return skew, kurt


def simulate(
    data_mod,
    sr_ann_true: float = 0.0,
    n_years: float = 2.0,
    freq: int = TRADING_DAYS,
    n_sims: int = 4000,
    dist: str = "normal",
    skew_target: float = -1.0,
    conf: float = 0.95,
    sr_star_ann: float = 0.0,
    seed: int = 834,
) -> dict:
    """Monte-Carlo ``n_sims`` backtests of a KNOWN world; measure Sharpe dispersion + PSR rejection.

    Returns the array of observed annualised Sharpes, the per-backtest PSR (vs ``sr_star_ann``,
    using each backtest's *own* measured moments), the fraction that "reject SR<=sr*" at ``conf``,
    and its Wilson band. On the null (``sr_ann_true = 0``) the reject fraction is the **false
    positive rate** (should be ~``1 - conf``); on a genuine world it is the **detection power**.
    """
    panel, truth = data_mod.synthetic_panel(
        sr_ann=sr_ann_true, n_sims=n_sims, n_years=n_years, freq=freq,
        dist=dist, skew_target=skew_target, seed=seed,
    )
    sr_obs = _panel_sharpes(panel, freq)
    skew, kurt = _panel_moments(panel)
    n_obs = n_years * freq
    sr = sr_obs / np.sqrt(freq)
    sr_star = sr_star_ann / np.sqrt(freq)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    with np.errstate(invalid="ignore"):
        stat = np.where(denom > 0, (sr - sr_star) * np.sqrt(n_obs - 1.0) / np.sqrt(denom), np.nan)
    psr = stats.norm.cdf(stat)
    reject = psr >= conf
    k = int(np.nansum(reject))
    lo, hi = wilson_interval(k, len(reject))
    return {
        "sr_ann_true": sr_ann_true,
        "n_years": n_years,
        "freq": freq,
        "n_sims": int(len(reject)),
        "sr_obs": sr_obs,
        "psr": psr,
        "reject_frac": float(k / len(reject)),
        "reject_lo": lo,
        "reject_hi": hi,
        "median_obs_sr": float(np.nanmedian(sr_obs)),
        "sd_obs_sr": float(np.nanstd(sr_obs, ddof=1)),
        "truth_skew": truth.skew,
        "truth_kurt": truth.kurt,
    }


def luck_prob(
    data_mod,
    threshold_sr: float = 1.0,
    n_years: float = 2.0,
    freq: int = TRADING_DAYS,
    n_sims: int = 4000,
    seed: int = 834,
) -> dict:
    """Fraction of backtests of a **worthless** (true Sharpe 0) world that post observed Sharpe
    >= ``threshold_sr`` — the "luck beats skill in a short window" number."""
    panel, _ = data_mod.synthetic_panel(
        sr_ann=0.0, n_sims=n_sims, n_years=n_years, freq=freq, seed=seed,
    )
    sr_obs = _panel_sharpes(panel, freq)
    k = int(np.nansum(sr_obs >= threshold_sr))
    lo, hi = wilson_interval(k, len(sr_obs))
    return {
        "threshold_sr": threshold_sr, "n_years": n_years,
        "frac": float(k / len(sr_obs)), "lo": lo, "hi": hi,
        "best_sr": float(np.nanmax(sr_obs)), "n_sims": int(len(sr_obs)),
    }


def power_curve(
    data_mod,
    sr_ann_true: float,
    year_grid,
    freq: int = TRADING_DAYS,
    n_sims: int = 4000,
    conf: float = 0.95,
    seed: int = 834,
) -> dict:
    """PSR detection rate of a **genuine** ``sr_ann_true`` world at each track length in years."""
    fracs, los, his = [], [], []
    for i, ny in enumerate(year_grid):
        s = simulate(data_mod, sr_ann_true=sr_ann_true, n_years=float(ny), freq=freq,
                     n_sims=n_sims, conf=conf, seed=seed + i)
        fracs.append(s["reject_frac"]); los.append(s["reject_lo"]); his.append(s["reject_hi"])
    return {
        "sr_ann_true": sr_ann_true, "years": np.asarray(year_grid, dtype=float),
        "reject_frac": np.asarray(fracs), "reject_lo": np.asarray(los),
        "reject_hi": np.asarray(his),
        "min_trl": min_trl_years(sr_ann_true, freq=freq, conf=conf),
        "min_trl_power": min_trl_for_power(sr_ann_true, freq=freq, conf=conf, power=conf),
    }
