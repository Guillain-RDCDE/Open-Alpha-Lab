"""Spectral core — the periodogram, and the **red-noise significance test** that decides
whether a peak is a cycle or a shape in the noise.

This is the scientific heart of the study, and it is one idea: a financial level is not
white noise. It is *red* — slow, persistent, autocorrelated — and a red-noise process,
fed through a periodogram, sprouts tall, broad, **random** peaks all on its own. The naked
eye (and the cycle-drawing tool) reads those peaks as "an 80-day cycle". So the only honest
question is not *"is there a peak at 80 days?"* (there always is, somewhere) but:

    **Does the peak at the claimed period stand above what AR(1) red noise produces by
    chance, at the same sample length?**

We answer it the standard climatology way (Mann & Lees 1996; Torrence & Compo 1998):
fit an AR(1) null to the series, Monte-Carlo thousands of surrogate red-noise series of the
same length, and read off the per-frequency quantile **envelope**. A real cycle pokes above
the 95th/99th-percentile envelope; a phantom sits inside it. Everything is pure NumPy,
deterministic given a seed, and the *same* pre-processing is applied to the data and every
surrogate, so the comparison is apples to apples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Cycle theory on the VIX lives in the weeks-to-months band; the tweet's claims are at
# ~40 and ~80 sessions (and stocks' "20-week" ≈ 100 sessions, "4-year" ≈ 1000). We default
# the search to that band so we test the claim on its own turf, not on spurious far tails.
DEFAULT_BAND = (15.0, 400.0)   # sessions


# --------------------------------------------------------------------------- #
# Pre-processing — linear detrend + taper, applied identically to data & surrogates
# --------------------------------------------------------------------------- #

def _detrend(x: np.ndarray) -> np.ndarray:
    """Remove the least-squares line. A trend is power at period ~∞ that would leak into
    long-period bins and fake a low-frequency 'cycle'; we kill it before transforming."""
    n = x.size
    t = np.arange(n, dtype=float)
    b, a = np.polyfit(t, x, 1)
    return x - (b * t + a)


def _taper_window(n: int, frac: float = 0.1) -> np.ndarray:
    """Split-cosine (Tukey) window of length ``n`` — softens the record's ends so the FFT
    doesn't read the hard truncation as broadband power (spectral leakage)."""
    w = np.ones(n)
    m = int(frac * n)
    if m > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
        w[:m] = ramp
        w[-m:] = ramp[::-1]
    return w


def _taper(x: np.ndarray, frac: float = 0.1) -> np.ndarray:
    """Apply :func:`_taper_window` to ``x`` (data and surrogates alike, so it can't bias)."""
    return x * _taper_window(x.size, frac)


def periodogram(x, detrend: bool = True, taper: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(periods, power)`` for a real series via the FFT.

    ``power`` is the squared FFT magnitude per frequency (the raw periodogram); ``periods``
    is ``1/frequency`` in sessions. The zero-frequency (mean) bin is dropped. This exact
    routine is what both the data and every red-noise surrogate pass through, so their
    spectra are directly comparable bin for bin.
    """
    xv = np.asarray(x, dtype=float)
    xv = xv[np.isfinite(xv)]
    if detrend:
        xv = _detrend(xv)
    if taper:
        xv = _taper(xv)
    n = xv.size
    spec = np.fft.rfft(xv)
    power = (np.abs(spec) ** 2) / n
    freqs = np.fft.rfftfreq(n)
    # Drop f=0 (mean); keep the rest. period = 1/f in sessions.
    periods = 1.0 / freqs[1:]
    return periods, power[1:]


# --------------------------------------------------------------------------- #
# The AR(1) red-noise null
# --------------------------------------------------------------------------- #

def ar1_rho(x) -> float:
    """Lag-1 autocorrelation — the single parameter of the red-noise null."""
    xv = np.asarray(x, dtype=float)
    xv = xv[np.isfinite(xv)]
    xv = xv - xv.mean()
    denom = float(np.sum(xv * xv))
    if denom <= 0:
        return 0.0
    return float(np.sum(xv[1:] * xv[:-1]) / denom)


def _simulate_ar1(rho: float, sd: float, n: int, rng: np.random.Generator) -> np.ndarray:
    eps = rng.standard_normal(n) * sd * np.sqrt(max(1.0 - rho**2, 1e-9))
    y = np.empty(n)
    y[0] = rng.standard_normal() * sd
    for i in range(1, n):
        y[i] = rho * y[i - 1] + eps[i]
    return y


def _surrogate_powers(
    rho: float, sd: float, n: int, n_sim: int,
    detrend: bool, taper: bool, rng: np.random.Generator,
) -> np.ndarray:
    """Periodogram power for ``n_sim`` AR(1) surrogates at once — the vectorised null.

    The simulation is inherently sequential in time, but vectorised *across* surrogates: one
    loop over ``n`` steps, each a single NumPy op on the whole ``n_sim``-vector. Detrend, taper
    and FFT then run batched along the frequency axis. Identical pre-processing to
    :func:`periodogram`, so data and surrogates stay comparable bin for bin. Returns an
    ``(n_sim, nfreq)`` array with the f=0 column dropped. This is what keeps the Monte-Carlo
    fast enough for CI; the per-frequency RNG stream is fixed by ``rng``.
    """
    scale = sd * np.sqrt(max(1.0 - rho**2, 1e-9))
    eps = rng.standard_normal((n_sim, n)) * scale
    y = np.empty((n_sim, n))
    y[:, 0] = rng.standard_normal(n_sim) * sd
    for t in range(1, n):
        y[:, t] = rho * y[:, t - 1] + eps[:, t]

    if detrend:
        idx = np.arange(n, dtype=float)
        tc = idx - idx.mean()
        slope = (y * tc).sum(axis=1) / float((tc * tc).sum())
        intercept = y.mean(axis=1) - slope * idx.mean()
        y = y - (slope[:, None] * idx[None, :] + intercept[:, None])
    if taper:
        y = y * _taper_window(n)[None, :]

    spec = np.fft.rfft(y, axis=1)
    power = (np.abs(spec) ** 2) / n
    return power[:, 1:]   # drop f=0 to match periodogram()


def red_noise_envelope(
    x,
    n_sim: int = 2000,
    quantiles=(0.95, 0.99),
    detrend: bool = True,
    taper: bool = True,
    seed: int = 0,
) -> dict:
    """Monte-Carlo the AR(1) null and return the per-frequency power **envelope**.

    Fits ``rho`` and the variance of ``x``, simulates ``n_sim`` surrogate red-noise series
    of the *same length*, runs each through the identical :func:`periodogram`, and takes the
    requested ``quantiles`` of power at every frequency bin. A data peak that exceeds the
    0.95 (resp. 0.99) curve is significant at 5% (resp. 1%) **at that frequency** against
    red noise. Returns ``{periods, data_power, rho, envelope: {q: curve}}``.
    """
    xv = np.asarray(x, dtype=float)
    xv = xv[np.isfinite(xv)]
    n = xv.size
    rho = ar1_rho(xv)
    sd = xv.std(ddof=1)
    rng = np.random.default_rng(seed)

    periods, data_power = periodogram(xv, detrend=detrend, taper=taper)
    sims = _surrogate_powers(rho, sd, n, n_sim, detrend, taper, rng)

    envelope = {q: np.quantile(sims, q, axis=0) for q in quantiles}
    return {
        "periods": periods,
        "data_power": data_power,
        "rho": float(rho),
        "n_sim": int(n_sim),
        "envelope": envelope,
        "_sims": sims,   # retained for band-wise p-values (see test_period)
    }


def significant_peaks(env: dict, q: float = 0.95, band=DEFAULT_BAND) -> pd.DataFrame:
    """Periods inside ``band`` where the data periodogram exceeds the ``q`` envelope.

    A local-maximum filter avoids reporting every bin of one fat peak. Columns:
    ``period, power, envelope, ratio`` (ratio = power / envelope, >1 ⇒ significant), sorted
    by descending ratio — the peaks a cycle theorist would point at, with the honest
    significance attached.
    """
    periods = env["periods"]
    power = env["data_power"]
    curve = env["envelope"][q]
    lo, hi = band
    in_band = (periods >= lo) & (periods <= hi)

    rows = []
    idx = np.where(in_band)[0]
    for i in idx:
        is_local_max = (
            (i == 0 or power[i] >= power[i - 1])
            and (i == power.size - 1 or power[i] >= power[i + 1])
        )
        if is_local_max and power[i] > curve[i]:
            rows.append({
                "period": float(periods[i]),
                "power": float(power[i]),
                "envelope": float(curve[i]),
                "ratio": float(power[i] / curve[i]) if curve[i] > 0 else np.inf,
            })
    df = pd.DataFrame(rows)
    return df.sort_values("ratio", ascending=False).reset_index(drop=True) if len(df) else df


def test_period(
    x,
    target_period: float,
    band_frac: float = 0.15,
    n_sim: int = 2000,
    detrend: bool = True,
    taper: bool = True,
    seed: int = 0,
) -> dict:
    """Monte-Carlo p-value for a cycle *at a claimed period* against AR(1) red noise.

    The honest version of "is there an 80-day cycle?". Define a band ``target_period·(1 ±
    band_frac)`` — wide enough that a real cycle whose period the theorist eyeballed within
    15% still counts (and which therefore *charges* for that searching freedom). Take the
    peak power in that band; then for each red-noise surrogate take *its* peak power in the
    same band. The p-value is the fraction of surrogates whose in-band peak meets or beats
    the data's. Small p ⇒ a peak this tall is hard for red noise to fake ⇒ a real cycle.

    Returns ``{target_period, band, data_peak_power, p_value, rho, n_sim}``.
    """
    xv = np.asarray(x, dtype=float)
    xv = xv[np.isfinite(xv)]
    n = xv.size
    rho = ar1_rho(xv)
    sd = xv.std(ddof=1)
    rng = np.random.default_rng(seed)

    periods, power = periodogram(xv, detrend=detrend, taper=taper)
    lo = target_period * (1 - band_frac)
    hi = target_period * (1 + band_frac)
    in_band = (periods >= lo) & (periods <= hi)
    if not in_band.any():
        return {"target_period": target_period, "band": (lo, hi), "data_peak_power": np.nan,
                "p_value": np.nan, "rho": float(rho), "n_sim": int(n_sim)}
    data_peak = float(power[in_band].max())

    sims = _surrogate_powers(rho, sd, n, n_sim, detrend, taper, rng)
    sim_band_peaks = sims[:, in_band].max(axis=1)
    n_ge = int(np.sum(sim_band_peaks >= data_peak))
    p_value = (n_ge + 1) / (n_sim + 1)   # add-one: never reports an impossible exact 0
    return {
        "target_period": float(target_period),
        "band": (float(lo), float(hi)),
        "data_peak_power": data_peak,
        "p_value": float(p_value),
        "rho": float(rho),
        "n_sim": int(n_sim),
    }
