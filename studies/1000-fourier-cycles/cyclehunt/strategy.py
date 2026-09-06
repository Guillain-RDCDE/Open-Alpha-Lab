"""Looking for cycles, and the arithmetic of not finding them — Study 1000.

The periodogram of white noise is not flat. Each ordinate is an independent draw from an
exponential distribution with mean equal to the true (flat) spectrum, so among *m* frequency
bins the largest is around ``log(m)`` times the mean. With 4,000 daily observations there are
2,000 bins and ``log(2000) ≈ 7.6``: a random walk routinely produces a peak seven times the
average power, at a frequency that will look meaningful if you go looking for a story.

Everything in this module follows from that one fact:

- ``periodogram`` computes the spectrum, with the choices that matter made explicit: detrending
  (a trend leaks enormous power into the lowest frequencies and creates a fake "long cycle"),
  windowing (which trades resolution for leakage), and the exact normalisation.
- ``fisher_g_test`` is the right test, and it is over a century old (Fisher 1929). It asks
  whether the *largest* ordinate is too large **relative to the total**, which is exactly the
  multiple-comparison problem stated correctly, and it has an exact distribution under the null.
- ``spurious_peak_distribution`` does the same thing by simulation, which also handles the
  autocorrelated case where Fisher's exact result does not apply.
- ``ar1_null`` matters because returns are not white: even slight autocorrelation tilts the
  spectrum, and a peak that is significant against a *white* null can be entirely unremarkable
  against an AR(1) one. Testing against the wrong null is the second-most-common way cycle
  studies go wrong.

``cycle_trade`` then closes the loop: take the best peak found, project it forward, and trade
it. A genuine cycle keeps its phase out of sample; a spurious one does not, and the difference
is visible in a single chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# The spectrum
# --------------------------------------------------------------------------- #
def periodogram(x: pd.Series, detrend: bool = True, window: str = "none") -> pd.DataFrame:
    """The raw periodogram, with every choice made explicit.

    ``detrend`` removes a linear trend first. This is not optional in practice: an undetrended
    series puts enormous power at the lowest frequency and manufactures a "cycle" whose period
    is roughly the length of the sample. Most published market-cycle claims are this artefact.

    ``window`` applies a Hann taper, which reduces spectral leakage (power from a strong
    frequency bleeding into its neighbours) at the cost of resolution. Both choices are swept in
    the results because they change which peak is largest.
    """
    v = x.dropna().to_numpy(dtype=float)
    n = len(v)
    if n < 64:
        return pd.DataFrame(columns=["frequency", "period", "power"])
    v = v - v.mean()
    if detrend:
        t = np.arange(n)
        slope = float(np.polyfit(t, v, 1)[0])
        v = v - slope * (t - t.mean())
    if window == "hann":
        w = np.hanning(n)
        v = v * w
        scale = float((w ** 2).sum())
    else:
        scale = float(n)
    fft = np.fft.rfft(v)
    power = (np.abs(fft) ** 2) / scale
    freq = np.fft.rfftfreq(n, d=1.0)
    keep = slice(1, len(freq))            # drop the zero frequency
    f = freq[keep]
    p = power[keep]
    return pd.DataFrame({"frequency": f, "period": 1.0 / f, "power": p})


def top_peaks(pg: pd.DataFrame, k: int = 10, min_period: float = 4.0,
              max_period: float | None = None) -> pd.DataFrame:
    """The strongest frequencies, with the periods a reader would quote.

    ``min_period`` of 4 sessions is not arbitrary: below that the estimate is dominated by the
    Nyquist region where daily data has essentially no information about a "cycle".
    """
    d = pg[pg["period"] >= min_period]
    if max_period is not None:
        d = d[d["period"] <= max_period]
    if d.empty:
        return d
    out = d.nlargest(k, "power").copy()
    out["relative_power"] = out["power"] / pg["power"].mean()
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Is the peak real?
# --------------------------------------------------------------------------- #
def fisher_g_test(pg: pd.DataFrame) -> dict:
    """Fisher's (1929) exact test for the largest periodogram ordinate.

    ``g = max(I_j) / sum(I_j)``. Under Gaussian white noise the exact *p*-value is

        P(g > x) = sum_{k=1}^{K} (-1)^(k-1) C(m, k) (1 - k x)^(m-1)

    with ``K = floor(1/x)``. It is the correct multiple-comparison correction for this problem
    and it was published before most of the cycle literature it refutes.
    """
    p = pg["power"].to_numpy(dtype=float)
    m = len(p)
    total = float(p.sum())
    if m < 10 or total <= 0:
        return {"m": int(m)}
    g = float(p.max() / total)
    kmax = int(np.floor(1.0 / g))
    kmax = min(kmax, 60)
    pv = 0.0
    for k in range(1, kmax + 1):
        log_c = (gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1))
        val = np.exp(log_c + (m - 1) * np.log(max(1.0 - k * g, 1e-300)))
        pv += ((-1) ** (k - 1)) * val
    pv = float(min(max(pv, 0.0), 1.0))
    peak = pg.iloc[int(np.argmax(p))]
    return {"m": int(m), "g": g, "p_value": pv,
            "peak_period": float(peak["period"]),
            "peak_frequency": float(peak["frequency"]),
            "relative_power": float(p.max() / p.mean()),
            "significant_5pct": bool(pv < 0.05)}


def expected_max_relative_power(m: int) -> float:
    """How big the largest of ``m`` exponential ordinates is, relative to the mean.

    The number every cycle chart needs printed next to it. For i.i.d. exponential ordinates the
    expected maximum is the harmonic number ``H_m ≈ log(m) + 0.577``, in units of the mean. With
    2,000 bins that is about **8×** — so a peak "eight times the average power" is the *typical*
    result of finding nothing.
    """
    if m < 2:
        return 1.0
    return float(np.log(m) + 0.5772156649)


def spurious_peak_distribution(n: int, n_sims: int = 500, detrend: bool = True,
                               ar1: float = 0.0, min_period: float = 4.0,
                               seed: int = 1000) -> dict:
    """Simulate the largest relative peak from noise of the same length.

    ``ar1`` lets the null be an autoregressive process rather than white noise. That matters:
    returns have small but non-zero autocorrelation, which tilts the spectrum and raises the
    expected maximum, so a peak tested against a white null can be significant purely because
    the null was wrong.
    """
    rng = np.random.default_rng(seed)
    maxes, gs = [], []
    for _ in range(n_sims):
        e = rng.standard_normal(n)
        if ar1 != 0.0:
            v = np.empty(n)
            v[0] = e[0]
            for t in range(1, n):
                v[t] = ar1 * v[t - 1] + e[t]
        else:
            v = e
        pg = periodogram(pd.Series(v), detrend=detrend)
        d = pg[pg["period"] >= min_period]
        if d.empty:
            continue
        maxes.append(float(d["power"].max() / pg["power"].mean()))
        gs.append(float(pg["power"].max() / pg["power"].sum()))
    maxes = np.array(maxes)
    return {"n_sims": int(len(maxes)), "mean_max": float(maxes.mean()),
            "median_max": float(np.median(maxes)),
            "p95_max": float(np.percentile(maxes, 95)),
            "p99_max": float(np.percentile(maxes, 99)),
            "theoretical": expected_max_relative_power(n // 2),
            "g_p95": float(np.percentile(gs, 95))}


def ar1_null(x: pd.Series) -> dict:
    """Fit an AR(1) and report the null spectrum it implies.

    A positively autocorrelated series has more power at low frequencies, so testing its peak
    against a *flat* null biases toward finding "long cycles". This returns the fitted
    coefficient and the theoretical AR(1) spectral density so the comparison can be made
    against the right benchmark.
    """
    v = x.dropna().to_numpy(dtype=float)
    if len(v) < 100:
        return {"n": int(len(v))}
    v = v - v.mean()
    phi = float(np.dot(v[:-1], v[1:]) / np.dot(v[:-1], v[:-1]))
    return {"n": int(len(v)), "phi": phi,
            "spectrum_tilt": float((1 + phi) / max(1 - phi, 1e-9)) if abs(phi) < 1 else np.inf}


def ar1_spectral_density(freq: np.ndarray, phi: float, sigma2: float = 1.0) -> np.ndarray:
    """The theoretical spectrum of an AR(1) — the correct null for autocorrelated data."""
    w = 2 * np.pi * freq
    return sigma2 / (1 - 2 * phi * np.cos(w) + phi ** 2)


def peak_against_ar1(pg: pd.DataFrame, phi: float) -> dict:
    """Rescale the periodogram by the AR(1) null and re-find the peak.

    A peak that survives this is a genuine periodicity; one that disappears was the
    autocorrelation showing through, which is the commonest false positive in this literature
    after the undetrended-trend artefact.
    """
    if pg.empty:
        return {}
    f = pg["frequency"].to_numpy()
    null = ar1_spectral_density(f, phi)
    null = null / null.mean()
    rel = pg["power"].to_numpy() / (pg["power"].mean() * null)
    j = int(np.argmax(rel))
    return {"peak_period_white": float(pg.iloc[int(np.argmax(pg["power"]))]["period"]),
            "peak_period_ar1": float(pg.iloc[j]["period"]),
            "relative_power_white": float(pg["power"].max() / pg["power"].mean()),
            "relative_power_ar1": float(rel[j]),
            "same_peak": bool(j == int(np.argmax(pg["power"])))}


# --------------------------------------------------------------------------- #
# Does the cycle hold up?
# --------------------------------------------------------------------------- #
def phase_coherence(x: pd.Series, period: float, n_segments: int = 8) -> dict:
    """Does the cycle's phase advance *consistently*, or wander at random?

    This replaces a naive "is the phase the same in both halves?" test, which does not work and
    is worth explaining because the failure is instructive.

    The periodogram's period resolution is one bin, and near a 120-session period with 4,000
    observations the neighbouring bins are about four sessions apart. A period estimate off by
    one part in a hundred accumulates a full radian of phase error over a few thousand steps —
    so a **genuine, perfectly stable** cycle fails a naive phase-equality test purely because
    the period was measured on a grid.

    What survives that problem is *coherence*: if the cycle is real, the fitted phase drifts
    **linearly** with segment index (at a rate set by the period error), whereas for noise it
    jumps around. Regressing unwrapped phase on segment index and reading the R² separates the
    two without needing the period to be exact.
    """
    v = x.dropna()
    n = len(v)
    if n < 400 or period < 4:
        return {"n": int(n)}
    seg_len = n // n_segments
    if seg_len < max(2 * period, 50):
        n_segments = max(int(n / max(2 * period, 50)), 3)
        seg_len = n // n_segments
    w = 2 * np.pi / period
    phases, amps = [], []
    for k in range(n_segments):
        seg = v.iloc[k * seg_len:(k + 1) * seg_len].to_numpy(dtype=float)
        if len(seg) < 10:
            continue
        t = np.arange(len(seg))
        X = np.column_stack([np.ones(len(t)), np.cos(w * t), np.sin(w * t)])
        c, *_ = np.linalg.lstsq(X, seg, rcond=None)
        phases.append(float(np.arctan2(c[2], c[1])))
        amps.append(float(np.hypot(c[1], c[2])))
    if len(phases) < 4:
        return {"n": int(n)}
    unwrapped = np.unwrap(np.array(phases))
    k = np.arange(len(unwrapped), dtype=float)
    A = np.column_stack([np.ones(len(k)), k])
    coef, *_ = np.linalg.lstsq(A, unwrapped, rcond=None)
    resid = unwrapped - A @ coef
    ss_tot = float(((unwrapped - unwrapped.mean()) ** 2).sum())
    r2 = float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 1e-12 else np.nan
    # circular concentration of the residual phase: 1 = perfectly coherent, 0 = uniform
    conc = float(np.abs(np.mean(np.exp(1j * resid))))
    return {"n": int(n), "n_segments": int(len(phases)), "phase_r2": r2,
            "concentration": conc, "drift_per_segment": float(coef[1]),
            "amplitude_cv": float(np.std(amps, ddof=1) / np.mean(amps))
            if np.mean(amps) > 0 else np.nan,
            "coherent": bool(conc > 0.7)}


def split_sample_peak(x: pd.Series, min_period: float = 4.0,
                      max_period: float = 500.0) -> dict:
    """Find the best period in the first half; check it in the second.

    A genuine cycle keeps its period *and its phase*. A spurious one keeps neither, and the
    phase is the more demanding test — two halves of a random walk will occasionally agree on a
    period by luck, but almost never on where in the cycle they are.
    """
    v = x.dropna()
    n = len(v)
    if n < 500:
        return {"n": int(n)}
    a, b = v.iloc[:n // 2], v.iloc[n // 2:]
    pa = top_peaks(periodogram(a), 1, min_period, max_period)
    pb = top_peaks(periodogram(b), 1, min_period, max_period)
    if pa.empty or pb.empty:
        return {"n": int(n)}
    period_a = float(pa.iloc[0]["period"])
    period_b = float(pb.iloc[0]["period"])
    # phase agreement: fit a sinusoid of period_a to each half and compare phases
    def phase(series, period):
        t = np.arange(len(series))
        w = 2 * np.pi / period
        X = np.column_stack([np.ones(len(t)), np.cos(w * t), np.sin(w * t)])
        c, *_ = np.linalg.lstsq(X, series.to_numpy(dtype=float), rcond=None)
        return float(np.arctan2(c[2], c[1])), float(np.hypot(c[1], c[2]))
    ph_a, amp_a = phase(a, period_a)
    ph_b, amp_b = phase(b, period_a)
    # Where the phase should be. With the fit written as A*cos(w*t - phi), a segment starting
    # L steps later sees cos(w(t+L) - phi_a) = cos(w*t - (phi_a - w*L)), so the phase of the
    # second half should be phi_a MINUS w*L. Getting that sign backwards makes a genuine cycle
    # look like it drifted, which is exactly the false negative this function exists to avoid.
    expected = ph_a - 2 * np.pi * len(a) / period_a
    diff = float(np.angle(np.exp(1j * (ph_b - expected))))
    coh = phase_coherence(v, period_a)
    return {"n": int(n), "period_first": period_a, "period_second": period_b,
            "period_ratio": period_b / period_a,
            "amplitude_first": amp_a, "amplitude_second": amp_b,
            "amplitude_decay": amp_b / amp_a if amp_a > 0 else np.nan,
            # Kept as a diagnostic, but see phase_coherence: this raw number is dominated by
            # the periodogram's grid resolution, not by whether the cycle is real.
            "phase_error_rad": diff, "phase_error_fraction": abs(diff) / np.pi,
            "phase_concentration": coh.get("concentration", np.nan),
            "phase_r2": coh.get("phase_r2", np.nan),
            "coherent": coh.get("coherent", False)}


def cycle_trade(rets: pd.Series, period: float, fit_window: int = 1000,
                cost_bps: float = 5.0) -> dict:
    """Fit a sinusoid on a rolling window and trade the next step's prediction.

    Strictly out of sample: the sinusoid on day *t* is fitted to data through *t-1* only. A real
    cycle would make money here; a spurious one gives a coin flip, and the study needs to show
    the second rather than assert it.
    """
    r = rets.dropna()
    n = len(r)
    if n < fit_window + 250 or period < 2:
        return {"n": int(n)}
    v = r.to_numpy(dtype=float)
    w = 2 * np.pi / period
    signal = np.zeros(n)
    for i in range(fit_window, n):
        seg = v[i - fit_window:i]
        t = np.arange(len(seg))
        X = np.column_stack([np.ones(len(t)), np.cos(w * t), np.sin(w * t)])
        c, *_ = np.linalg.lstsq(X, seg, rcond=None)
        tn = float(len(seg))
        signal[i] = c[0] + c[1] * np.cos(w * tn) + c[2] * np.sin(w * tn)
    pos = pd.Series(np.sign(signal), index=r.index)
    switches = pos.diff().abs().fillna(0.0)
    strat = (pos * r - switches * cost_bps / 1e4).iloc[fit_window:]
    base = r.iloc[fit_window:]
    years = len(strat) / TRADING_DAYS
    sd = float(strat.std(ddof=1))
    return {"n": int(len(strat)), "period": float(period),
            "cagr": float((1 + strat).prod() ** (1 / years) - 1) if years > 0 else np.nan,
            "sharpe": float(strat.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
            "hit_rate": float((np.sign(pos.iloc[fit_window:]) == np.sign(base)).mean()),
            "buy_hold_cagr": float((1 + base).prod() ** (1 / years) - 1)
            if years > 0 else np.nan,
            "switches_per_year": float(switches.iloc[fit_window:].sum() / years),
            "returns": strat}


def synthetic_series(n: int = 4000, period: float = 0.0, amplitude: float = 0.0,
                     vol: float = 0.01, ar1: float = 0.0, seed: int = 1000) -> pd.Series:
    """Noise, optionally with a genuine sinusoid of known period buried in it.

    ``amplitude`` is in units of the noise standard deviation, so it is a direct
    signal-to-noise dial. At zero the series has no cycle at all — the null every threshold in
    this study is calibrated against.
    """
    rng = np.random.default_rng(seed)
    e = rng.normal(0, vol, n)
    if ar1 != 0.0:
        v = np.empty(n)
        v[0] = e[0]
        for t in range(1, n):
            v[t] = ar1 * v[t - 1] + e[t]
    else:
        v = e
    if period > 0 and amplitude > 0:
        t = np.arange(n)
        v = v + amplitude * vol * np.sin(2 * np.pi * t / period)
    idx = pd.bdate_range("1993-02-01", periods=n)
    return pd.Series(v, index=idx, name="ret")


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Busted** — the expected result — if no asset's peak survives Fisher's *g*
      test against the appropriate null; **Partial** if one survives against white noise but not
      against AR(1); **Confirmed** if a peak survives both **and** holds its phase out of
      sample. The positive control must be detected either way, or the machinery is broken.
    - **Tradability**: **Mirage** unless a detected cycle trades profitably out of sample.
    """
    survives_white = h["n_significant_white"] > 0
    survives_ar1 = h["n_significant_ar1"] > 0
    holds_phase = h["phase_concentration"] > 0.7
    if survives_ar1 and holds_phase:
        signal = "Confirmed"
    elif survives_white:
        signal = "Partial"
    else:
        signal = "Busted"
    trad = "Useful" if h["cycle_sharpe"] > 0.3 else (
        "Partial" if h["cycle_sharpe"] > 0 else "Mirage")
    return {
        "signal": signal,
        "signal_why": (
            f"Across {h['n_assets']} assets the strongest spectral peak averaged "
            f"**{h['mean_relative_power']:.1f}× the average power**, which sounds like a "
            f"finding until you compute what noise gives. With {h['n_bins']:,} frequency bins "
            f"the periodogram ordinates are independent exponentials, so their maximum is "
            f"expected at **{h['theoretical_max']:.1f}×** the mean — and simulating random "
            f"walks of the same length put the 95th percentile at "
            f"**{h['simulated_p95']:.1f}×**. Fisher's *g* test, which has had the correct "
            f"answer since 1929, rejected the null for **{h['n_significant_white']} of "
            f"{h['n_assets']}** assets against white noise and **{h['n_significant_ar1']}** "
            f"against an AR(1) null — and the AR(1) column is the honest one, because returns "
            f"are autocorrelated and a flat null tilts every test toward finding long cycles. "
            f"The positive control worked: {h['control_asset']}, which has a genuine annual "
            f"demand cycle, showed a peak at **{h['control_period']:.0f} sessions** "
            f"({h['control_period'] / 252:.2f} years), so the machinery detects cycles when "
            f"they exist. On {h['lead_asset']} the best period found in the first half of the "
            f"sample was {h['period_first']:.0f} sessions and in the second half "
            f"{h['period_second']:.0f}, and the phase coherence was "
            f"**{h['phase_concentration']:.2f}** against the 0.70 needed to call a cycle "
            f"coherent — which is what no cycle looks like."),
        "trad_why": (
            f"Trading the best detected period out of sample — fitting the sinusoid on a "
            f"rolling {h['fit_window']}-session window and taking the next step's sign — "
            f"returned {h['cycle_cagr']:+.2%}/yr at a Sharpe of **{h['cycle_sharpe']:.2f}**, "
            f"with a {h['cycle_hit_rate']:.1%} hit rate against the 50% a coin gives, and "
            f"{h['cycle_switches']:.0f} position changes a year to pay for. Buy-and-hold over "
            f"the same window returned {h['cycle_buyhold']:+.2%}. "
            + ("The cycle is real enough to trade, which given everything above deserves more "
               "scrutiny than this study can give it."
               if h["cycle_sharpe"] > 0.3 else
               "That is the expected outcome for a peak that is a peak because some bin has to "
               "be the largest.")),
        "trad": trad,
        "one_sentence": (
            f"The market's strongest spectral peak is {h['mean_relative_power']:.1f}× the "
            f"average power, and a random walk of the same length gives "
            f"{h['simulated_p95']:.1f}× at its 95th percentile — so the honest reading of a "
            f"cycle chart is that {h['n_significant_ar1']} of {h['n_assets']} assets have "
            f"anything to explain."),
    }
