"""The half-life of a volatility storm — Study 992.

Everyone agrees volatility clusters. Ask how long the clustering lasts and you get numbers
between five days and five months, all correct, because they are measuring different things:

- ``ar1_halflife`` fits an AR(1) to log realised volatility. Fast, standard, and it assumes the
  answer is a single exponential — which is the assumption most likely to be wrong.
- ``acf_halflife`` reads the lag at which the autocorrelation of |returns| first crosses 0.5.
  Assumption-free about the functional form, but noisy and sensitive to the volatility proxy.
- ``garch_halflife`` uses ``log(0.5) / log(alpha + beta)`` from a fitted GARCH(1,1) — the
  practitioner's standard, and typically the *longest* of the five because a GARCH's persistence
  parameter absorbs slow-moving components.
- ``ewma_implied_halflife`` inverts RiskMetrics' λ = 0.94, giving a fixed ~11 days by
  construction. Included precisely because it is an assumption dressed as a measurement, and
  half the risk systems in the world use it.
- ``impulse_response_halflife`` measures directly: after a volatility shock, how many days until
  half the excess has gone? No model, just conditional means.

The disagreement is the finding, and ``two_component_fit`` explains it: volatility behaves like
a **fast component with a half-life of days plus a slow one with a half-life of months**. Every
single-number estimator returns a weighted average of the two, and the weight depends on which
horizon the estimator happens to look at. That is why a GARCH half-life exceeds an ACF half-life
systematically rather than randomly.

``practical_decay`` turns the whole thing into the only question a trader has: after a big day,
how much longer than normal is tomorrow, next week, next month?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

EQUITY_DAYS = 252
CRYPTO_DAYS = 365
METHODS = ("ar1", "acf", "garch", "ewma", "impulse")


# --------------------------------------------------------------------------- #
# Calendar and proxy
# --------------------------------------------------------------------------- #
def annualisation_factor(s: pd.Series) -> float:
    """Observations per year on the series' own calendar."""
    v = s.dropna()
    if len(v) < 30:
        return EQUITY_DAYS
    years = (v.index[-1] - v.index[0]).days / 365.25
    return float(len(v) / years) if years > 0 else EQUITY_DAYS


def realised_vol(r: pd.Series, window: int = 21, ann: float | None = None) -> pd.Series:
    """Trailing realised volatility, annualised on the asset's own calendar."""
    a = annualisation_factor(r) if ann is None else ann
    return (r.rolling(window).std(ddof=1) * np.sqrt(a)).dropna().rename("vol")


def log_vol(r: pd.Series, window: int = 21) -> pd.Series:
    """Log realised volatility — the variable the AR(1) is fitted to.

    Logs rather than levels because volatility is bounded below by zero and its innovations are
    roughly multiplicative. An AR(1) fitted to the level implies a process that can go negative,
    and its residuals are wildly heteroskedastic.
    """
    v = realised_vol(r, window)
    return np.log(v.replace(0, np.nan)).dropna().rename("log_vol")


# --------------------------------------------------------------------------- #
# Five estimators of one quantity
# --------------------------------------------------------------------------- #
def _halflife_from_phi(phi: float) -> float:
    """Convert a persistence coefficient to a half-life in periods."""
    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return np.nan
    return float(np.log(0.5) / np.log(phi))


def ar1_halflife(r: pd.Series, window: int = 21) -> dict:
    """Fit ``log_vol[t] = c + phi * log_vol[t-1] + e`` and convert phi to a half-life.

    One wrinkle that matters: the realised-volatility series built from a rolling window is
    itself smoothed, which pushes phi up and the half-life with it. The ``window`` argument is
    swept in the results for that reason, and a window of 1 (absolute returns) is the
    unsmoothed limit.
    """
    lv = log_vol(r, window)
    if len(lv) < 200:
        return {"n": int(len(lv))}
    y = lv.iloc[1:].to_numpy()
    x = lv.iloc[:-1].to_numpy()
    n = len(y)
    A = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    s2 = float((resid ** 2).sum() / max(n - 2, 1))
    se = float(np.sqrt(s2 * np.linalg.pinv(A.T @ A)[1, 1]))
    phi = float(coef[1])
    # A higher phi means a LONGER half-life, so the upper confidence bound on the half-life
    # comes from the upper bound on phi. Getting this backwards is easy and silent.
    return {"n": int(n), "phi": phi, "se": se, "halflife": _halflife_from_phi(phi),
            "halflife_lo": _halflife_from_phi(max(phi - 1.96 * se, 1e-6)),
            "halflife_hi": _halflife_from_phi(min(phi + 1.96 * se, 0.9999)),
            "window": window}


def acf_halflife(r: pd.Series, max_lag: int = 250, proxy: str = "abs") -> dict:
    """The lag at which the autocorrelation of the volatility proxy first falls below 0.5.

    Interpolated between integer lags. Assumes nothing about functional form, which is its
    virtue; it is noisy and depends on the proxy, which is its cost. Both proxies are offered
    because squared returns are noisier than absolute returns (Forsberg & Ghysels 2007).
    """
    x = r.dropna()
    p = x.abs() if proxy == "abs" else x.pow(2)
    if len(p) < 500:
        return {"n": int(len(p))}
    acf = np.array([p.autocorr(lag) for lag in range(1, max_lag + 1)])
    below = np.flatnonzero(acf < 0.5 * acf[0])
    if len(below) == 0:
        return {"n": int(len(p)), "halflife": np.nan, "acf_1": float(acf[0]),
                "acf_at_max_lag": float(acf[-1]), "proxy": proxy}
    k = int(below[0])
    if k == 0:
        hl = 1.0
    else:
        y0, y1 = acf[k - 1], acf[k]
        target = 0.5 * acf[0]
        hl = float(k + (y0 - target) / (y0 - y1)) if y0 != y1 else float(k + 1)
    return {"n": int(len(p)), "halflife": hl, "acf_1": float(acf[0]),
            "acf_at_max_lag": float(acf[-1]), "proxy": proxy,
            "acf": acf}


def fit_garch11(r: pd.Series, max_obs: int = 6000) -> dict:
    """GARCH(1,1) by maximum likelihood with variance targeting.

    Variance targeting fixes the unconditional variance at the sample value and optimises only
    over the two persistence parameters, which makes the likelihood far better behaved than the
    three-parameter version and costs almost nothing in fit. Alpha and beta are parameterised
    through a logit so the stationarity constraint alpha + beta < 1 holds by construction rather
    than by a penalty.
    """
    x = r.dropna().to_numpy()
    if len(x) > max_obs:
        x = x[-max_obs:]
    n = len(x)
    if n < 500:
        return {"n": int(n)}
    x = x - x.mean()
    uncond = float(np.var(x, ddof=1))
    if uncond <= 0:
        return {"n": int(n)}

    def unpack(theta):
        persistence = 1.0 / (1.0 + np.exp(-theta[0]))          # in (0, 1)
        alpha_share = 1.0 / (1.0 + np.exp(-theta[1]))          # alpha's share of it
        alpha = persistence * alpha_share
        beta = persistence - alpha
        return alpha, beta

    def nll(theta):
        alpha, beta = unpack(theta)
        omega = uncond * (1.0 - alpha - beta)
        if omega <= 0:
            return 1e10
        sigma2 = uncond
        total = 0.0
        for t in range(n):
            total += np.log(sigma2) + x[t] ** 2 / sigma2
            sigma2 = omega + alpha * x[t] ** 2 + beta * sigma2
            if not np.isfinite(sigma2) or sigma2 <= 0:
                return 1e10
        return 0.5 * total

    best = None
    for start in ([2.5, -2.0], [3.5, -2.5], [1.5, -1.5]):
        try:
            res = optimize.minimize(nll, np.array(start), method="Nelder-Mead",
                                    options={"maxiter": 400, "xatol": 1e-4,
                                             "fatol": 1e-4})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    if best is None:
        return {"n": int(n)}
    alpha, beta = unpack(best.x)
    return {"n": int(n), "alpha": float(alpha), "beta": float(beta),
            "persistence": float(alpha + beta),
            "omega": float(uncond * (1 - alpha - beta)),
            "uncond_vol": float(np.sqrt(uncond)), "nll": float(best.fun)}


def garch_halflife(r: pd.Series, max_obs: int = 6000) -> dict:
    """The practitioner's standard: log(0.5) / log(alpha + beta)."""
    g = fit_garch11(r, max_obs)
    if "persistence" not in g:
        return g
    g["halflife"] = _halflife_from_phi(g["persistence"])
    return g


def ewma_implied_halflife(lam: float = 0.94) -> float:
    """RiskMetrics' λ, inverted. Fixed by assumption, not measured — that is the point."""
    return _halflife_from_phi(lam)


def impulse_response_halflife(r: pd.Series, window: int = 21, shock_q: float = 0.95,
                              max_lag: int = 120) -> dict:
    """Model-free: after a volatility shock, how long until half the excess is gone?

    Take every day whose realised volatility sits above the ``shock_q`` quantile, average the
    path of volatility over the following ``max_lag`` days, and find where it has decayed
    halfway back to the unconditional mean. No functional form, no parameters — just conditional
    means, which is also why it needs a lot of data to be stable.
    """
    v = realised_vol(r, window)
    if len(v) < 1000:
        return {"n": int(len(v))}
    lv = np.log(v)
    mu = float(lv.mean())
    thresh = float(lv.quantile(shock_q))
    starts = np.flatnonzero((lv.to_numpy() > thresh))
    starts = starts[starts < len(lv) - max_lag]
    if len(starts) < 30:
        return {"n": int(len(v)), "n_shocks": int(len(starts))}
    arr = lv.to_numpy()
    paths = np.array([arr[s:s + max_lag + 1] for s in starts])
    mean_path = paths.mean(axis=0) - mu
    if mean_path[0] <= 0:
        return {"n": int(len(v)), "n_shocks": int(len(starts))}
    target = mean_path[0] / 2.0
    below = np.flatnonzero(mean_path < target)
    hl = float(below[0]) if len(below) else np.nan
    return {"n": int(len(v)), "n_shocks": int(len(starts)),
            "initial_excess": float(mean_path[0]), "halflife": hl,
            "excess_at_5d": float(mean_path[min(5, max_lag)]),
            "excess_at_21d": float(mean_path[min(21, max_lag)]),
            "excess_at_63d": float(mean_path[min(63, max_lag)]),
            "path": mean_path}


def halflife_table(r: pd.Series, window: int = 21) -> pd.DataFrame:
    """All five estimators of the same quantity, side by side."""
    a = ar1_halflife(r, window)
    c = acf_halflife(r)
    g = garch_halflife(r)
    i = impulse_response_halflife(r, window)
    return pd.DataFrame([
        {"method": "ar1", "halflife": a.get("halflife", np.nan),
         "note": f"phi = {a.get('phi', np.nan):.4f} on {window}d realised vol"},
        {"method": "acf", "halflife": c.get("halflife", np.nan),
         "note": f"|returns|, acf(1) = {c.get('acf_1', np.nan):.3f}"},
        {"method": "garch", "halflife": g.get("halflife", np.nan),
         "note": f"alpha+beta = {g.get('persistence', np.nan):.4f}"},
        {"method": "ewma", "halflife": ewma_implied_halflife(),
         "note": "lambda = 0.94, assumed not measured"},
        {"method": "impulse", "halflife": i.get("halflife", np.nan),
         "note": f"{i.get('n_shocks', 0)} shocks above the 95th percentile"},
    ]).set_index("method")


# --------------------------------------------------------------------------- #
# Why they disagree
# --------------------------------------------------------------------------- #
def two_component_fit(r: pd.Series, window: int = 21, max_lag: int = 250) -> dict:
    """Fit ``acf(k) = w * exp(-k/t1) + (1-w) * exp(-k/t2)`` to the volatility autocorrelation.

    The explanation for the whole disagreement. If volatility is a fast component plus a slow
    one, then every single-number estimator returns a weighted average whose weight depends on
    which lags that estimator emphasises — which is why the ordering of the five is stable
    across assets rather than random.
    """
    v = np.log(realised_vol(r, window))
    if len(v) < 1000:
        return {"n": int(len(v))}
    lags = np.arange(1, max_lag + 1)
    acf = np.array([v.autocorr(int(k)) for k in lags])
    ok = np.isfinite(acf)
    lags, acf = lags[ok], acf[ok]
    if len(lags) < 50:
        return {"n": int(len(v))}

    def model(p, k):
        w, t1, t2 = p
        w = 1 / (1 + np.exp(-w))
        t1, t2 = np.exp(t1), np.exp(t2)
        return w * np.exp(-k / t1) + (1 - w) * np.exp(-k / t2)

    def loss(p):
        return float(np.sum((model(p, lags) - acf) ** 2))

    best = None
    for start in ([0.0, 1.0, 4.0], [1.0, 0.5, 5.0], [-0.5, 1.5, 3.5]):
        try:
            res = optimize.minimize(loss, np.array(start), method="Nelder-Mead",
                                    options={"maxiter": 2000})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    if best is None:
        return {"n": int(len(v))}
    w = float(1 / (1 + np.exp(-best.x[0])))
    t1, t2 = float(np.exp(best.x[1])), float(np.exp(best.x[2]))
    if t1 > t2:                       # by convention the first component is the fast one
        w, t1, t2 = 1 - w, t2, t1
    single = float(np.sum((np.exp(-lags / max(t1, 1e-6)) - acf) ** 2))
    return {"n": int(len(v)), "weight_fast": w, "tau_fast": t1, "tau_slow": t2,
            "halflife_fast": t1 * np.log(2), "halflife_slow": t2 * np.log(2),
            "sse_two_component": float(best.fun), "sse_single": single,
            "improvement": float(1 - best.fun / single) if single > 0 else np.nan}


def window_sweep(r: pd.Series, windows=(1, 5, 10, 21, 63)) -> pd.DataFrame:
    """The AR(1) half-life against the window used to build its input.

    Two biases pull in opposite directions here, and the sweep is the only way to see both.

    **Attenuation (short windows).** Realised volatility is a *noisy proxy* for the latent
    volatility. Classical errors-in-variables says a regressor measured with noise has its
    coefficient biased toward zero — so a one-day proxy (log absolute returns, which has an
    enormous idiosyncratic component) gives a phi far below the truth and a half-life of well
    under a day. That is not a measurement of volatility's persistence; it is a measurement of
    how noisy a single day's absolute return is.

    **Smoothing (long windows).** A rolling mean induces autocorrelation of its own. A 63-day
    window makes consecutive observations share 62 of 63 days, so phi is pushed up and the
    half-life with it, whatever the underlying process does.

    The truth sits between, and the sweep shows where the curve flattens — which is the honest
    way to pick a window rather than defending whichever one gave the desired answer.
    """
    rows = []
    for w in windows:
        a = ar1_halflife(r, w) if w > 1 else _ar1_on_abs(r)
        rows.append({"window": w, "phi": a.get("phi", np.nan),
                     "halflife": a.get("halflife", np.nan), "n": a.get("n", 0)})
    return pd.DataFrame(rows).set_index("window")


def _ar1_on_abs(r: pd.Series) -> dict:
    """The unsmoothed limit: an AR(1) on log absolute returns, no rolling window at all."""
    x = np.log(r.abs().replace(0, np.nan)).dropna()
    if len(x) < 200:
        return {"n": int(len(x))}
    y, xx = x.iloc[1:].to_numpy(), x.iloc[:-1].to_numpy()
    n = len(y)
    A = np.column_stack([np.ones(n), xx])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    phi = float(coef[1])
    return {"n": int(n), "phi": phi, "halflife": _halflife_from_phi(phi), "window": 1}


def practical_decay(r: pd.Series, window: int = 21, shock_q: float = 0.95,
                    horizons=(2, 5, 21, 63, 126)) -> pd.DataFrame:
    """The only question a trader asks: after a big move, how much wilder is the next month?"""
    v = realised_vol(r, window)
    if len(v) < 1000:
        return pd.DataFrame()
    r_ = r.reindex(v.index)
    thresh = float(v.quantile(shock_q))
    hot = v > thresh
    base = float(r_.std())
    rows = []
    for hz in horizons:
        fwd = r_.rolling(hz).std().shift(-hz)
        was_hot = hot.shift(1).fillna(False).astype(bool)
        after_hot = fwd[was_hot].dropna()
        after_calm = fwd[~was_hot].dropna()
        if len(after_hot) < 30:
            continue
        rows.append({"horizon": hz, "n_hot": len(after_hot),
                     "vol_after_shock": float(after_hot.mean() / base),
                     "vol_after_calm": float(after_calm.mean() / base),
                     "ratio": float(after_hot.mean() / after_calm.mean())})
    return pd.DataFrame(rows).set_index("horizon")


def synthetic_vol(n: int = 6000, halflife: float = 20.0, second_halflife: float = 0.0,
                  weight_fast: float = 0.5, base_vol: float = 0.01,
                  seed: int = 992) -> pd.Series:
    """Returns whose log-volatility has an EXACTLY known half-life (or two).

    With ``second_halflife = 0`` the process is a single AR(1) and every estimator should
    recover ``halflife``. With a second component switched on, no single number is correct — and
    watching each estimator settle on a different wrong answer is the study's central
    demonstration.
    """
    rng = np.random.default_rng(seed)
    phi1 = 0.5 ** (1.0 / halflife)
    c1 = np.zeros(n)
    for t in range(1, n):
        c1[t] = phi1 * c1[t - 1] + rng.normal(0, 1)
    c1 /= max(c1.std(), 1e-9)
    if second_halflife and second_halflife > 0:
        phi2 = 0.5 ** (1.0 / second_halflife)
        c2 = np.zeros(n)
        for t in range(1, n):
            c2[t] = phi2 * c2[t - 1] + rng.normal(0, 1)
        c2 /= max(c2.std(), 1e-9)
        logv = np.sqrt(weight_fast) * c1 + np.sqrt(1 - weight_fast) * c2
    else:
        logv = c1
    vol = base_vol * np.exp(0.45 * logv - 0.45 ** 2 / 2)
    idx = pd.bdate_range("1993-02-01", periods=n)
    return pd.Series(rng.normal(0, 1, n) * vol, index=idx, name="ret")


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Confirmed** (clustering is real and persistent) if the impulse-response
      half-life exceeds five days **and** the two-component fit beats the single-exponential fit
      by more than 20% of squared error — i.e. the clustering is real *and* it is not one
      simple exponential; **Partial** if only the first holds; **Busted** if volatility shows no
      measurable persistence.
    - **Tradability**: **Useful** if volatility after a shock is still materially elevated at
      one month (ratio above 1.2) — long enough to reposition against; **Partial** if it is
      elevated at a week but not a month; **Mirage** otherwise.
    """
    persistent = h["impulse_halflife"] > 5
    multiscale = h["two_component_improvement"] > 0.20
    signal = ("Confirmed" if (persistent and multiscale)
              else ("Partial" if persistent else "Busted"))
    if h["ratio_21d"] > 1.2:
        trad = "Useful"
    elif h["ratio_5d"] > 1.2:
        trad = "Partial"
    else:
        trad = "Mirage"
    return {
        "signal": signal,
        "signal_why": (
            f"For {h['asset']} over {h['n_days']:,} sessions, the five standard ways of "
            f"measuring the half-life of a volatility storm give **{h['hl_min']:.0f} to "
            f"{h['hl_max']:.0f} days** — a spread of {h['hl_max'] / h['hl_min']:.1f}×. They are "
            f"not contradicting each other; they are measuring different things, because "
            f"volatility is not one process. Fitting a two-component autocorrelation gives a "
            f"**fast part with a {h['halflife_fast']:.1f}-day half-life carrying "
            f"{h['weight_fast']:.0%} of the variation, and a slow part at "
            f"{h['halflife_slow']:.0f} days** — a fit that beats the single-exponential version "
            f"by **{h['two_component_improvement']:.0%}** of squared error. Every single-number "
            f"estimator returns a weighted average of those two, weighted by whichever lags it "
            f"happens to look at, which is why GARCH ({h['hl_garch']:.0f} days) systematically "
            f"exceeds the raw autocorrelation reading ({h['hl_acf']:.0f} days) rather than "
            f"differing from it at random. The model-free impulse response — after a "
            f"95th-percentile volatility day, how long until half the excess is gone — says "
            f"**{h['impulse_halflife']:.0f} days**."),
        "trad": trad,
        "trad_why": (
            f"The version that decides anything: after a top-5% volatility day, realised "
            f"volatility over the next week ran **{h['ratio_5d']:.2f}×** the level following a "
            f"normal day, over the next month **{h['ratio_21d']:.2f}×**, and over the next "
            f"quarter {h['ratio_63d']:.2f}×. That is a slow enough decay to reposition against "
            f"— nobody needs a same-day reaction to exploit a signal that is still worth "
            f"{h['ratio_21d']:.2f}× a month later. Two caveats travel with it: the half-life is "
            f"itself unstable (across the {h['n_assets']} assets here it ranges "
            f"{h['cross_min']:.0f} to {h['cross_max']:.0f} days), and the AR(1) number is "
            f"partly an artefact of the estimation window — sweeping that window from 1 to 63 "
            f"days moves the answer from {h['sweep_min']:.1f} to {h['sweep_max']:.0f} days "
            f"on identical data, attenuated at the short end by proxy noise and inflated at "
            f"the long end by the rolling mean's own autocorrelation."),
        "one_sentence": (
            f"Volatility's half-life is {h['hl_min']:.0f} days or {h['hl_max']:.0f} days "
            f"depending on how you ask, because it is genuinely two processes — a "
            f"{h['halflife_fast']:.0f}-day one and a {h['halflife_slow']:.0f}-day one — and "
            f"every single-number answer is an average of them."),
    }
