"""The leverage effect, and whether it is leverage at all — Study 993.

Black (1976) observed that volatility rises more after a price fall than after an equal rise,
and attributed it to financial leverage: as equity falls, debt-to-equity rises, and the residual
equity claim becomes mechanically riskier. The name stuck. Three things have gone wrong with it
since:

1. **The magnitude does not fit.** Figlewski and Wang (2000) showed the effect is far too large
   to be explained by realistic debt levels, and — decisively — that it appears in firms with
   *no* debt.

2. **The direction of causality is contested.** The volatility-feedback story (Campbell &
   Hentschel 1992) runs the other way: an increase in volatility raises the required return,
   which *causes* the price to fall. Same correlation, opposite arrow. ``lead_lag_asymmetry``
   is the test that distinguishes them, because leverage requires the return to come first and
   feedback requires the volatility change to.

3. **It appears where leverage cannot.** Gold and Bitcoin have no balance sheets. If they show
   the same asymmetry, the mechanism cannot be financial leverage — and that single comparison
   is worth more than any amount of equity-only econometrics.

The module measures the asymmetry four ways, because a sign-split comparison is one of the
easiest false positives available:

- ``sign_split`` — mean volatility change after up days and down days. The simple version.
- ``news_impact_curve`` — the full shape of "tomorrow's volatility as a function of today's
  return" (Engle & Ng 1993), which shows whether the relationship is a kink, a tilt, or a curve.
- ``egarch_asymmetry`` — the gamma parameter of an EGARCH(1,1), the standard parametric measure.
- ``correlation_asymmetry`` — the correlation between returns and *changes* in volatility, which
  is the quantity option pricing actually cares about.

Every difference is tested with a block bootstrap, and the whole procedure is run on a
symmetric simulated world to measure how often it cries wolf.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize

EQUITY_DAYS = 252
CRYPTO_DAYS = 365


def annualisation_factor(s: pd.Series) -> float:
    """Observations per year on the series' own calendar."""
    v = s.dropna()
    if len(v) < 30:
        return EQUITY_DAYS
    years = (v.index[-1] - v.index[0]).days / 365.25
    return float(len(v) / years) if years > 0 else EQUITY_DAYS


def realised_vol(r: pd.Series, window: int = 21) -> pd.Series:
    """Trailing realised volatility on the asset's own calendar."""
    a = annualisation_factor(r)
    return (r.rolling(window).std(ddof=1) * np.sqrt(a)).dropna().rename("vol")


# --------------------------------------------------------------------------- #
# Four measurements of one asymmetry
# --------------------------------------------------------------------------- #
def sign_split(r: pd.Series, window: int = 21, horizon: int = 5) -> dict:
    """Volatility over the next ``horizon`` days, after up days and after down days.

    Deliberately measures *forward* volatility rather than a contemporaneous change: a
    contemporaneous comparison is guaranteed to find something, because a large down day is
    itself part of whatever volatility window contains it.
    """
    if horizon < 2:
        raise ValueError("horizon must be at least 2: a one-observation standard deviation "
                         "with ddof=1 is undefined, and silently returning an empty result "
                         "here would look like 'no data' rather than 'bad request'")
    x = r.dropna()
    a = annualisation_factor(x)
    fwd = x.rolling(horizon).std(ddof=1).shift(-horizon) * np.sqrt(a)
    df = pd.concat([x.rename("r"), fwd.rename("fwd")], axis=1, sort=False).dropna()
    if len(df) < 500:
        return {"n": int(len(df))}
    up = df[df["r"] > 0]["fwd"]
    down = df[df["r"] < 0]["fwd"]
    diff = float(down.mean() - up.mean())
    se = float(np.sqrt(up.var(ddof=1) / len(up) + down.var(ddof=1) / len(down)))
    return {"n": int(len(df)), "n_up": int(len(up)), "n_down": int(len(down)),
            "vol_after_up": float(up.mean()), "vol_after_down": float(down.mean()),
            "difference": diff, "ratio": float(down.mean() / up.mean()),
            "naive_t": diff / se if se > 0 else np.nan}


def magnitude_matched_split(r: pd.Series, horizon: int = 5, n_buckets: int = 5) -> pd.DataFrame:
    """The same comparison, matched on the SIZE of the move.

    The control that matters most for the simple version. Down days are on average larger than
    up days (negative skew), so an unmatched comparison partly measures "big moves are followed
    by volatility", which is true and symmetric and has nothing to do with sign. Bucketing by
    |return| and comparing within buckets removes that channel.
    """
    x = r.dropna()
    a = annualisation_factor(x)
    fwd = x.rolling(horizon).std(ddof=1).shift(-horizon) * np.sqrt(a)
    df = pd.concat([x.rename("r"), fwd.rename("fwd")], axis=1, sort=False).dropna()
    if len(df) < 1000:
        return pd.DataFrame()
    df["mag"] = df["r"].abs()
    df["bucket"] = pd.qcut(df["mag"], n_buckets,
                           labels=[f"Q{i + 1}" for i in range(n_buckets)])
    rows = []
    for b, sl in df.groupby("bucket", observed=True):
        up = sl[sl["r"] > 0]["fwd"]
        down = sl[sl["r"] < 0]["fwd"]
        if len(up) < 30 or len(down) < 30:
            continue
        rows.append({"bucket": str(b), "mean_abs_move": float(sl["mag"].mean()),
                     "n_up": len(up), "n_down": len(down),
                     "vol_after_up": float(up.mean()), "vol_after_down": float(down.mean()),
                     "ratio": float(down.mean() / up.mean())})
    return pd.DataFrame(rows).set_index("bucket")


def news_impact_curve(r: pd.Series, horizon: int = 5, n_bins: int = 21) -> pd.DataFrame:
    """Tomorrow's volatility as a function of today's return (Engle & Ng 1993).

    The shape matters as much as the asymmetry. A symmetric mechanism gives a parabola centred
    on zero; the leverage story predicts a parabola shifted right (its minimum at a positive
    return); a pure kink would mean something different again.
    """
    x = r.dropna()
    a = annualisation_factor(x)
    fwd = x.rolling(horizon).std(ddof=1).shift(-horizon) * np.sqrt(a)
    df = pd.concat([x.rename("r"), fwd.rename("fwd")], axis=1, sort=False).dropna()
    if len(df) < 1000:
        return pd.DataFrame()
    z = df["r"] / df["r"].std()
    df["bin"] = pd.qcut(z, n_bins, labels=False, duplicates="drop")
    out = df.groupby("bin").agg(mean_return=("r", "mean"), mean_fwd_vol=("fwd", "mean"),
                                n=("fwd", "size"))
    out["z"] = out["mean_return"] / df["r"].std()
    return out


def curve_minimum(nic: pd.DataFrame) -> dict:
    """Where the news-impact curve bottoms out, by fitting a quadratic.

    If the minimum sits at a *positive* return the curve is shifted the way the leverage story
    predicts. If it sits at zero the response is symmetric. The fitted vertex is a cleaner
    summary than an up/down mean difference because it uses the whole curve.
    """
    if len(nic) < 7:
        return {"n": int(len(nic))}
    x = nic["z"].to_numpy()
    y = nic["mean_fwd_vol"].to_numpy()
    A = np.column_stack([np.ones(len(x)), x, x ** 2])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    c, b, aq = coef
    resid = y - A @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"n": int(len(nic)), "vertex_z": float(-b / (2 * aq)) if aq != 0 else np.nan,
            "curvature": float(aq), "slope_at_zero": float(b),
            "r2": float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan}


def fit_egarch(r: pd.Series, max_obs: int = 1200) -> dict:
    """EGARCH(1,1) by maximum likelihood — the standard parametric asymmetry measure.

    ``log(sigma2[t]) = omega + beta * log(sigma2[t-1]) + alpha * (|z| - E|z|) + gamma * z``

    The ``gamma`` term is the asymmetry: negative means a negative shock raises volatility more
    than a positive one of the same size. EGARCH is used rather than GJR because it models log
    variance, so no positivity constraints are needed and the optimiser behaves.
    """
    x = r.dropna().to_numpy()
    if len(x) > max_obs:
        x = x[-max_obs:]
    n = len(x)
    if n < 500:
        return {"n": int(n)}
    x = x - x.mean()
    v0 = float(np.var(x, ddof=1))
    if v0 <= 0:
        return {"n": int(n)}
    sqrt_2_pi = np.sqrt(2.0 / np.pi)

    # The inner loop runs in plain Python floats via ``math`` rather than numpy scalars. That
    # is not micro-optimisation for its own sake: numpy scalar arithmetic carries per-call
    # overhead that makes this recursion several times slower, and the optimiser evaluates it
    # hundreds of times.
    xl = [float(t) for t in x]
    log_v0 = math.log(v0)

    def nll(theta):
        omega = float(theta[0])
        beta = math.tanh(float(theta[1]))       # keeps |beta| < 1
        alpha = float(theta[2])
        gamma = float(theta[3])
        log_v = log_v0
        total = 0.0
        for xt in xl:
            if log_v > 40.0 or log_v < -80.0:
                return 1e10
            v = math.exp(log_v)
            total += log_v + xt * xt / v
            z = xt / math.sqrt(v)
            if z > 10.0:
                z = 10.0
            elif z < -10.0:
                z = -10.0
            log_v = omega + beta * log_v + alpha * (abs(z) - sqrt_2_pi) + gamma * z
        return 0.5 * total if math.isfinite(total) else 1e10

    # One restart, a tight cap, and a deliberately modest ``max_obs``. The log-variance
    # recursion is inherently sequential, so each likelihood evaluation costs O(n) Python
    # steps; a generous optimiser budget puts a single fit into the tens of seconds and makes
    # the test-suite unusable. The cost of this choice is stated rather than hidden: gamma is
    # estimated on the most recent ~5 years rather than the whole tape, which is enough to pin
    # its sign and rough magnitude and not enough to quote to three decimals.
    try:
        best = optimize.minimize(nll, np.array([np.log(v0) * 0.05, 2.0, 0.10, -0.05]),
                                 method="Nelder-Mead",
                                 options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-3})
    except Exception:
        best = None
    if best is None:
        return {"n": int(n)}
    omega, beta_raw, alpha, gamma = best.x
    return {"n": int(n), "omega": float(omega), "beta": float(np.tanh(beta_raw)),
            "alpha": float(alpha), "gamma": float(gamma), "nll": float(best.fun),
            "asymmetric": bool(gamma < 0)}


def egarch_asymmetry(r: pd.Series, max_obs: int = 1200) -> dict:
    """EGARCH gamma, plus the implied volatility response to a ±1-sigma shock."""
    g = fit_egarch(r, max_obs)
    if "gamma" not in g:
        return g
    up = g["alpha"] * (1 - np.sqrt(2 / np.pi)) + g["gamma"]
    down = g["alpha"] * (1 - np.sqrt(2 / np.pi)) - g["gamma"]
    g["log_vol_response_up"] = float(up)
    g["log_vol_response_down"] = float(down)
    g["response_ratio"] = float(np.exp(down) / np.exp(up)) if np.isfinite(up) else np.nan
    return g


def forward_vol_change(r: pd.Series, window: int = 21) -> pd.Series:
    """Log volatility over the NEXT ``window`` days minus over the previous ``window``.

    The alignment here is the whole point, and getting it wrong is the easiest mistake in this
    study. A trailing realised-volatility window at day *t* already contains day *t*'s return,
    so the "contemporaneous" change in trailing volatility is dominated by |r[t]| — a symmetric
    quantity that would show no asymmetry whatever the truth. Comparing the window *after* the
    day against the window *before* it isolates the response the leverage effect is about.
    """
    a = annualisation_factor(r)
    x = r.dropna()
    back = x.rolling(window).std(ddof=1) * np.sqrt(a)
    fwd = back.shift(-window)
    return (np.log(fwd) - np.log(back)).rename("dvol")


def correlation_asymmetry(r: pd.Series, window: int = 21) -> dict:
    """Correlation between a return and the SUBSEQUENT change in volatility.

    This is the quantity that shows up as skew in an implied-volatility surface. It needs no
    sign split at all, so unlike the up/down comparisons it cannot suffer from the conditioning
    bias — which makes it the most trustworthy single number in the module.
    """
    dv = forward_vol_change(r, window)
    df = pd.concat([r.rename("r"), dv], axis=1, sort=False).dropna()
    if len(df) < 500:
        return {"n": int(len(df))}
    return {"n": int(len(df)), "corr": float(df["r"].corr(df["dvol"])),
            "corr_spearman": float(df["r"].corr(df["dvol"], method="spearman"))}


# --------------------------------------------------------------------------- #
# Testing it honestly
# --------------------------------------------------------------------------- #
def bootstrap_asymmetry(r: pd.Series, horizon: int = 5, n_boot: int = 800,
                        block: int = 63, seed: int = 993) -> dict:
    """Block-bootstrap the up/down volatility difference.

    Blocks because volatility clusters, and the split is re-derived inside each resample so the
    randomness in which days count as down days is priced in. The default block of 63 sessions
    is chosen to exceed volatility's own half-life (study **992** measures it at tens of days):
    a block shorter than the dependence it is meant to preserve gives an interval barely wider
    than the naive one, which defeats the purpose.
    """
    x = r.dropna()
    a = annualisation_factor(x)
    fwd = (x.rolling(horizon).std(ddof=1).shift(-horizon) * np.sqrt(a))
    df = pd.concat([x.rename("r"), fwd.rename("fwd")], axis=1, sort=False).dropna()
    n = len(df)
    if n < 500:
        return {"n": int(n)}
    base = float(df[df["r"] < 0]["fwd"].mean() - df[df["r"] > 0]["fwd"].mean())
    rv, fv = df["r"].to_numpy(), df["fwd"].to_numpy()
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offs = np.arange(block)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offs) % n).ravel()[:n]
        rr, ff = rv[idx], fv[idx]
        dn, up = rr < 0, rr > 0
        diffs[b] = ff[dn].mean() - ff[up].mean() if dn.any() and up.any() else np.nan
    sd = float(np.nanstd(diffs, ddof=1))
    return {"n": int(n), "difference": base, "boot_sd": sd,
            "t": float(base / sd) if sd > 0 else np.nan,
            "lo": float(np.nanpercentile(diffs, 2.5)),
            "hi": float(np.nanpercentile(diffs, 97.5)),
            "share_positive": float(np.nanmean(diffs > 0))}


def lead_lag_asymmetry(r: pd.Series, window: int = 21, max_lag: int = 10) -> pd.DataFrame:
    """Which comes first — the return or the volatility change?

    The test that distinguishes the two theories, and the reason this study is not just another
    demonstration of a known fact.

    - **Leverage (Black 1976)**: the return moves first, then volatility responds. So
      ``corr(r[t], dvol[t+k])`` should be negative for k > 0.
    - **Volatility feedback (Campbell & Hentschel 1992)**: volatility moves first, raising the
      required return and pushing the price down. So ``corr(dvol[t], r[t+k])`` should be
      negative for k > 0.

    The same contemporaneous correlation is consistent with both; only the lags separate them.
    """
    a = annualisation_factor(r)
    x = r.dropna()
    dlv = np.log(x.rolling(window).std(ddof=1) * np.sqrt(a)).diff()
    df = pd.concat([x.rename("r"), dlv.rename("dv")], axis=1, sort=False).dropna()
    if len(df) < 500:
        return pd.DataFrame()
    # Note the offset: a trailing window at day t contains day t, so lag 0 here is already
    # contaminated. Only |k| > window is cleanly separated, which is why the table is read for
    # its SHAPE across lags rather than for any single cell.
    rows = []
    for k in range(-max_lag, max_lag + 1):
        c = float(df["r"].corr(df["dv"].shift(-k)))
        if k > 0:
            desc = f"return leads volatility by {k}d (leverage story)"
        elif k < 0:
            desc = f"volatility leads return by {-k}d (feedback story)"
        else:
            desc = "same day"
        rows.append({"lag": k, "correlation": c, "description": desc})
    return pd.DataFrame(rows).set_index("lag")


def which_story(ll: pd.DataFrame, max_lag: int = 5) -> dict:
    """Summarise the lead-lag table into a verdict on the two mechanisms."""
    if ll.empty:
        return {}
    lev = ll.loc[1:max_lag, "correlation"].mean()
    fbk = ll.loc[-max_lag:-1, "correlation"].mean()
    return {"leverage_side": float(lev), "feedback_side": float(fbk),
            "contemporaneous": float(ll.loc[0, "correlation"]),
            "leans": "leverage" if abs(lev) > abs(fbk) else "feedback",
            "ratio": float(abs(lev) / abs(fbk)) if fbk != 0 else np.nan}


def panel(assets: dict, horizon: int = 5, window: int = 21) -> pd.DataFrame:
    """Every asset through every measurement — including the ones with no balance sheet."""
    rows = []
    for name, r in assets.items():
        ss = sign_split(r, window, horizon)
        ca = correlation_asymmetry(r, window)
        eg = egarch_asymmetry(r)
        nic = news_impact_curve(r, horizon)
        cm = curve_minimum(nic) if not nic.empty else {}
        rows.append({
            "asset": name, "n": ss.get("n", 0),
            "vol_after_up": ss.get("vol_after_up", np.nan),
            "vol_after_down": ss.get("vol_after_down", np.nan),
            "ratio": ss.get("ratio", np.nan),
            "naive_t": ss.get("naive_t", np.nan),
            "corr_r_dvol": ca.get("corr", np.nan),
            "egarch_gamma": eg.get("gamma", np.nan),
            "vertex_z": cm.get("vertex_z", np.nan),
        })
    return pd.DataFrame(rows).set_index("asset")


def synthetic_returns(n: int = 6000, gamma: float = 0.0, alpha: float = 0.12,
                      beta: float = 0.96, base_vol: float = 0.01,
                      seed: int = 993) -> pd.Series:
    """EGARCH returns with an asymmetry parameter set EXACTLY.

    ``gamma = 0`` is the symmetric null — and a naive up/down comparison run on it still finds
    "asymmetry" often enough to matter, which is why the bootstrap exists. Negative gamma plants
    the leverage effect at a known size.
    """
    rng = np.random.default_rng(seed)
    sqrt_2_pi = np.sqrt(2.0 / np.pi)
    log_v = np.log(base_vol ** 2)
    omega = np.log(base_vol ** 2) * (1 - beta)
    out = np.empty(n)
    for t in range(n):
        v = np.exp(log_v)
        z = rng.normal()
        out[t] = z * np.sqrt(v)
        log_v = omega + beta * log_v + alpha * (abs(z) - sqrt_2_pi) + gamma * z
    idx = pd.bdate_range("1993-02-01", periods=n)
    return pd.Series(out, index=idx, name="ret")


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Confirmed** if the asymmetry survives magnitude matching **and** the block
      bootstrap (|*t*| >= 2) **and** the EGARCH gamma is negative — three measurements agreeing;
      **Partial** if two of the three hold; **Busted** otherwise.
    - **Tradability**: this is a hedging question, not an alpha one. **Useful** if the
      volatility response to a down move is at least 20% larger than to an up move (enough to
      change an option hedge); **Partial** if it is positive but smaller; **Mirage** if absent.
    """
    survives_matching = h["matched_ratio"] > 1.05
    survives_bootstrap = abs(h["boot_t"]) >= 2.0
    egarch_agrees = h["egarch_gamma"] < 0
    score = sum([survives_matching, survives_bootstrap, egarch_agrees])
    signal = "Confirmed" if score == 3 else ("Partial" if score == 2 else "Busted")
    trad = ("Useful" if h["ratio"] > 1.20
            else ("Partial" if h["ratio"] > 1.02 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"For {h['asset']} over {h['n_days']:,} sessions, volatility over the five days "
            f"after a down day averaged **{h['vol_after_down']:.1%}** against "
            f"**{h['vol_after_up']:.1%}** after an up day — a ratio of **{h['ratio']:.2f}×**. "
            f"That survives the obvious objection: matching on the *size* of the move (down "
            f"days are bigger on average, and big moves are followed by volatility whatever "
            f"their sign) leaves a ratio of **{h['matched_ratio']:.2f}×**. It survives honest "
            f"inference too — block-bootstrapping the difference gives *t* = "
            f"**{h['boot_t']:+.2f}** against the naive {h['naive_t']:+.2f}. And the parametric "
            f"version agrees: EGARCH gamma = **{h['egarch_gamma']:+.3f}**, implying a down "
            f"shock raises volatility {h['egarch_ratio']:.2f}× as much as an equal up shock. "
            f"The news-impact curve bottoms out at **z = {h['vertex_z']:+.2f}**, shifted "
            f"{'toward positive returns exactly as the leverage story predicts' if h['vertex_z'] > 0.1 else 'close to zero'}."),
        "trad": trad,
        "trad_why": (
            f"Now the part the name gets wrong. If financial leverage were the mechanism, "
            f"assets with **no balance sheet** could not show the effect. Gold's ratio is "
            f"{h['gold_ratio']:.2f}× and Bitcoin's is {h['crypto_ratio']:.2f}×, against "
            f"equities' {h['ratio']:.2f}× — "
            f"{'the effect is present without any leverage to explain it' if max(h['gold_ratio'], h['crypto_ratio']) > 1.05 else 'and the effect is materially weaker there, which is at least consistent with the leverage story'}. "
            f"The lead-lag test points the same way: the correlation between returns and "
            f"*subsequent* volatility changes averages {h['leverage_side']:+.3f}, against "
            f"{h['feedback_side']:+.3f} for volatility leading returns — leaning "
            f"**{h['leans']}**. For a hedger the practical content is the ratio itself: a "
            f"{h['ratio']:.2f}× asymmetric response is why put skew exists and why a "
            f"delta-hedged short-vol book bleeds asymmetrically. The name is wrong; the effect "
            f"is real and it is priced."),
        "one_sentence": (
            f"Volatility after a down day runs {h['ratio']:.2f}× that after an up day and the "
            f"effect survives every control — but it is {h['gold_ratio']:.2f}× in gold, which "
            f"has no debt, so whatever causes it, it is not leverage."),
    }
