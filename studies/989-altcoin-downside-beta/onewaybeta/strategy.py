"""Up-beta, down-beta, and the trap between them — Study 989.

Split a sample on the sign of the market return, fit a beta in each half, and you will
almost always find that they differ. That is the easiest false positive in finance, and it has
three distinct causes that have to be separated before any claim about "downside beta" means
anything:

1. **Conditioning on the regressor is not free.** Splitting on the sign of *x* truncates its
   distribution in each half. The two conditional betas of a perfectly symmetric bivariate
   normal are *not* equal in a finite sample, and their sampling variance is much larger than
   the full-sample beta's. ``asymmetry_test`` bootstraps the difference rather than comparing
   two standard errors that were never designed to be compared.

2. **Correlation rises in crashes for reasons unrelated to beta.** Longin and Solnik (2001) and
   Ang and Chen (2002) established that measured *correlation* increases in the tails even in a
   multivariate normal with constant parameters — purely a conditioning artefact. So the study
   reports both, and reports the artefact-implied value beside the measured one.

3. **Betas move over time.** An altcoin whose beta rose in 2021 and fell in 2023 will show a
   spurious up/down difference if those two periods happened to be a bull and a bear. The
   ``time_varying_control`` splits by *era* first, then by sign inside each era.

The measurements are then given a name: ``bawa_lindenberg_beta`` is the downside beta from the
lower-partial-moment CAPM literature (Bawa & Lindenberg 1977), ``hogan_warren_beta`` the
semivariance version, and ``coskewness`` the third-moment statistic that a genuine asymmetry
should also show up in. Three independent measurements of one alleged phenomenon is the
minimum standard for believing it.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

CRYPTO_DAYS = 365


# --------------------------------------------------------------------------- #
# Betas, conditional and otherwise
# --------------------------------------------------------------------------- #
def align(y: pd.Series, x: pd.Series) -> pd.DataFrame:
    """Both series on their common, non-missing dates."""
    return pd.concat([y.rename("y"), x.rename("x")], axis=1, sort=False).dropna()


def ols_beta(y: pd.Series, x: pd.Series) -> dict:
    """Plain OLS beta with HC1 errors."""
    df = align(y, x)
    n = len(df)
    if n < 60:
        return {"n": int(n), "beta": np.nan, "se": np.nan, "alpha": np.nan, "r2": np.nan}
    A = np.column_stack([np.ones(n), df["x"].to_numpy()])
    coef, *_ = np.linalg.lstsq(A, df["y"].to_numpy(), rcond=None)
    resid = df["y"].to_numpy() - A @ coef
    XtX_inv = np.linalg.pinv(A.T @ A)
    V = XtX_inv @ (A.T @ np.diag(resid ** 2) @ A) @ XtX_inv * n / max(n - 2, 1)
    ss_tot = float(((df["y"] - df["y"].mean()) ** 2).sum())
    return {"n": int(n), "alpha": float(coef[0]), "beta": float(coef[1]),
            "se": float(np.sqrt(max(V[1, 1], 0.0))),
            "r2": float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan}


def conditional_betas(y: pd.Series, x: pd.Series, threshold: float = 0.0) -> dict:
    """Beta on days the market rose, and on days it fell."""
    df = align(y, x)
    up = df[df["x"] > threshold]
    down = df[df["x"] < -threshold] if threshold > 0 else df[df["x"] <= threshold]
    bu = ols_beta(up["y"], up["x"])
    bd = ols_beta(down["y"], down["x"])
    return {"beta_up": bu["beta"], "beta_down": bd["beta"],
            "n_up": bu["n"], "n_down": bd["n"],
            "se_up": bu["se"], "se_down": bd["se"],
            "difference": bd["beta"] - bu["beta"],
            "threshold": threshold}


def bawa_lindenberg_beta(y: pd.Series, x: pd.Series, target: float = 0.0) -> float:
    """Downside beta from the lower-partial-moment CAPM (Bawa & Lindenberg 1977).

    ``cov(y, x | x < target) / var(x | x < target)`` computed with the *conditional* means,
    which is what the LPM-CAPM specifies and what most casual "downside beta" calculations get
    wrong by using the unconditional means.
    """
    df = align(y, x)
    sl = df[df["x"] < target]
    if len(sl) < 60:
        return np.nan
    xv = sl["x"].to_numpy()
    yv = sl["y"].to_numpy()
    vx = float(np.var(xv, ddof=1))
    return float(np.cov(xv, yv, ddof=1)[0, 1] / vx) if vx > 0 else np.nan


def hogan_warren_beta(y: pd.Series, x: pd.Series, target: float = 0.0) -> float:
    """The semivariance beta (Hogan & Warren 1974): E[(y-t)(x-t)^-] / E[((x-t)^-)^2]."""
    df = align(y, x)
    xd = np.minimum(df["x"].to_numpy() - target, 0.0)
    yd = df["y"].to_numpy() - target
    denom = float((xd ** 2).mean())
    return float((yd * xd).mean() / denom) if denom > 0 else np.nan


def coskewness(y: pd.Series, x: pd.Series) -> float:
    """Standardised coskewness: E[ey * ex^2] / (sd(y) * var(x)).

    A negative value means the asset does disproportionately badly when the market moves a lot
    — the third-moment fingerprint of the same phenomenon that an up/down beta split is trying
    to detect. If the beta split says "asymmetry" and this says nothing, be suspicious.
    """
    df = align(y, x)
    if len(df) < 100:
        return np.nan
    ey = df["y"] - df["y"].mean()
    ex = df["x"] - df["x"].mean()
    denom = float(ey.std(ddof=1) * ex.var(ddof=1))
    return float((ey * ex ** 2).mean() / denom) if denom > 0 else np.nan


@lru_cache(maxsize=64)
def _normal_tail_benchmark(rho: float, n: int, q: float, n_sims: int = 200,
                           seed: int = 989) -> tuple:
    """What tail correlation a bivariate NORMAL with correlation ``rho`` would show.

    Cached on (rho, n, q) because it is called once per asset and the answer depends on nothing
    else. Rho is rounded by the caller so that near-identical assets share one simulation.
    """
    rng = np.random.default_rng(seed)
    sim_down, sim_up = [], []
    for _ in range(n_sims):
        z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
        zx, zy = z[:, 1], z[:, 0]
        l2, h2 = np.quantile(zx, q), np.quantile(zx, 1 - q)
        sim_down.append(np.corrcoef(zy[zx <= l2], zx[zx <= l2])[0, 1])
        sim_up.append(np.corrcoef(zy[zx >= h2], zx[zx >= h2])[0, 1])
    return float(np.mean(sim_down)), float(np.mean(sim_up))


def tail_correlation(y: pd.Series, x: pd.Series, q: float = 0.10) -> dict:
    """Correlation in the market's worst and best deciles, with the *normal* benchmark.

    Longin & Solnik (2001): measured correlation rises in the tails even under a bivariate
    normal with constant correlation, purely because conditioning truncates the distribution.
    So the raw tail correlation is meaningless without the value a normal would have produced,
    which this function simulates and returns alongside.
    """
    df = align(y, x)
    if len(df) < 200:
        return {"n": int(len(df))}
    lo, hi = df["x"].quantile(q), df["x"].quantile(1 - q)
    rho = float(df["y"].corr(df["x"]))
    down = df[df["x"] <= lo]
    up = df[df["x"] >= hi]
    # Rounded so near-identical assets share one cached simulation.
    sim_down, sim_up = _normal_tail_benchmark(round(rho, 2), len(df), q)
    return {"n": int(len(df)), "overall": rho,
            "down_tail": float(down["y"].corr(down["x"])),
            "up_tail": float(up["y"].corr(up["x"])),
            "normal_down_tail": sim_down, "normal_up_tail": sim_up,
            "excess_down": float(down["y"].corr(down["x"]) - sim_down)}


# --------------------------------------------------------------------------- #
# Testing the difference honestly
# --------------------------------------------------------------------------- #
def _fast_beta(y: np.ndarray, x: np.ndarray) -> float:
    """cov/var on raw arrays — the inner loop of the bootstrap, with no pandas in it.

    Identical arithmetic to ``ols_beta``'s slope; separated out because the bootstrap calls it
    a few hundred thousand times and building a DataFrame each time makes the honest test too
    slow to run, which is how people end up shipping the dishonest one.
    """
    n = len(y)
    if n < 30:
        return np.nan
    xm, ym = x.mean(), y.mean()
    vx = float(((x - xm) ** 2).sum())
    return float(((x - xm) * (y - ym)).sum() / vx) if vx > 0 else np.nan


def asymmetry_test(y: pd.Series, x: pd.Series, n_boot: int = 1000, block: int = 20,
                   seed: int = 989) -> dict:
    """Block-bootstrap the up/down beta difference.

    Two things this does that a naive comparison of two standard errors does not. It resamples
    in **blocks**, because crypto returns are volatility-clustered and a day-level bootstrap
    would understate the uncertainty. And it re-derives the split inside each resample, so the
    sampling variation in *which days count as down days* is included — that variation is a
    large part of the total and is invisible to the textbook two-sample test.
    """
    df = align(y, x)
    n = len(df)
    if n < 300:
        return {"n": int(n)}
    base = conditional_betas(df["y"], df["x"])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    yv, xv = df["y"].to_numpy(), df["x"].to_numpy()
    offsets = np.arange(block)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets) % n).ravel()[:n]
        ys, xs = yv[idx], xv[idx]
        up = xs > 0
        diffs[b] = _fast_beta(ys[~up], xs[~up]) - _fast_beta(ys[up], xs[up])
    sd = float(np.nanstd(diffs, ddof=1))
    return {"n": int(n), "beta_up": base["beta_up"], "beta_down": base["beta_down"],
            "difference": base["difference"], "boot_sd": sd,
            "t": float(base["difference"] / sd) if sd > 0 else np.nan,
            "lo": float(np.nanpercentile(diffs, 2.5)),
            "hi": float(np.nanpercentile(diffs, 97.5)),
            "share_positive": float(np.nanmean(diffs > 0))}


def naive_two_sample_t(cb: dict) -> float:
    """The test most people run: (beta_down - beta_up) / sqrt(se_up^2 + se_down^2).

    Kept in the module so the results can show how much larger it is than the honest one. It
    ignores that the split itself is random and that the residuals are clustered.
    """
    se = np.sqrt(cb["se_up"] ** 2 + cb["se_down"] ** 2)
    return float(cb["difference"] / se) if se > 0 else np.nan


def threshold_sweep(y: pd.Series, x: pd.Series,
                    thresholds=(0.0, 0.01, 0.02, 0.03, 0.05)) -> pd.DataFrame:
    """The asymmetry at several definitions of "a down day"."""
    rows = []
    for t in thresholds:
        cb = conditional_betas(y, x, t)
        rows.append({"threshold": t, "n_up": cb["n_up"], "n_down": cb["n_down"],
                     "beta_up": cb["beta_up"], "beta_down": cb["beta_down"],
                     "difference": cb["difference"], "naive_t": naive_two_sample_t(cb)})
    return pd.DataFrame(rows).set_index("threshold")


def time_varying_control(y: pd.Series, x: pd.Series, n_eras: int = 4) -> pd.DataFrame:
    """Split by era first, then by sign inside each era.

    A beta that drifted upward across the sample will show a fake up/down difference if the
    high-beta years happened to be down years. Fitting within eras removes that channel.
    """
    df = align(y, x)
    rows = []
    for k, ix in enumerate(np.array_split(np.arange(len(df)), n_eras)):
        sl = df.iloc[ix]
        cb = conditional_betas(sl["y"], sl["x"])
        rows.append({"era": f"{sl.index[0].date()} to {sl.index[-1].date()}",
                     "n": len(sl), "beta_all": ols_beta(sl["y"], sl["x"])["beta"],
                     "beta_up": cb["beta_up"], "beta_down": cb["beta_down"],
                     "difference": cb["difference"]})
    return pd.DataFrame(rows).set_index("era")


def panel_summary(alts: dict, bench: pd.Series) -> pd.DataFrame:
    """Every altcoin through every measurement."""
    rows = []
    for name, r in alts.items():
        cb = conditional_betas(r, bench)
        rows.append({
            "asset": name, "n": cb["n_up"] + cb["n_down"],
            "beta": ols_beta(r, bench)["beta"],
            "beta_up": cb["beta_up"], "beta_down": cb["beta_down"],
            "difference": cb["difference"], "naive_t": naive_two_sample_t(cb),
            "bawa_lindenberg": bawa_lindenberg_beta(r, bench),
            "hogan_warren": hogan_warren_beta(r, bench),
            "coskewness": coskewness(r, bench),
        })
    return pd.DataFrame(rows).set_index("asset")


# --------------------------------------------------------------------------- #
# What it costs
# --------------------------------------------------------------------------- #
def capture_ratios(y: pd.Series, x: pd.Series) -> dict:
    """Upside and downside capture — the practitioner's version of the same question.

    Capture ratios are compounded, not averaged, so they answer "what did I actually end up
    with" rather than "what was the average sensitivity". The two can disagree, and when they
    do the compounded version is the one that spends.

    One implementation note that matters. The textbook capture ratio compounds *every* up
    period and divides. On monthly data over a decade that is fine; on several thousand daily
    observations it **saturates** — compound two thousand down days of any magnitude and both
    numerator and denominator approach −100%, so the ratio approaches 1 regardless of the
    truth. This uses the **per-day geometric mean** instead, which is the same quantity
    normalised by horizon and is well defined at any sample length.
    """
    df = align(y, x)
    up, down = df[df["x"] > 0], df[df["x"] <= 0]
    if len(up) < 30 or len(down) < 30:
        return {"n": int(len(df))}

    def geo(s):
        """Per-observation geometric mean return."""
        v = np.log1p(s.to_numpy())
        return float(np.expm1(v.mean()))

    gux, guy = geo(up["x"]), geo(up["y"])
    gdx, gdy = geo(down["x"]), geo(down["y"])
    cu = guy / gux if gux != 0 else np.nan
    cd = gdy / gdx if gdx != 0 else np.nan
    return {"n": int(len(df)), "up_capture": cu, "down_capture": cd,
            "ratio": cu / cd if cd and np.isfinite(cd) and cd != 0 else np.nan,
            "up_days": len(up), "down_days": len(down),
            "geo_up_market": gux, "geo_down_market": gdx}


def drawdown_comparison(prices: pd.Series, bench_px: pd.Series) -> dict:
    """The number a holder actually experiences."""
    df = pd.concat([prices.rename("a"), bench_px.rename("b")], axis=1, sort=False).dropna()
    if len(df) < 200:
        return {"n": int(len(df))}
    da = df["a"] / df["a"].cummax() - 1
    db = df["b"] / df["b"].cummax() - 1
    worst = db.idxmin()
    return {"n": int(len(df)), "max_dd": float(da.min()), "bench_max_dd": float(db.min()),
            "dd_ratio": float(da.min() / db.min()) if db.min() != 0 else np.nan,
            "dd_at_bench_worst": float(da.loc[worst]),
            "recovery_ratio": float((df["a"].iloc[-1] / df["a"].max())
                                    / (df["b"].iloc[-1] / df["b"].max()))}


def synthetic_world(n: int = 3000, beta_up: float = 1.5, beta_down: float = 1.5,
                    idio_vol: float = 0.03, bench_vol: float = 0.60,
                    seed: int = 989) -> pd.DataFrame:
    """Bitcoin and an altcoin with independently controllable up- and down-betas.

    With ``beta_up == beta_down`` the world is symmetric and every test here must say so — even
    though a naive up/down split will still report a difference a large fraction of the time,
    which is the failure mode the study is built around.
    """
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0005, bench_vol / np.sqrt(CRYPTO_DAYS), n)
    beta_t = np.where(b > 0, beta_up, beta_down)
    a = beta_t * b + rng.normal(0, idio_vol, n)
    idx = pd.date_range("2017-11-09", periods=n, freq="D")
    return pd.DataFrame({"bench": b, "alt": a}, index=idx)


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** only if the *median* altcoin's down-beta exceeds its up-beta with a
      block-bootstrapped |*t*| >= 2 **and** the coskewness agrees in sign for a majority of the
      panel — one measurement is an artefact factory, two agreeing is evidence; **Weak** if the
      difference is positive but fails one of those; **None** otherwise.
    - **Tradability**: **Mirage** by construction if the effect is not Real — there is nothing
      to trade. Otherwise **Useful** if the down-capture penalty is large enough to change a
      sizing decision (over 20 percentage points), **Partial** below that.
    """
    positive = h["median_difference"] > 0
    significant = abs(h["median_boot_t"]) >= 2.0
    corroborated = h["share_negative_coskew"] >= 0.5
    signal = ("Real" if (positive and significant and corroborated)
              else ("Weak" if positive else "None"))
    if signal != "Real":
        trad = "Mirage"
    else:
        gap = h["median_down_capture"] - h["median_up_capture"]
        trad = "Useful" if gap > 0.20 else "Partial"
    return {
        "signal": signal,
        "signal_why": (
            f"Across {h['n_alts']} majors against Bitcoin over {h['years']:.1f} years, the "
            f"median altcoin's beta is **{h['median_beta_up']:.2f} on Bitcoin's up days and "
            f"{h['median_beta_down']:.2f} on its down days** — a difference of "
            f"**{h['median_difference']:+.2f}**. The naive two-sample *t* on that difference "
            f"averages {h['median_naive_t']:+.2f}; the block-bootstrapped one, which also "
            f"accounts for the randomness in *which days count as down days*, is "
            f"**{h['median_boot_t']:+.2f}** — smaller by a factor of "
            f"{abs(h['median_naive_t'] / h['median_boot_t']) if h['median_boot_t'] else float('nan'):.1f}. "
            f"The corroborating measurement: {h['share_negative_coskew']:.0%} of the panel has "
            f"negative coskewness, the third-moment fingerprint of the same phenomenon. And the "
            f"control that matters most — under a symmetric simulated world with one beta, a "
            f"naive split still declares asymmetry in **{h['null_false_positive']:.0%}** of "
            f"runs."),
        "trad": trad,
        "trad_why": (
            f"Compounded rather than averaged, the median altcoin captured "
            f"**{h['median_up_capture']:.0%} of Bitcoin's up days and "
            f"{h['median_down_capture']:.0%} of its down days**, and ran a maximum drawdown of "
            f"{h['median_max_dd']:.0%} against Bitcoin's {h['bench_max_dd']:.0%} "
            f"({h['median_dd_ratio']:.2f}×). "
            + ("Since the asymmetry does not clear the bar above, there is no asymmetry trade "
               "here to size — what remains is the plain observation that these are "
               "higher-volatility instruments, which the beta already told you."
               if signal != "Real" else
               "That gap is large enough to matter to position sizing: the leverage you are "
               "buying is not the leverage you are paying for.")),
        "one_sentence": (
            f"The median altcoin's down-beta exceeds its up-beta by {h['median_difference']:+.2f}, "
            f"which sounds decisive until you notice that a symmetric simulated world produces "
            f"the same verdict {h['null_false_positive']:.0%} of the time."),
    }
