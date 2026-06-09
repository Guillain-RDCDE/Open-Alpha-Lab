"""The teardown engine — turn the session panel into the numbers that earn the verdict.

Four questions, four tools:

1. **Is a conditional rate even distinguishable from its baseline on this sample?**
   :func:`rate` (point + Wilson interval) and :func:`conditional_table` lay every cell of
   "P(close red | morning condition)" side by side with an honest small-sample interval and
   the lift over baseline. On n = 25 the Wilson interval is wide on purpose — that *is* the
   first finding.

2. **How much of the headline is forecasting vs a mechanical head-start?**
   :func:`mechanical_vs_predictive` splits "P(session red | OC-red)" — which is already
   half-committed the moment the morning is red — from "P(rest-of-day red | OC-red)", the part
   actually unknown at 10:30. The collapse between the two is the load-bearing result.

3. **Does the second signal add anything over the first?**
   :func:`ib_increment` compares the close-red rate of the full confluence against OC-red-but-
   *not*-rejected (a two-proportion test + Fisher exact). Under the baked-in synthetic null the
   increment is ~0; the real run asks whether reality differs.

4. **How easily does a prompt that mines many "confluences" manufacture an 88%?**
   :func:`mining_inflation` Monte-Carlos the *best* observed rate across a bank of candidate
   confluences, each measured on a tiny sample, when the *true* edge is modest — showing how
   far selection + small-n inflate the number you end up quoting.

Binomial intervals and the beta-binomial posterior use SciPy; the rest is NumPy/stdlib.
Everything is deterministic given a seed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

Z95 = 1.959963984540054


# --------------------------------------------------------------------------- #
# Proportions, intervals, posteriors
# --------------------------------------------------------------------------- #

def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k/n`` (sane at small n and 0/100%)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def beta_binomial(k: int, n: int, prior_a: float = 1.0, prior_b: float = 1.0,
                  cred: float = 0.95, thresholds: tuple[float, ...] = ()) -> dict:
    """Posterior on the true rate under a Beta(``prior_a``, ``prior_b``) prior (uniform by default).

    Returns the posterior mean, the equal-tailed credible interval, and — for each value in
    ``thresholds`` — the posterior probability the true rate exceeds it. This is the honest read
    on "22 of 25": a point estimate of 88% with a posterior that still puts real mass well below it.
    """
    from scipy.stats import beta as _beta

    a, b = prior_a + k, prior_b + (n - k)
    lo, hi = _beta.ppf([(1 - cred) / 2, 1 - (1 - cred) / 2], a, b)
    out = {
        "k": int(k), "n": int(n),
        "posterior_mean": float(a / (a + b)),
        "cred_low": float(lo), "cred_high": float(hi),
    }
    for t in thresholds:
        out[f"P(rate>{t:g})"] = float(_beta.sf(t, a, b))
    return out


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Two-sided two-proportion z-test of ``k1/n1`` vs ``k2/n2`` (pooled SE)."""
    if n1 == 0 or n2 == 0:
        return {"p1": float("nan"), "p2": float("nan"), "diff": float("nan"),
                "z": float("nan"), "p_value": float("nan")}
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"p1": float(p1), "p2": float(p2), "diff": float(p1 - p2),
            "z": float(z), "p_value": float(p)}


def _align(cond: pd.Series, event: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    cond = cond.astype("boolean")
    event = event.astype("boolean")
    valid = cond.notna() & event.notna()
    c = cond[valid].to_numpy(dtype=bool)
    e = event[valid].to_numpy(dtype=bool)
    return c, e


def rate(cond: pd.Series, event: pd.Series, z: float = Z95) -> dict:
    """P(event | condition): the count, the rate, and its Wilson interval (NA rows dropped)."""
    c, e = _align(cond, event)
    n = int(c.sum())
    k = int((c & e).sum())
    lo, hi = wilson_ci(k, n, z)
    return {"k": k, "n": n, "rate": (k / n if n else float("nan")),
            "wilson_low": lo, "wilson_high": hi}


def conditional_table(masks: dict[str, pd.Series], event: pd.Series,
                      baseline_key: str = "baseline") -> pd.DataFrame:
    """One row per condition: k, n, rate, Wilson CI, and the lift (pp) over the baseline rate."""
    rows = {name: rate(mask, event) for name, mask in masks.items()}
    df = pd.DataFrame(rows).T
    base = df.loc[baseline_key, "rate"]
    df["lift_pp"] = (df["rate"] - base) * 100
    df = df[["k", "n", "rate", "wilson_low", "wilson_high", "lift_pp"]]
    df[["k", "n"]] = df[["k", "n"]].astype(int)
    return df


# --------------------------------------------------------------------------- #
# Mechanical head-start vs genuine forecast
# --------------------------------------------------------------------------- #

def mechanical_vs_predictive(feat: pd.DataFrame) -> dict:
    """Split "OC-red -> red close" into a mechanical head-start and a genuine forecast.

    The pitch quotes **P(session red | OC-red)** — but by 10:30 an OC-red day is already trading
    below its open, so most of that probability is just "it didn't fully recover", not a forecast.
    The honestly forecastable quantity is **P(rest-of-day red | OC-red)** — whether 10:30->16:00
    *continues* down — measured against its own baseline P(rest-of-day red). Returns both
    conditional rates, both baselines, the two lifts, and the share of the headline lift that is
    the mechanical head-start rather than continuation.
    """
    from .signals import oc_red, session_red, rest_red

    oc = oc_red(feat)
    sess = rate(oc, session_red(feat))
    base_sess = rate(pd.Series(True, index=feat.index, dtype="boolean"), session_red(feat))
    cont = rate(oc, rest_red(feat))
    base_cont = rate(pd.Series(True, index=feat.index, dtype="boolean"), rest_red(feat))

    headline_lift = sess["rate"] - base_sess["rate"]
    continuation_lift = cont["rate"] - base_cont["rate"]
    mechanical_lift = headline_lift - continuation_lift
    return {
        "headline_rate": sess["rate"],          # P(session red | OC-red) — what's sold
        "headline_baseline": base_sess["rate"],
        "headline_lift_pp": headline_lift * 100,
        "continuation_rate": cont["rate"],       # P(rest red | OC-red) — the real forecast
        "continuation_baseline": base_cont["rate"],
        "continuation_lift_pp": continuation_lift * 100,
        "mechanical_lift_pp": mechanical_lift * 100,
        "mechanical_share": (mechanical_lift / headline_lift if headline_lift else float("nan")),
        "n_oc_red": sess["n"],
    }


def welch_t(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch's unequal-variance t-test of mean(a) vs mean(b) (two-sided, normal-tail p)."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    na, nb = a.size, b.size
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else 0.0
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "t": float(t),
            "p_value": float(math.erfc(abs(t) / math.sqrt(2)))}


def continuation_test(feat: pd.DataFrame) -> dict:
    """Is the genuine forecast — afternoon continuation — statistically real?

    Tests whether the rest-of-day (10:30→16:00) differs after a red opening hour vs a green one,
    two ways: a two-proportion z on the *sign* (P(rest red | OC-red) vs P(rest red | OC-green))
    and a Welch t on the *return* (mean rest_ret by group, in bps). A significant contrast means
    the morning carries real directional information into the afternoon — the small, true core of
    the claim, separate from whether it is *tradable* (see :func:`afternoon_short_backtest`).
    """
    from .signals import oc_red, rest_red

    oc = oc_red(feat)
    rr = rest_red(feat)
    red = rate(oc, rr)
    grn = rate(~oc, rr)
    z = two_proportion_z(red["k"], red["n"], grn["k"], grn["n"])

    rest = feat["rest_ret"].to_numpy(float)
    ocb = oc.fillna(False).to_numpy(bool)
    wt = welch_t(rest[ocb], rest[~ocb])
    return {
        "rest_red_oc_red": red["rate"], "rest_red_oc_green": grn["rate"],
        "sign_diff_pp": (red["rate"] - grn["rate"]) * 100, "sign_z": z["z"],
        "sign_p_value": z["p_value"],
        "mean_rest_oc_red_bps": wt["mean_a"] * 1e4, "mean_rest_oc_green_bps": wt["mean_b"] * 1e4,
        "mean_diff_bps": wt["diff"] * 1e4, "mean_t": wt["t"], "mean_p_value": wt["p_value"],
        "n_oc_red": red["n"], "n_oc_green": grn["n"],
    }


def afternoon_short_backtest(feat: pd.DataFrame,
                             costs_bps: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0),
                             trading_days: int = 252) -> dict:
    """Beat 6 made into a number: short 10:30→close on every red-opening-hour day, sweep costs.

    The only positive-expectancy reading of the claim is to trade the continuation: on an OC-red
    day, short the rest of the session (pnl = −rest_ret). Reports the gross per-trade mean (bps),
    its t-stat and annualised Sharpe, the win rate, the **break-even round-trip cost** (the cost
    that zeroes the mean), and a net-of-cost table. The decisive honesty check: a gross Sharpe is
    not an edge if its own t-stat is ~0 and the break-even sits inside realistic costs.
    """
    from .signals import oc_red

    oc = oc_red(feat).fillna(False).to_numpy(bool)
    pnl = -feat["rest_ret"].to_numpy(float)[oc]
    pnl = pnl[np.isfinite(pnl)]
    n = pnl.size
    if n < 2:
        return {"n_trades": n}
    mean = float(pnl.mean())
    sd = float(pnl.std(ddof=1))
    years = len(feat) / trading_days
    trades_per_year = n / years if years > 0 else float("nan")
    t_stat = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
    gross_sharpe = mean / sd * math.sqrt(trades_per_year) if sd > 0 else float("nan")

    net = {}
    for c in costs_bps:
        m = mean - c * 1e-4
        net[c] = {"net_mean_bps": m * 1e4,
                  "net_sharpe": (m / sd * math.sqrt(trades_per_year)) if sd > 0 else float("nan")}
    return {
        "n_trades": n, "trades_per_year": trades_per_year,
        "gross_mean_bps": mean * 1e4, "gross_t": t_stat, "gross_sharpe": gross_sharpe,
        "win_rate": float((pnl > 0).mean()), "break_even_cost_bps": mean * 1e4,
        "net": net,
    }


def ib_increment(feat: pd.DataFrame) -> dict:
    """Does IB-rejection add anything *on top of* OC-red? (confluence vs OC-red-not-rejected).

    Compares the close-red rate of the full confluence (OC-red & IB-rejected) against OC-red days
    that were *not* rejected, with a two-proportion z and a Fisher exact p-value. A near-zero,
    non-significant difference means the second signal is redundant given the first — the baked-in
    truth of the synthetic, and the question the real run answers.
    """
    from .signals import oc_red, ib_high_rejected, session_red

    oc, ib, sess = oc_red(feat), ib_high_rejected(feat), session_red(feat)
    conf = rate(oc & ib, sess)
    ctrl = rate(oc & ~ib, sess)
    z = two_proportion_z(conf["k"], conf["n"], ctrl["k"], ctrl["n"])

    fisher_p = float("nan")
    if conf["n"] and ctrl["n"]:
        from scipy.stats import fisher_exact
        table = [[conf["k"], conf["n"] - conf["k"]], [ctrl["k"], ctrl["n"] - ctrl["k"]]]
        fisher_p = float(fisher_exact(table)[1])

    return {
        "confluence_rate": conf["rate"], "confluence_k": conf["k"], "confluence_n": conf["n"],
        "control_rate": ctrl["rate"], "control_k": ctrl["k"], "control_n": ctrl["n"],
        "increment_pp": (conf["rate"] - ctrl["rate"]) * 100 if ctrl["n"] else float("nan"),
        "z": z["z"], "z_p_value": z["p_value"], "fisher_p_value": fisher_p,
    }


# --------------------------------------------------------------------------- #
# Forking paths — how selection + small-n inflate a modest edge into a headline
# --------------------------------------------------------------------------- #

def mining_inflation(p_true: float, n_cond: int, n_candidates: int,
                     n_sim: int = 20000, observed: float | None = None,
                     seed: int = 0) -> dict:
    """Monte-Carlo the *best* observed rate a prompt finds across many candidate confluences.

    The honest model of "I gave Claude my API key and combined reports until one hit": you
    evaluate ``n_candidates`` confluences, each selecting a small ``n_cond`` sessions whose true
    close-red probability is the *same modest* ``p_true`` (the real edge), and you quote the
    **maximum** observed rate. Returns the expected and 95th-percentile best rate, the inflation
    over ``p_true``, and — if ``observed`` is given — how often pure selection alone reaches it.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    draws = rng.binomial(n_cond, p_true, size=(n_sim, n_candidates)) / n_cond
    best = draws.max(axis=1)
    out = {
        "p_true": float(p_true), "n_cond": int(n_cond), "n_candidates": int(n_candidates),
        "expected_best_rate": float(best.mean()),
        "best_rate_p95": float(np.percentile(best, 95)),
        "inflation_pp": float((best.mean() - p_true) * 100),
    }
    if observed is not None:
        out["observed"] = float(observed)
        out["P(best>=observed)"] = float((best >= observed).mean())
    return out
