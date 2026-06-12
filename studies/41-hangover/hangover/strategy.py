"""The omen, and the only fair way to score it.

The January Barometer makes a *directional* claim: sign(January) predicts sign(rest-of-year). The
honest test is not "is the rest-of-year up after an up January" (it almost always is — stocks rise
most years) but "does conditioning on January beat the **base rate** of just predicting up?" Plus a
tradable version (sit in cash for the rest of the year after a down January, crediting the T-bill)
measured against simply holding the index.

Every conditional cell here rides on a handful of years (30 down Januaries in 76), so each
proportion carries a **Wilson interval** and each claimed difference a test — Fisher's exact for
proportions, a seeded permutation test for the rest-of-year mean gap. Small-n honesty is the study.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

Z95 = 1.959963984540054


def annual_table(monthly_ret: pd.Series) -> pd.DataFrame:
    """Collapse a monthly-return series into yearly ``(jan, roy)`` pairs.

    ``jan`` = the January return; ``roy`` = the compounded February-through-December return. Years
    without a January or with fewer than six months are dropped, as is the final partial year.
    """
    df = pd.DataFrame({"ret": pd.Series(monthly_ret).astype(float)})
    df["year"] = df.index.year
    df["month"] = df.index.month
    rows = []
    for y, g in df.groupby("year"):
        if 1 in set(g["month"]) and len(g) >= 6 and g["month"].max() >= 11:
            jan = g.loc[g["month"] == 1, "ret"].iloc[0]
            roy = (1.0 + g.loc[g["month"] >= 2, "ret"]).prod() - 1.0
            rows.append((y, jan, roy))
    return pd.DataFrame(rows, columns=["year", "jan", "roy"]).set_index("year")


def base_rate(tbl: pd.DataFrame) -> float:
    """The unconditional P(rest-of-year > 0) — the bar the omen has to clear."""
    return float((tbl["roy"] > 0).mean())


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k/n`` (sane at small n and 0/100%)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def fisher_exact_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for two binomial proportions ``k1/n1`` vs ``k2/n2``.

    Pure stdlib (hypergeometric with ``math.comb``): the probability, under a common rate, of a
    2×2 table as or more extreme than the observed one. The honest test for the omen's conditional
    cells, where n is 23–53 years and a normal approximation flatters the difference.
    """
    if min(n1, n2) == 0:
        return float("nan")
    total, k = n1 + n2, k1 + k2
    denom = math.comb(total, k)
    lo, hi = max(0, k - n2), min(n1, k)
    p_obs = math.comb(n1, k1) * math.comb(n2, k - k1) / denom
    p = 0.0
    for x in range(lo, hi + 1):
        px = math.comb(n1, x) * math.comb(n2, k - x) / denom
        if px <= p_obs * (1.0 + 1e-9):
            p += px
    return float(min(1.0, p))


def roy_mean_permutation_p(tbl: pd.DataFrame, n_iter: int = 20000, seed: int = 41) -> float:
    """Permutation p-value (two-sided) for the up-vs-down-January rest-of-year **mean** gap.

    Shuffles the January labels ``n_iter`` times (seeded — deterministic) and asks how often a
    random split produces a mean gap at least as large as the observed one.
    """
    rng = np.random.default_rng(seed)
    roy = tbl["roy"].to_numpy(dtype=float)
    up = (tbl["jan"] > 0).to_numpy()
    n_up = int(up.sum())
    if n_up == 0 or n_up == len(roy):
        return float("nan")
    obs = abs(roy[up].mean() - roy[~up].mean())
    count = 0
    for _ in range(n_iter):
        perm = rng.permutation(len(roy)) < n_up
        if abs(roy[perm].mean() - roy[~perm].mean()) >= obs - 1e-15:
            count += 1
    return float(count / n_iter)


def barometer_accuracy(tbl: pd.DataFrame) -> float:
    """Directional accuracy: how often sign(January) matches sign(rest-of-year)."""
    return float((np.sign(tbl["jan"]) == np.sign(tbl["roy"])).mean())


def conditional_means(tbl: pd.DataFrame) -> pd.DataFrame:
    """Rest-of-year mean and P(up) with its Wilson 95% interval, split by the sign of January (the
    believers' headline split). On 30–46 observations the intervals are wide — that *is* the point."""
    out = {}
    for lab, mask in [("jan_up", tbl["jan"] > 0), ("jan_down", tbl["jan"] <= 0), ("unconditional", tbl["roy"].notna())]:
        s = tbl.loc[mask, "roy"]
        k, n = int((s > 0).sum()), int(len(s))
        lo, hi = wilson_ci(k, n)
        out[lab] = {"n": n, "roy_mean": float(s.mean()), "p_up": float(k / n) if n else float("nan"),
                    "wilson_low": lo, "wilson_high": hi}
    return pd.DataFrame(out).T


def split_tests(tbl: pd.DataFrame) -> dict:
    """The tests behind the headline split: Fisher exact on P(up | jan up) vs P(up | jan down), and a
    seeded permutation p on the rest-of-year mean gap. Returns the counts too, so nothing hides."""
    up, down = tbl[tbl["jan"] > 0], tbl[tbl["jan"] <= 0]
    k1, n1 = int((up["roy"] > 0).sum()), int(len(up))
    k2, n2 = int((down["roy"] > 0).sum()), int(len(down))
    return {
        "k_up": k1, "n_up": n1, "k_down": k2, "n_down": n2,
        "fisher_p": fisher_exact_p(k1, n1, k2, n2),
        "mean_gap": float(up["roy"].mean() - down["roy"].mean()),
        "mean_perm_p": roy_mean_permutation_p(tbl),
    }


def accuracy_with_ci(tbl: pd.DataFrame) -> dict:
    """Directional accuracy and the base rate, each with hit counts and a Wilson 95% interval."""
    hits = int((np.sign(tbl["jan"]) == np.sign(tbl["roy"])).sum())
    ups = int((tbl["roy"] > 0).sum())
    n = int(len(tbl))
    a_lo, a_hi = wilson_ci(hits, n)
    b_lo, b_hi = wilson_ci(ups, n)
    return {"n": n, "acc_k": hits, "accuracy": hits / n, "acc_low": a_lo, "acc_high": a_hi,
            "base_k": ups, "base_rate": ups / n, "base_low": b_lo, "base_high": b_hi}


def decay_split(tbl: pd.DataFrame, split_year: int = 1972) -> dict:
    """The pre/post-publication split, with the test the comparison deserves.

    Directional accuracy before vs after ``split_year`` (Fisher exact on the hit counts — the
    subsamples are ~23 vs ~53 years, far too small to eyeball), plus each era's down-January
    rest-of-year mean. Returns counts, accuracies, Wilson bounds and the Fisher p.
    """
    out = {}
    eras = {"pre": tbl[tbl.index <= split_year], "post": tbl[tbl.index > split_year]}
    for lab, sl in eras.items():
        hits, n = int((np.sign(sl["jan"]) == np.sign(sl["roy"])).sum()), int(len(sl))
        lo, hi = wilson_ci(hits, n)
        down = sl[sl["jan"] <= 0]
        out.update({f"{lab}_k": hits, f"{lab}_n": n, f"{lab}_acc": hits / n if n else float("nan"),
                    f"{lab}_low": lo, f"{lab}_high": hi,
                    f"{lab}_down_mean": float(down["roy"].mean()) if len(down) else float("nan")})
    out["fisher_p"] = fisher_exact_p(out["pre_k"], out["pre_n"], out["post_k"], out["post_n"])
    return out


def cash_roy(tbill_yield: pd.Series) -> pd.Series:
    """Feb–Dec cash return per year from a monthly annualised T-bill yield series.

    Compounds ``yield/12`` across each year's February–December prints. Years with no yield data
    (^IRX starts in 1960) get zero — the conservative, stated choice for the 1950s cash years.
    """
    y = pd.Series(tbill_yield).astype(float).dropna()
    feb_dec = y[y.index.month >= 2]
    out = (1.0 + feb_dec / 12.0).groupby(feb_dec.index.year).prod() - 1.0
    out.index.name = "year"
    return out.rename("cash_roy")


def barometer_returns(tbl: pd.DataFrame, cash: pd.Series | None = None) -> pd.Series:
    """The tradable omen: capture the rest-of-year return after an up January, else sit in cash.

    ``cash`` is an optional per-year Feb–Dec cash return (see :func:`cash_roy`) credited in the
    down-January years — leaving cash at zero short-changes the rule on its ~30 cash years.
    """
    cash_leg = np.zeros(len(tbl)) if cash is None else \
        pd.Series(cash).reindex(tbl.index).fillna(0.0).to_numpy(dtype=float)
    return pd.Series(np.where(tbl["jan"] > 0, tbl["roy"], cash_leg), index=tbl.index, name="barometer")


def buy_hold_roy(tbl: pd.DataFrame) -> pd.Series:
    """Benchmark: always hold the index for the rest of the year (the thing the omen must beat)."""
    return tbl["roy"].rename("buy_hold")


def summary(annual_returns: pd.Series) -> dict:
    """Annualised stats for a *yearly* return series (CAGR, annual Sharpe, max-drawdown, hit-rate)."""
    r = pd.Series(annual_returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("cagr", "sharpe", "vol", "max_drawdown", "hit_rate", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    cagr = eq.iloc[-1] ** (1.0 / len(r)) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "cagr": float(cagr),
        "sharpe": float(mean / std) if std > 0 else np.nan,  # yearly obs → already annual
        "vol": float(std),
        "max_drawdown": float(dd),
        "hit_rate": float((r > 0).mean()),
        "n": int(len(r)),
    }
