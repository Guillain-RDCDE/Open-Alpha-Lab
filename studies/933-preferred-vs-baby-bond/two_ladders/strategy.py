"""Strategy + inference for Study 933 — Same Issuer, Two Ladders.

The construction. For each of eight issuers we hold two rungs of the *same* balance
sheet: the listed **preferred** (junior, perpetual, often non-cumulative) and the listed
**baby bond** (senior or junior subordinated notes, $25 par, dated). Two ladders are
formed as **equal-weight baskets**, rebalanced monthly:

- ``pref`` — equal weight across the eight preferreds;
- ``bond`` — equal weight across the eight baby bonds.

Weights are formed from the month-end state through day ``t`` and applied from day
``t+1`` — **one execution lag, the only one in the study**. Between rebalances the
weights drift with the legs (a real buy-and-hold basket), so turnover — and therefore
cost — is genuinely small.

The race. Both ladders are measured **excess-of-cash** (minus BIL's total return) so the
Sharpe comparison is apples-to-apples, and the headline test is the **pairwise**
difference ``r_pref − r_bond``, issuer by issuer: because both legs are the same credit,
the issuer's spread factor cancels in the difference and what is left is the price of
*seniority alone*. HAC (Newey-West) *t* on the mean difference, a circular block bootstrap
on the Sharpe advantage, an era cut, a cost sweep and a borrow sweep on the long-short.

Costs. One-way bps x NAV on turnover only. Baby bonds and $25-par preferreds are thin —
the default 25 bps one-way is a **PROXY** for a realistic retail-tape half-spread and is
swept from 5 to 100 bps. The long-short leg additionally pays a **borrow ASSUMPTION** on
the short side, swept 0-300 bp/yr, because neither rung is easy to borrow.

Returns are **simple** (arithmetic) so basket returns are exact weighted sums and the
wealth path compounds correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives (mirror of Study 912 / 803)
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


def auto_lags(n: int) -> int:
    """Newey-West bandwidth, the usual 4*(n/100)^(2/9) rule."""
    return int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))


def hac_t(x) -> float:
    """HAC *t* of the mean of a series/array, with the automatic bandwidth."""
    arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    return newey_west_t(arr, lags=auto_lags(len(arr)))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Basket construction (equal weight, monthly rebalance, ONE execution lag)
# --------------------------------------------------------------------------- #
def equal_weight_basket(returns: pd.DataFrame, cost_bps: float = 25.0) -> pd.DataFrame:
    """Equal-weight, monthly-rebalanced buy-and-hold basket of the columns of ``returns``.

    Target weights are set at each month end from information through day ``t`` and are
    **applied from day t+1** (the study's single execution lag). Between rebalances the
    weights drift with the legs, so turnover is the drift correction plus the reset — a
    faithful, low-turnover retail ladder rather than a daily-rebalanced fiction.

    Cost is ``cost_bps`` one-way x NAV x turnover, charged on the rebalance day. Returns a
    frame with ``gross``, ``turnover`` and ``net`` daily simple returns.
    """
    r = returns.dropna(how="any").astype(float)
    if r.empty:
        return pd.DataFrame(columns=["gross", "turnover", "net"])
    n = r.shape[1]
    idx = r.index
    month = pd.Series(idx, index=idx).dt.to_period("M")
    # Rebalance takes EFFECT on the first bar of each new month, i.e. it is decided on the
    # last bar of the previous month -> exactly one day of lag between signal and action.
    effective = np.array(month.ne(month.shift(1)).to_numpy(), dtype=bool)
    effective[0] = True

    w = np.full(n, 1.0 / n)
    gross = np.zeros(len(idx))
    turn = np.zeros(len(idx))
    R = r.to_numpy()
    for i in range(len(idx)):
        if effective[i] and i > 0:
            target = np.full(n, 1.0 / n)
            turn[i] = float(np.abs(target - w).sum())
            w = target
        gross[i] = float(w @ R[i])
        # weights drift with realised returns (buy-and-hold between rebalances)
        grown = w * (1.0 + R[i])
        tot = grown.sum()
        w = grown / tot if tot > 0 else np.full(n, 1.0 / n)

    cost = cost_bps * 1e-4
    out = pd.DataFrame(
        {"gross": gross, "turnover": turn, "net": gross - turn * cost}, index=idx
    )
    return out


# --------------------------------------------------------------------------- #
# Performance summary
# --------------------------------------------------------------------------- #
def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series.

    Pass an **excess-of-cash** series to get the excess Sharpe. Reports CAGR (geometric),
    annualised vol, max drawdown of the series as given, and the HAC *t* on the mean.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    cagr = (float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
            if years > 0 and wealth.iloc[-1] > 0 else float("nan"))
    dd = float((wealth / wealth.cummax() - 1.0).min())
    return {
        "n_days": int(n),
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": dd,
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy(), lags=auto_lags(n)),
    }


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


# --------------------------------------------------------------------------- #
# The pairwise (same-issuer) difference — the heart of the study
# --------------------------------------------------------------------------- #
def pairwise_spread(pref: pd.DataFrame, bond: pd.DataFrame) -> pd.Series:
    """Equal-weight mean across issuers of the daily ``r_pref − r_bond`` difference.

    Because the two legs are the *same* obligor, the issuer's credit factor cancels in the
    difference: what remains is the price of **seniority alone**, free of any issuer or
    sector tilt. The cash leg cancels too, so this series is identical gross of cash.

    This is a **statistical estimator, not a portfolio**. It takes an arithmetic mean of
    daily percentage differences — which implicitly re-equal-weights the eight pairs every
    day and charges **no cost** — so it is *not* the return of anything you could hold.
    Two consequences, both named in the write-up: (i) the tradable expression is the
    monthly-rebalanced, costed ladder spread (:func:`compare` -> ``ret_spread_ann``) and the
    dollar-neutral :func:`long_short`; (ii) on a leg that crashes and rebounds the
    arithmetic mean overstates the compounded outcome by roughly sigma^2/2 — the Jensen
    trap, which on this panel flips the sign of the biggest single contributor.
    """
    common = pref.index.intersection(bond.index)
    cols = [c for c in pref.columns if c in bond.columns]
    d = pref.loc[common, cols] - bond.loc[common, cols]
    return d.mean(axis=1).rename("pref_minus_bond")


def per_pair_table(pref: pd.DataFrame, bond: pd.DataFrame,
                   periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """Issuer-by-issuer: annualised return and vol of each rung, and the paired spread.

    One row per issuer, with the HAC *t* on that issuer's own (preferred − bond) daily
    difference — the honest cross-sectional picture behind the panel average.
    """
    common = pref.index.intersection(bond.index)
    rows = []
    for c in pref.columns:
        if c not in bond.columns:
            continue
        p = pref.loc[common, c].dropna()
        b = bond.loc[common, c].dropna()
        d = (p - b).dropna()
        rows.append({
            "issuer": c,
            "pref_cagr": float((1.0 + p).prod() ** (periods_per_year / len(p)) - 1.0),
            "bond_cagr": float((1.0 + b).prod() ** (periods_per_year / len(b)) - 1.0),
            "pref_vol": float(p.std(ddof=1) * np.sqrt(periods_per_year)),
            "bond_vol": float(b.std(ddof=1) * np.sqrt(periods_per_year)),
            "pref_dd": max_drawdown(p),
            "bond_dd": max_drawdown(b),
            "spread_ann": float(d.mean() * periods_per_year),
            "spread_t": hac_t(d),
        })
    return pd.DataFrame(rows).set_index("issuer")


# --------------------------------------------------------------------------- #
# The ladder race (the headline)
# --------------------------------------------------------------------------- #
def compare(
    pref: pd.DataFrame,
    bond: pd.DataFrame,
    cash: pd.Series,
    cost_bps: float = 25.0,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """Race the preferred ladder against the baby-bond ladder on the same eight issuers.

    Both ladders are equal-weight, monthly-rebalanced, net of ``cost_bps`` one-way, and
    reported **excess-of-cash** (minus the BIL total-return leg). Returns per-ladder
    summaries, the excess-Sharpe advantage (preferred − bond), the HAC *t* on the daily
    *net ladder* return difference, and the pairwise same-issuer spread with its own HAC *t*.
    """
    common = pref.index.intersection(bond.index).intersection(cash.index)
    p, b, c = pref.loc[common], bond.loc[common], cash.loc[common].astype(float)

    bk_p = equal_weight_basket(p, cost_bps=cost_bps)
    bk_b = equal_weight_basket(b, cost_bps=cost_bps)

    e_pref = (bk_p["net"] - c).rename("e_pref")
    e_bond = (bk_b["net"] - c).rename("e_bond")

    s_pref, s_bond = summary(e_pref), summary(e_bond)
    diff = (bk_p["net"] - bk_b["net"]).rename("ladder_diff")
    pw = pairwise_spread(p, b)

    return {
        "pref": s_pref,
        "bond": s_bond,
        "dd_pref_abs": max_drawdown(bk_p["net"]),
        "dd_bond_abs": max_drawdown(bk_b["net"]),
        "excess_sharpe_adv": s_pref["sharpe"] - s_bond["sharpe"],
        "ret_spread_ann": float(diff.mean() * periods_per_year),
        "t_ladder_diff": hac_t(diff),
        "pairwise_spread_ann": float(pw.mean() * periods_per_year),
        "t_pairwise": hac_t(pw),
        "vol_ratio": s_pref["vol_ann"] / s_bond["vol_ann"] if s_bond["vol_ann"] > 0 else float("nan"),
        "n_days": int(len(common)),
        "turnover_ann_pref": float(bk_p["turnover"].sum() * periods_per_year / max(len(common), 1)),
        "e_pref": e_pref, "e_bond": e_bond, "diff": diff, "pairwise": pw,
        "bk_pref": bk_p, "bk_bond": bk_b,
    }


# --------------------------------------------------------------------------- #
# Bootstrap Sharpe / spread CI (circular block bootstrap)
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    series: pd.Series,
    stat: str = "sharpe",
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 933,
    periods_per_year: int = TRADING_DAYS,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe or mean of a daily series.

    Blocks of ``block`` consecutive days preserve the vol clustering that makes naive
    i.i.d. bootstraps over-confident on credit tapes. ``stat='sharpe'`` returns the
    annualised Sharpe, ``stat='mean'`` the annualised mean return.
    """
    r = np.asarray(pd.Series(series).dropna(), dtype=float)
    n = r.size
    ann = np.sqrt(periods_per_year) if stat == "sharpe" else periods_per_year
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": n}

    def _f(x: np.ndarray) -> float:
        if stat == "sharpe":
            sd = x.std(ddof=1)
            return float(x.mean() / sd * ann) if sd > 0 else float("nan")
        return float(x.mean() * ann)

    point = _f(r)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[i] = _f(r[idx])
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point, "ci_low": float(lo), "ci_high": float(hi),
        "frac_negative": float((valid < 0).mean()), "n_obs": int(n),
        "n_boot": int(valid.size), "block": int(block), "stat": stat,
    }


def bootstrap_sharpe_adv_ci(
    e_a: pd.Series,
    e_b: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 933,
    periods_per_year: int = TRADING_DAYS,
    alpha: float = 0.05,
) -> dict:
    """Block-bootstrap CI for the **Sharpe advantage** Sharpe(a) − Sharpe(b).

    Blocks are drawn *jointly* for the two arms, so the strong contemporaneous correlation
    between the two rungs of the same balance sheets is preserved — resampling them
    independently would inflate the apparent precision of the difference.
    """
    common = e_a.dropna().index.intersection(e_b.dropna().index)
    a = np.asarray(e_a.loc[common], dtype=float)
    b = np.asarray(e_b.loc[common], dtype=float)
    n = a.size
    ann = np.sqrt(periods_per_year)
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}

    def _sh(x):
        sd = x.std(ddof=1)
        return x.mean() / sd * ann if sd > 0 else np.nan

    point = float(_sh(a) - _sh(b))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[i] = _sh(a[idx]) - _sh(b[idx])
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": int(n),
            "n_boot": int(valid.size), "block": int(block)}


# --------------------------------------------------------------------------- #
# Era cut, cost sweep, borrow sweep
# --------------------------------------------------------------------------- #
def era_cut(pref: pd.DataFrame, bond: pd.DataFrame, cash: pd.Series,
            split: str = "2024-04-01", cost_bps: float = 25.0) -> dict:
    """Split the sample at ``split`` and re-run the race on each half.

    A real seniority premium should keep its sign in *both* halves; one driven by a single
    episode (a rate shock, one issuer's blow-up) will not.
    """
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        p, b = pref.loc[sl], bond.loc[sl]
        c = cash.loc[sl]
        if len(p) < 60:
            out[tag] = None
            continue
        cmp = compare(p, b, c, cost_bps=cost_bps)
        out[tag] = {
            "n_days": cmp["n_days"],
            "sharpe_pref": cmp["pref"]["sharpe"],
            "sharpe_bond": cmp["bond"]["sharpe"],
            "excess_sharpe_adv": cmp["excess_sharpe_adv"],
            "pairwise_spread_ann": cmp["pairwise_spread_ann"],
            "t_pairwise": cmp["t_pairwise"],
            "dd_pref_abs": cmp["dd_pref_abs"],
            "dd_bond_abs": cmp["dd_bond_abs"],
        }
    return out


def cost_sweep(pref: pd.DataFrame, bond: pd.DataFrame, cash: pd.Series,
               cost_grid=(5.0, 15.0, 25.0, 50.0, 100.0)) -> list[dict]:
    """Sweep the one-way transaction-cost PROXY across a thin-tape plausible range.

    Both ladders are low turnover (monthly reset of a drifted equal weight), so cost is
    close to symmetric across the two arms and mostly cancels in the spread — which is the
    point: the *pairwise* estimator is nearly cost-free, the *long-short* is not.
    """
    rows = []
    for c in cost_grid:
        cmp = compare(pref, bond, cash, cost_bps=c)
        rows.append({
            "cost_bps": c,
            "sharpe_pref": cmp["pref"]["sharpe"],
            "sharpe_bond": cmp["bond"]["sharpe"],
            "excess_sharpe_adv": cmp["excess_sharpe_adv"],
            "ret_spread_ann": cmp["ret_spread_ann"],
            "t_ladder_diff": cmp["t_ladder_diff"],
        })
    return rows


def long_short(pref: pd.DataFrame, bond: pd.DataFrame, cash: pd.Series,
               cost_bps: float = 25.0, borrow_bps_ann: float = 100.0,
               long_leg: str = "pref", periods_per_year: int = TRADING_DAYS) -> dict:
    """The dollar-neutral trade: long one ladder, short the other, financed at cash.

    ``borrow_bps_ann`` is an **ASSUMPTION** — neither $25-par preferreds nor baby bonds are
    general-collateral easy-to-borrow, so it is swept rather than asserted. The short leg
    pays borrow daily on NAV; both legs pay the one-way cost on their own turnover. Being
    dollar-neutral, the position earns no cash rate on top (the short proceeds finance the
    long), so the series is already an excess return.
    """
    common = pref.index.intersection(bond.index).intersection(cash.index)
    bk_p = equal_weight_basket(pref.loc[common], cost_bps=cost_bps)
    bk_b = equal_weight_basket(bond.loc[common], cost_bps=cost_bps)
    borrow_d = borrow_bps_ann * 1e-4 / periods_per_year
    if long_leg == "pref":
        r = bk_p["net"] - bk_b["net"] - borrow_d
    else:
        r = bk_b["net"] - bk_p["net"] - borrow_d
    s = summary(r)
    return {
        "long_leg": long_leg, "borrow_bps_ann": borrow_bps_ann, "cost_bps": cost_bps,
        "ann_return": float(r.mean() * periods_per_year),
        "sharpe": s["sharpe"], "vol_ann": s["vol_ann"],
        "max_drawdown": s["max_drawdown"], "t": hac_t(r), "n_days": int(len(r)),
    }


def borrow_sweep(pref: pd.DataFrame, bond: pd.DataFrame, cash: pd.Series,
                 borrow_grid=(0.0, 50.0, 100.0, 200.0, 300.0),
                 cost_bps: float = 25.0, long_leg: str = "pref") -> list[dict]:
    """Sweep the borrow ASSUMPTION on the short leg of the dollar-neutral trade."""
    return [long_short(pref, bond, cash, cost_bps=cost_bps,
                       borrow_bps_ann=bb, long_leg=long_leg) for bb in borrow_grid]


# --------------------------------------------------------------------------- #
# Stress episodes & the seniority question
# --------------------------------------------------------------------------- #
def stress_table(pref: pd.DataFrame, bond: pd.DataFrame,
                 episodes: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Cumulative return of each ladder through named stress windows.

    If seniority is worth anything it should show up *here*: the senior rung is supposed to
    lose less when the issuer's credit is questioned.
    """
    rows = []
    for name, (a, z) in episodes.items():
        p = pref.loc[a:z]
        b = bond.loc[a:z]
        if len(p) < 5:
            continue
        rows.append({
            "episode": name,
            "days": int(len(p)),
            "pref_pct": float(((1.0 + p.mean(axis=1)).prod() - 1.0) * 100),
            "bond_pct": float(((1.0 + b.mean(axis=1)).prod() - 1.0) * 100),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["gap_pp"] = df["pref_pct"] - df["bond_pct"]
        df = df.set_index("episode")
    return df


def staleness_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-leg microstructure diagnostics: share of exactly-zero days and lag-1 autocorrelation.

    $25-par preferreds and baby bonds are **thin retail tapes**. Two artefacts follow and
    both distort a *daily* risk comparison:

    - a high share of exactly-zero return days = a **stale** print (no trade, the close is
      carried forward), which *understates* measured vol;
    - a strongly **negative** lag-1 autocorrelation = **bid-ask bounce**, the close ping-ponging
      between bid and offer, which *overstates* measured vol.

    Neither is a property of the capital structure, so any risk claim taken from daily data
    has to be re-checked at a lower frequency (see :func:`vol_ratio_by_frequency`).
    """
    rows = []
    for c in returns.columns:
        x = pd.Series(returns[c]).astype(float).dropna()
        rows.append({
            "leg": c,
            "n": int(len(x)),
            "zero_share": float((x.abs() < 1e-12).mean()) if len(x) else float("nan"),
            "ac1": float(x.autocorr(1)) if len(x) > 2 else float("nan"),
            "vol_ann": float(x.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(x) > 1 else float("nan"),
        })
    return pd.DataFrame(rows).set_index("leg")


_FREQS = (("daily", None, TRADING_DAYS), ("weekly", "W-FRI", 52), ("monthly", "ME", 12))


def _resample(df, rule):
    """Compound daily simple returns up to ``rule`` (None = leave daily)."""
    if rule is None:
        return df
    return (1.0 + df).resample(rule).prod() - 1.0


def vol_ratio_by_frequency(pref: pd.DataFrame, bond: pd.DataFrame) -> pd.DataFrame:
    """Is the junior rung *really* the riskier one, or is that a daily-tape artefact?

    Recomputes the equal-weight ladder vol ratio **and** the per-pair count of
    "preferred more volatile than bond" at daily, weekly and monthly frequency. Staleness
    and bid-ask bounce (see :func:`staleness_table`) live at the daily horizon and wash out
    under compounding, so a risk ordering that survives to monthly is a property of the
    instruments and one that does not is a property of the tape.
    """
    rows = []
    for name, rule, ann in _FREQS:
        p, b = _resample(pref, rule), _resample(bond, rule)
        lp, lb = p.mean(axis=1), b.mean(axis=1)
        sp, sb = float(lp.std(ddof=1)), float(lb.std(ddof=1))
        cols = [c for c in p.columns if c in b.columns]
        riskier = int(sum(p[c].std(ddof=1) > b[c].std(ddof=1) for c in cols))
        rows.append({
            "freq": name, "n_obs": int(len(p)),
            "vol_pref": sp * np.sqrt(ann), "vol_bond": sb * np.sqrt(ann),
            "vol_ratio": sp / sb if sb > 0 else float("nan"),
            "pref_riskier": riskier, "n_pairs": len(cols),
        })
    return pd.DataFrame(rows).set_index("freq")


def drop_issuers(pref: pd.DataFrame, bond: pd.DataFrame, cash: pd.Series,
                 drop, cost_bps: float = 25.0) -> dict:
    """Re-run the whole race with ``drop`` (a list of issuer labels) removed.

    The concentration probe. A panel result that is a *property of seniority* should survive
    the removal of any one or two obligors; one that is a property of a **distress episode**
    will not — and neither will its risk ordering, because a distressed issuer's junior rung
    moves like equity for reasons that have nothing to do with the price of subordination.
    """
    keep = [c for c in pref.columns if c not in set(drop)]
    p, b = pref[keep], bond[keep]
    cmp = compare(p, b, cash, cost_bps=cost_bps)
    freq = vol_ratio_by_frequency(p, b)
    return {
        "dropped": list(drop), "n_pairs": len(keep),
        "pairwise_spread_ann": cmp["pairwise_spread_ann"], "t_pairwise": cmp["t_pairwise"],
        "excess_sharpe_adv": cmp["excess_sharpe_adv"], "vol_ratio": cmp["vol_ratio"],
        "dd_pref_abs": cmp["dd_pref_abs"], "dd_bond_abs": cmp["dd_bond_abs"],
        "vol_ratio_monthly": float(freq.loc["monthly", "vol_ratio"]),
        "pref_riskier_monthly": int(freq.loc["monthly", "pref_riskier"]),
    }


def hit_rate(pref: pd.DataFrame, bond: pd.DataFrame) -> dict:
    """How many of the issuer pairs show a positive (preferred − bond) spread?

    A panel average driven by one or two names is not a seniority premium; a real one wins
    in most pairs. Reported with a Wilson interval on the win share.
    """
    tbl = per_pair_table(pref, bond)
    k = int((tbl["spread_ann"] > 0).sum())
    n = int(len(tbl))
    lo, hi = wilson_interval(k, n)
    return {"wins": k, "n": n, "share": k / n if n else float("nan"),
            "ci_low": lo, "ci_high": hi}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(pref: pd.DataFrame, bond: pd.DataFrame, cash: pd.Series,
                     cost_bps: float = 0.0) -> dict:
    """Run the pairwise estimator on a synthetic panel.

    On a panel with a *planted* junior carry premium the pairwise spread must recover it
    (positive, HAC *t* large); on the exact null the spread must be centred on zero. Proves
    the estimator is unbiased — it never supports a real-tape stamp.
    """
    cmp = compare(pref, bond, cash, cost_bps=cost_bps)
    return {
        "pairwise_spread_ann": cmp["pairwise_spread_ann"],
        "t_pairwise": cmp["t_pairwise"],
        "excess_sharpe_adv": cmp["excess_sharpe_adv"],
        "sharpe_pref": cmp["pref"]["sharpe"],
        "sharpe_bond": cmp["bond"]["sharpe"],
        "vol_ratio": cmp["vol_ratio"],
        "n_days": cmp["n_days"],
    }
