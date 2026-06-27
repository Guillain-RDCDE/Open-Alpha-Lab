"""Strategy + inference for Study 516 — Dividend-Month-Premium.

The claim (Hartzmark & Solomon 2013, *The Dividend Month Premium*, JFE): stocks earn
abnormally high returns in the calendar months they are *predicted* to pay a dividend —
a price-pressure effect from yield-seeking demand that clusters around the predictable
payment schedule. We measure it two complementary ways:

    * **Per-month premium.** Pool every stock-month; compare the mean monthly return on
      **predicted dividend months** against the mean on **non-dividend months**. The premium
      is predicted-mean minus rest-mean (in basis points per month), with a Welch t.

    * **Tradable monthly overlay.** A long-only strategy that holds a name equal-weighted ONLY
      in months it is *predicted* (one month ahead, from its past cadence — no look-ahead) to
      pay a dividend, versus always-invested buy & hold. We compare the mean monthly return
      *while in a predicted-dividend month* to buy & hold, and charge per-entry turnover costs.

Inference, the desk's shared spirit:

  * a **one-sample t** of the per-name predicted-vs-baseline premium against 0 (the primary
    real-tape test) and a **Welch t** of pooled predicted-months vs rest-months;
  * a **placebo / label-shuffle** null — randomly relocate the predicted-dividend flags within
    each name's calendar (preserving the count of "predicted" months), and ask how often a
    random tag set yields as large a premium (the honest test for a calendar effect);
  * **one-month execution lag** (the flag is known a month ahead; we enter the close at the
    start of the predicted month) and **one-way costs × turnover** on the tradable overlay.

The decisive number is the per-name premium t on the REAL tape, and whether the overlay's edge
survives the per-entry turnover cost of trading a calendar that fires only the dividend months.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12.0


# --------------------------------------------------------------------------- #
# Per-month premium (pooled predicted vs rest)
# --------------------------------------------------------------------------- #
def pooled_returns(flags: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pool all stock-months into (predicted-month returns, rest-month returns).

    ``flags`` is ``{ticker: (periods, ret, is_predicted)}`` from ``data.build_predicted_flags``
    (or the synthetic equivalent).
    """
    pred, rest = [], []
    for _tk, (_p, ret, fl) in flags.items():
        pred.append(ret[fl])
        rest.append(ret[~fl])
    pred = np.concatenate(pred) if pred else np.array([])
    rest = np.concatenate(rest) if rest else np.array([])
    return pred, rest


def per_name_premium(flags: dict) -> np.ndarray:
    """One predicted-vs-baseline premium per NAME (the within-name effect).

    For each name: mean monthly return in its predicted-dividend months minus its mean monthly
    return in its non-dividend months. A within-name baseline, so cross-name level differences
    cancel. Returns one premium per name — the sample for the primary one-sample t. (Hartzmark-
    Solomon's effect is a within-firm comparison of dividend vs non-dividend months, so one
    observation per firm is the natural, conservative unit; it also avoids the per-month
    pseudo-replication that would inflate a naive t.)
    """
    out = []
    for _tk, (_p, ret, fl) in flags.items():
        pred = ret[fl]
        base = ret[~fl]
        if len(pred) < 3 or len(base) < 6:
            continue
        out.append(float(pred.mean()) - float(base.mean()))
    return np.asarray(out, dtype=float)


def predicted_month_fraction(flags: dict) -> float:
    """Share of all stock-months that are predicted dividend months."""
    tot = 0
    pred = 0
    for _tk, (_p, ret, fl) in flags.items():
        tot += len(fl)
        pred += int(fl.sum())
    return pred / tot if tot else float("nan")


def share_of_return(flags: dict) -> dict:
    """What fraction of cumulative log return is earned in predicted dividend months."""
    pred_lr = 0.0
    all_lr = 0.0
    pred_m = 0
    all_m = 0
    for _tk, (_p, ret, fl) in flags.items():
        lr = np.log1p(ret)
        pred_lr += float(lr[fl].sum())
        all_lr += float(lr.sum())
        pred_m += int(fl.sum())
        all_m += len(ret)
    return {
        "pred_logret": pred_lr, "all_logret": all_lr,
        "share_return": (pred_lr / all_lr) if all_lr else float("nan"),
        "share_months": (pred_m / all_m) if all_m else float("nan"),
        "pred_months": pred_m, "all_months": all_m,
    }


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def ttest_vs_zero(sample: np.ndarray) -> float:
    """One-sample t of ``sample`` against 0."""
    sample = np.asarray(sample, dtype=float)
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float(sample.mean() / se)


def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of mean(sample) - mean(base) (unequal variance). NaN if either < 2."""
    sample = np.asarray(sample, dtype=float)
    base = np.asarray(base, dtype=float)
    if len(sample) < 2 or len(base) < 2:
        return float("nan")
    m1, m0 = sample.mean(), base.mean()
    v1 = sample.var(ddof=1) / len(sample)
    v0 = base.var(ddof=1) / len(base)
    se = np.sqrt(v1 + v0)
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def placebo_pvalue(flags: dict, n_draws: int = 20_000, seed: int = 516) -> dict:
    """Random-tag placebo null for the per-month premium.

    For each name we keep the *number* of predicted months fixed but randomly relocate which
    months carry the tag (a random calendar of the same density). Each draw recomputes the
    pooled predicted-minus-rest mean; ``p`` = P[shuffled premium >= observed]. The honest answer
    to "could a random set of calendar months of the same density have looked this special?".
    """
    rng = np.random.default_rng(seed)
    rets = []
    counts = []
    for _tk, (_p, ret, fl) in flags.items():
        rets.append(ret)
        counts.append(int(fl.sum()))
    pred, rest = pooled_returns(flags)
    obs = float(pred.mean() - rest.mean())

    means = np.empty(n_draws)
    for d in range(n_draws):
        pred_sum = 0.0
        pred_n = 0
        rest_sum = 0.0
        rest_n = 0
        for r, k in zip(rets, counts):
            n = len(r)
            if k <= 0 or k >= n:
                rest_sum += r.sum()
                rest_n += n
                continue
            sel = rng.choice(n, size=k, replace=False)
            mask = np.zeros(n, dtype=bool)
            mask[sel] = True
            pred_sum += r[mask].sum()
            pred_n += k
            rest_sum += r[~mask].sum()
            rest_n += n - k
        means[d] = (pred_sum / pred_n) - (rest_sum / rest_n)
    p = float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()), "p_value": p, "draws": means}


def summarize_premium(flags: dict, n_draws: int = 20_000, placebo: bool = True) -> dict:
    """Headline per-month premium stats: pooled means, Welch t, per-name one-sample t, placebo."""
    pred, rest = pooled_returns(flags)
    nm = per_name_premium(flags)
    welch = welch_t(pred, rest)
    t_nm = ttest_vs_zero(nm)
    p = (placebo_pvalue(flags, n_draws=n_draws)["p_value"]
         if placebo else float("nan"))
    sr = share_of_return(flags)
    return {
        "n_pred": int(len(pred)), "n_rest": int(len(rest)), "n_names": int(len(nm)),
        "pred_mean_bps": float(pred.mean() * 1e4), "rest_mean_bps": float(rest.mean() * 1e4),
        "premium_bps": float((pred.mean() - rest.mean()) * 1e4),
        "welch_t": welch, "name_premium_bps": float(nm.mean() * 1e4), "t_name": t_nm,
        "p_placebo": p,
        "share_return": sr["share_return"], "share_months": sr["share_months"],
    }


# --------------------------------------------------------------------------- #
# Tradable monthly overlay
# --------------------------------------------------------------------------- #
def overlay_stats(flags: dict, cost_bps: float = 5.0) -> dict:
    """Long-only predicted-dividend-month overlay vs always-invested buy & hold.

    The overlay holds a name equal-weighted ONLY in its predicted-dividend months (the flag is
    known a month ahead from the past cadence, so it is entered at the start of the predicted
    month — a clean one-month execution lag, no look-ahead). Each predicted month is one round
    trip: enter + exit = ``2 * cost_bps`` one-way, charged against that month's return. We report
    the gross and net mean monthly return *while in a predicted month*, annualised, versus the
    basket's buy & hold mean monthly return annualised.
    """
    pred, rest = pooled_returns(flags)
    all_ret = np.concatenate([pred, rest])
    c = cost_bps / 1e4
    per_month_cost = 2.0 * c           # full round trip every predicted month
    gross_monthly = float(pred.mean())
    net_monthly = gross_monthly - per_month_cost
    bh_monthly = float(all_ret.mean())
    return {
        "gross_monthly_bps": gross_monthly * 1e4, "net_monthly_bps": net_monthly * 1e4,
        "bh_monthly_bps": bh_monthly * 1e4,
        "gross_ann_pct": (np.expm1(np.log1p(gross_monthly) * MONTHS)) * 100,
        "net_ann_pct": (np.expm1(np.log1p(max(net_monthly, -0.99)) * MONTHS)) * 100,
        "bh_ann_pct": (np.expm1(np.log1p(bh_monthly) * MONTHS)) * 100,
        "cost_bps": cost_bps, "per_month_cost_bps": per_month_cost * 1e4,
        "n_pred_months": int(len(pred)),
    }


def build_actual_flags(prices, divs) -> dict:
    """Per-name flag using the *contemporaneous* ex-dividend month (the look-ahead 'cheat').

    Tags month ``M`` if the name actually paid a dividend in month ``M`` — using information not
    knowable until the payment happens. Comparing this against the *predicted* (past-only) flag
    is the third-axis test: Hartzmark-Solomon's claim is that the premium is **predictable in
    advance**, so the predicted flag should be at least as strong as the actual one. Imported
    here (not data.py) to keep the data layer focused; uses the same monthly returns.
    """
    import numpy as np

    from . import data as _data
    mret = _data.monthly_returns(prices)
    dv = divs.copy()
    dv["ym"] = dv["date"].dt.to_period("M")
    paid = set(zip(dv["ticker"], dv["ym"]))
    out = {}
    for tk in mret.columns:
        s = mret[tk].dropna()
        if len(s) < 24:
            continue
        per = s.index.to_period("M")
        vals = s.values
        fl = np.array([(tk, pp) in paid for pp in per])
        out[tk] = (per, vals, fl)
    return out


def name_premium_robust(flags: dict) -> dict:
    """The per-name premium t is deterministic on the real tape (no RNG); report it plainly.

    Kept as a named entry point so the verify script + notebooks can show the primary t is not
    a lucky-seed artefact: it is a fixed function of the cached data, with no random component.
    """
    nm = per_name_premium(flags)
    return {"n_names": int(len(nm)), "mean_bps": float(nm.mean() * 1e4),
            "t": ttest_vs_zero(nm)}
