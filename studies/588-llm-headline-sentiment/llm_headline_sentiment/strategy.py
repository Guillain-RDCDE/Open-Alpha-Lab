"""The engine and its honest controls — Study 588 (LLM-Headline-Sentiment).

The claim, at full strength: an LLM reads the day's news headlines, emits a mood score, and
that score forecasts *tomorrow's* market return. The tradable expression is a next-day sign
timer — go long when today's mood is positive, flat/short when negative — held one day.

This module measures that, honestly, on the deterministic synthetic tape (``data.py``), and
demonstrates the two ways this kind of study fools people:

1. **The predictive regression.** OLS of the *forward* (next-day) return on today's sentiment,
   with a **Newey-West HAC** *t*-stat (persistent sentiment => autocorrelated regressors, so a
   naive OLS *t* is over-confident). The slope's sign and HAC *t* are the headline.

2. **The label-shuffle placebo.** Break the sentiment->return link by permuting the return
   labels and re-fit many times; the p-value is the tail probability of the observed |t|. A real
   link sits in the tail; noise does not.

3. **The multiple-testing trap (the money shot).** A researcher tries K sentiment recipes and
   reports the best. ``best_of_k_tstat`` returns the max |t| over the bank; ``maxt_pvalue`` is
   the *permutation-max* (Westfall-Young / White Reality Check) null — the distribution of the
   *best* |t| under no signal. Even when *every* feature is noise, the naive best-of-K |t| looks
   "significant"; the max-t null shows it isn't. This is the overfitting demo the study exists
   to make.

4. **The look-ahead trap.** Contemporaneous scoring (score today's headlines *from* today's
   return, i.e. regress same-day return on same-day sentiment that was itself built with
   hindsight) inflates the apparent edge. We contrast the honest lagged design (t -> t+1) with a
   contemporaneous one to show how easy it is to leak.

5. **Costs.** A one-day sign timer flips often; we charge a one-way cost x NAV per flip (both
   sides of a round trip counted) and, when short, a borrow. Net Sharpe is labeled next to
   gross.

6. **A seed-robust synthetic positive control.** The house rule: average the HAC *t* over
   >= 20 seeds so no lucky seed manufactures significance. The engine must read ~0 at ``beta = 0``
   and clear the bar as ``beta`` grows.

Execution convention: the signal on day *t* (today's LLM mood) earns the return of *t+1* — one
lag, baked into ``data.synthetic_series`` as the ``forward_ret`` column (``forward_ret_t`` is the
t->t+1 return). No second silent shift. The contemporaneous variant in ``look_ahead_contrast``
is labeled a *leak*, never the headline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Predictive regression with a Newey-West HAC t-stat
# ---------------------------------------------------------------------------
def _hac_se(x: np.ndarray, resid: np.ndarray, lags: int) -> float:
    """Newey-West HAC standard error of the slope in a simple regression y = a + b x.

    Returns the HAC SE of ``b`` given the (demeaned handled via design) regressor ``x`` and the
    fitted residuals ``resid``.
    """
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    u = resid.reshape(-1, 1)
    # Meat: S = sum_t w_l * (x_t u_t)(x_{t-l} u_{t-l})' summed with Bartlett weights.
    Xu = X * u
    S = Xu.T @ Xu
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return float(np.sqrt(cov[1, 1]))


def predictive_regression(sentiment: pd.Series, forward_ret: pd.Series,
                          hac_lags: int = 5) -> dict:
    """OLS of forward return on sentiment with a Newey-West HAC *t* on the slope.

    Returns ``{slope, se, t (HAC), t_ols, r2, n}``. A positive slope with HAC *t* >= 2 is the
    claim: today's mood forecasts tomorrow's return.
    """
    s = pd.concat([sentiment, forward_ret], axis=1).dropna()
    if len(s) < 20:
        return {"slope": float("nan"), "se": float("nan"), "t": float("nan"),
                "t_ols": float("nan"), "r2": float("nan"), "n": int(len(s))}
    x = s.iloc[:, 0].to_numpy(dtype=float)
    y = s.iloc[:, 1].to_numpy(dtype=float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    # OLS SE (for the contrast)
    dof = max(n - 2, 1)
    sigma2 = float(resid @ resid) / dof
    ols_se = float(np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1]))
    hac_se = _hac_se(x, resid, hac_lags)
    b = float(coef[1])
    return {
        "slope": b,
        "se": hac_se,
        "t": b / hac_se if hac_se > 0 else float("nan"),
        "t_ols": b / ols_se if ols_se > 0 else float("nan"),
        "r2": r2,
        "n": n,
    }


def _abs_t(sentiment: np.ndarray, y: np.ndarray, hac_lags: int) -> float:
    n = len(sentiment)
    X = np.column_stack([np.ones(n), sentiment])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    se = _hac_se(sentiment, resid, hac_lags)
    return abs(coef[1] / se) if se > 0 else 0.0


# ---------------------------------------------------------------------------
# Label-shuffle placebo (single feature)
# ---------------------------------------------------------------------------
def placebo_pvalue(sentiment: pd.Series, forward_ret: pd.Series,
                   n_perm: int = 2000, hac_lags: int = 5, seed: int = 588) -> float:
    """Label-shuffle placebo p-value for the predictive slope's |HAC t|.

    Permute the forward-return labels ``n_perm`` times; the p-value is the fraction of shuffles
    whose |t| >= the observed |t|. A real link sits in the tail; noise does not.
    """
    s = pd.concat([sentiment, forward_ret], axis=1).dropna()
    if len(s) < 20:
        return float("nan")
    x = s.iloc[:, 0].to_numpy(dtype=float)
    y = s.iloc[:, 1].to_numpy(dtype=float)
    obs = _abs_t(x, y, hac_lags)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        yp = y[rng.permutation(len(y))]
        if _abs_t(x, yp, hac_lags) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# The multiple-testing trap — best-of-K and the permutation-max null
# ---------------------------------------------------------------------------
def all_feature_tstats(features: pd.DataFrame, forward_ret: pd.Series,
                       hac_lags: int = 5) -> np.ndarray:
    """HAC |t| of every feature in the bank against the same forward-return series."""
    y = forward_ret.to_numpy(dtype=float)
    out = np.empty(features.shape[1])
    for j, col in enumerate(features.columns):
        out[j] = _abs_t(features[col].to_numpy(dtype=float), y, hac_lags)
    return out


def best_of_k(features: pd.DataFrame, forward_ret: pd.Series, hac_lags: int = 5) -> dict:
    """The naive 'report the winner' statistic: max |HAC t| over the feature bank.

    Returns ``{best_t, best_feature, k}``. This is what a data-miner would headline — and it is
    biased upward by construction (max of K noisy statistics).
    """
    ts = all_feature_tstats(features, forward_ret, hac_lags)
    j = int(np.argmax(ts))
    return {"best_t": float(ts[j]), "best_feature": str(features.columns[j]),
            "k": int(features.shape[1])}


def maxt_pvalue(features: pd.DataFrame, forward_ret: pd.Series,
                n_perm: int = 1000, hac_lags: int = 5, seed: int = 588) -> dict:
    """Westfall-Young / White-Reality-Check permutation-max p-value for best-of-K.

    Under the joint null (no feature predicts), permute the return labels and record the
    *maximum* |t| across all K features each time. The corrected p-value for the observed
    best-of-K is the fraction of permutations whose max-|t| >= the observed best-|t|. This
    controls the family-wise error the naive best-of-K ignores.

    Returns ``{best_t, naive_p, maxt_p, k}`` where ``naive_p`` is the *single-test* p-value the
    miner would quote for the winner (from the same permutation, per-feature) and ``maxt_p`` is
    the honest family-wise one.
    """
    y = forward_ret.to_numpy(dtype=float)
    feat_arrs = [features[c].to_numpy(dtype=float) for c in features.columns]
    k = len(feat_arrs)
    obs_ts = np.array([_abs_t(x, y, hac_lags) for x in feat_arrs])
    obs_best = float(obs_ts.max())
    obs_best_j = int(np.argmax(obs_ts))
    rng = np.random.default_rng(seed)
    ge_max = 0
    ge_single = 0
    for _ in range(n_perm):
        yp = y[rng.permutation(len(y))]
        perm_ts = np.array([_abs_t(x, yp, hac_lags) for x in feat_arrs])
        if perm_ts.max() >= obs_best - 1e-12:
            ge_max += 1
        # single-test null for the *winning* feature only
        if perm_ts[obs_best_j] >= obs_best - 1e-12:
            ge_single += 1
    return {
        "best_t": obs_best,
        "naive_p": (ge_single + 1) / (n_perm + 1),
        "maxt_p": (ge_max + 1) / (n_perm + 1),
        "k": k,
    }


# ---------------------------------------------------------------------------
# The look-ahead trap — lagged (honest) vs contemporaneous (leak)
# ---------------------------------------------------------------------------
def look_ahead_contrast(df: pd.DataFrame, leak: float = 0.25, hac_lags: int = 5,
                        seed: int = 588) -> dict:
    """Honest lagged design vs a contemporaneous *leak*, on the same tape.

    - **Lagged (honest):** today's sentiment vs *tomorrow's* return — the tradable, no-look-ahead
      estimate (the ``forward_ret`` column is already t->t+1). This is the headline.
    - **Contemporaneous (leak):** the classic mistake. A mood score labelled *after the fact*
      (or a vendor feed timestamped to the session it describes) is contaminated by the very
      return it is scored against. We simulate that by blending a fraction ``leak`` of the
      same-day return *into* the sentiment, then regressing the same-day return on this
      hindsight-tainted score. It looks far stronger than any trader could have exploited,
      because it is partly reading the answer.

    The same-day return realised *into* day *t* is ``forward_ret`` shifted one day. Returns
    ``{lagged_t, contemporaneous_t}``; the gap between them *is* the look-ahead warning — a
    contemporaneous |t| well above the honest lagged |t| is the tell.
    """
    fr = df["forward_ret"]
    sent = df["sentiment"]
    lagged = predictive_regression(sent, fr, hac_lags)["t"]

    # Same-day return (realised into day t) = the previous row's t->t+1 return.
    same_day = fr.shift(1)
    joined = pd.concat([sent, same_day], axis=1).dropna()
    joined.columns = ["sent", "same_day"]
    # A hindsight-tainted score: standardize the same-day return and blend it into sentiment.
    sd = joined["same_day"]
    sd_z = (sd - sd.mean()) / (sd.std(ddof=1) + 1e-12)
    tainted = (1.0 - leak) * joined["sent"] + leak * sd_z * joined["sent"].std(ddof=1)
    contemp = predictive_regression(tainted, joined["same_day"], hac_lags)["t"]
    return {"lagged_t": float(lagged), "contemporaneous_t": float(contemp)}


# ---------------------------------------------------------------------------
# The tradable next-day sign timer — gross & net Sharpe
# ---------------------------------------------------------------------------
def sign_timer(df: pd.DataFrame, cost_bps: float = 1.0, borrow_ann_bps: float = 50.0,
               long_short: bool = True) -> dict:
    """A one-day sign timer: position = sign(sentiment_t), earns forward_ret_t.

    - ``long_short=True``: +1 / -1 book (short pays borrow when negative).
    - ``long_short=False``: +1 / 0 long-only book.

    Charges a one-way ``cost_bps`` x NAV on every change of position (a round trip = two flips),
    and a daily-pro-rated annual borrow on the short side. Returns gross and net annualised
    Sharpe plus mean net return, with turnover.
    """
    s = df.dropna()
    if len(s) < 20:
        return {"sharpe_gross": float("nan"), "sharpe_net": float("nan"),
                "mean_net": float("nan"), "turnover": float("nan"), "n": int(len(s))}
    pos = np.sign(s["sentiment"].to_numpy(dtype=float))
    if not long_short:
        pos = np.clip(pos, 0, 1)
    r = s["forward_ret"].to_numpy(dtype=float)
    gross = pos * r
    # costs: |Δpos| flips * one-way cost; count entry into the book too
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cost = dpos * cost_bps * 1e-4
    borrow_daily = borrow_ann_bps * 1e-4 / TRADING_DAYS
    borrow = np.where(pos < 0, borrow_daily, 0.0)
    net = gross - cost - borrow
    turnover = float(dpos.mean())

    def _sharpe(v):
        sd = v.std(ddof=1)
        return float(v.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")

    return {
        "sharpe_gross": _sharpe(gross),
        "sharpe_net": _sharpe(net),
        "mean_net": float(net.mean() * TRADING_DAYS),
        "turnover": turnover,
        "n": int(len(s)),
    }


# ---------------------------------------------------------------------------
# Seed-robust synthetic positive control (house rule)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, beta: float, n_seeds: int = 25, base_seed: int = 588,
                     hac_lags: int = 5) -> float:
    """Average the predictive HAC *t* over ``n_seeds`` synthetic worlds for a planted ``beta``.

    The house rule: any synthetic-dependent claim averages a HAC statistic over >= 20 seeds so no
    single lucky RNG seed can manufacture significance. Reads ~0 at ``beta = 0`` and rises with
    ``beta``.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        df, _ = data_mod.synthetic_series(beta=beta, seed=s)
        ts.append(predictive_regression(df["sentiment"], df["forward_ret"], hac_lags)["t"])
    return float(np.nanmean(ts))


def synthetic_maxt_falsepos(data_mod, n_features: int = 40, n_seeds: int = 25,
                            base_seed: int = 588, alpha: float = 0.05,
                            n_perm: int = 400, hac_lags: int = 5) -> dict:
    """Under the joint null (n_true=0), how often does naive best-of-K vs max-t 'find' a signal?

    For each of ``n_seeds`` pure-noise feature banks, record whether the naive winner's
    single-test p < ``alpha`` (the miner's mistake) and whether the family-wise max-t p < ``alpha``
    (the honest test). Returns the false-positive rates of each — the naive one should blow past
    ``alpha``, the corrected one should sit near it.
    """
    naive_hits = 0
    maxt_hits = 0
    for s in range(base_seed, base_seed + n_seeds):
        feats, fr, _ = data_mod.synthetic_feature_bank(n_features=n_features, n_true=0, seed=s)
        res = maxt_pvalue(feats, fr, n_perm=n_perm, hac_lags=hac_lags, seed=s)
        naive_hits += int(res["naive_p"] < alpha)
        maxt_hits += int(res["maxt_p"] < alpha)
    return {
        "naive_fpr": naive_hits / n_seeds,
        "maxt_fpr": maxt_hits / n_seeds,
        "n_seeds": n_seeds,
        "n_features": n_features,
    }
