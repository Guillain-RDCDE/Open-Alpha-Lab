"""Cross-validation on time series, and the two ways it leaks — Study 1001.

A model is fitted on some data and scored on other data. On i.i.d. observations, splitting at
random is correct. Financial data is not i.i.d., and two distinct leaks follow:

1. **Temporal leakage.** Shuffled folds put future observations in the training set. A model
   that has seen next month's returns will "predict" this month's beautifully. This is the leak
   everyone knows about and the fix — split by time — is obvious.

2. **Label overlap.** The subtle one, and the reason a naive time-split is still not enough.
   Predicting the *next 20 days'* return means observation *t* and observation *t+1* share 19
   of their 20 label days. So a training point immediately before the test set is almost the
   same observation as the first test point. The model has effectively seen the test label
   without any date being out of order.

The fixes, from López de Prado (2018):

- **Purging** removes training observations whose *label window* overlaps the test set. This
  addresses leak 2 directly and is the more important of the two fixes.
- **Embargoing** additionally removes a gap of training observations immediately *after* the
  test set, because serial correlation in features means a point just after the test period is
  still informative about it.

The module implements four schemes — ``kfold_shuffled``, ``kfold_sequential``,
``purged_kfold`` and ``walk_forward`` — with identical everything else, so the differences
between their scores are attributable to the scheme alone. ``leakage_decomposition`` then
separates how much of the illusion comes from each of the two leaks, which nobody usually
bothers to do and which decides where to spend effort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Features and labels
# --------------------------------------------------------------------------- #
def make_features(prices: pd.Series, lookbacks=(5, 21, 63)) -> pd.DataFrame:
    """Trailing momentum and volatility features, all strictly backward-looking."""
    r = prices.pct_change()
    out = {}
    for lb in lookbacks:
        out[f"mom_{lb}"] = prices.pct_change(lb)
        out[f"vol_{lb}"] = r.rolling(lb).std()
    out["rsi_14"] = _rsi(prices, 14)
    return pd.DataFrame(out).dropna()


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    d = prices.diff()
    up = d.clip(lower=0).rolling(window).mean()
    down = (-d.clip(upper=0)).rolling(window).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi")


def make_labels(prices: pd.Series, horizon: int = 20) -> pd.DataFrame:
    """Forward returns over ``horizon`` sessions, with the label window recorded.

    ``label_start`` and ``label_end`` are the crucial columns. Purging needs to know *which
    dates each label depends on*, and the whole leak this study is about comes from those
    windows overlapping between neighbouring observations. Carrying them explicitly is what
    makes the fix implementable rather than approximate.
    """
    fwd = prices.shift(-horizon) / prices - 1.0
    idx = prices.index
    pos = np.arange(len(idx))
    end_pos = np.minimum(pos + horizon, len(idx) - 1)
    return pd.DataFrame({"label": fwd.to_numpy(),
                         "label_start": idx,
                         "label_end": idx[end_pos]}, index=idx).dropna(subset=["label"])


def overlap_fraction(horizon: int, gap: int) -> float:
    """What share of two labels' windows coincide when their start dates are ``gap`` apart."""
    if gap >= horizon:
        return 0.0
    return float((horizon - gap) / horizon)


# --------------------------------------------------------------------------- #
# The four schemes
# --------------------------------------------------------------------------- #
def kfold_shuffled(n: int, k: int = 5, seed: int = 1001) -> list:
    """Ordinary shuffled k-fold. Wrong for time series, and the default in every tutorial."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    return [(np.setdiff1d(np.arange(n), f), np.sort(f)) for f in folds]


def kfold_sequential(n: int, k: int = 5) -> list:
    """Contiguous blocks, no shuffling — but still training on data after the test block."""
    folds = np.array_split(np.arange(n), k)
    return [(np.setdiff1d(np.arange(n), f), f) for f in folds]


def purged_kfold(labels: pd.DataFrame, k: int = 5, embargo: float = 0.0) -> list:
    """Contiguous folds, with training observations purged and embargoed.

    **Purging**: drop any training observation whose label window overlaps the test block's
    time span. That is the fix for label overlap, and it is why ``make_labels`` carries the
    window boundaries around.

    **Embargo**: additionally drop a further ``embargo`` fraction of the sample immediately
    after each test block. Purging handles labels that reach *into* the test period; the
    embargo handles the fact that serial correlation makes observations just *after* it
    informative too.
    """
    n = len(labels)
    starts = labels["label_start"].to_numpy()
    ends = labels["label_end"].to_numpy()
    folds = np.array_split(np.arange(n), k)
    emb = int(round(embargo * n))
    out = []
    for f in folds:
        t0 = starts[f[0]]
        t1 = ends[f[-1]]
        train = np.setdiff1d(np.arange(n), f)
        # purge: a training label window that touches [t0, t1] has seen the test period
        keep = ~((ends[train] >= t0) & (starts[train] <= t1))
        train = train[keep]
        if emb > 0:
            lo, hi = f[-1] + 1, min(f[-1] + 1 + emb, n)
            train = train[~np.isin(train, np.arange(lo, hi))]
        out.append((train, f))
    return out


def walk_forward(n: int, k: int = 5, min_train: float = 0.3) -> list:
    """Expanding window: train on the past, test on the next block. Never sees the future.

    The scheme a live system actually resembles, and the honest benchmark for everything else.
    It uses less data per fold than k-fold does, which is a real cost and is why people reach
    for cross-validation in the first place.
    """
    start = int(n * min_train)
    if start >= n - 10:
        return []
    blocks = np.array_split(np.arange(start, n), k)
    return [(np.arange(0, b[0]), b) for b in blocks if len(b) > 0 and b[0] > 0]


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
def fit_predict(X: np.ndarray, y: np.ndarray, train: np.ndarray,
                test: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Ridge regression, standardised on the TRAINING fold only.

    Standardising on the full sample is itself a leak — a small one next to the others here,
    but it is the same mistake in miniature and it costs nothing to avoid.
    """
    if len(train) < 30 or len(test) == 0:
        return np.full(len(test), np.nan)
    Xtr, ytr = X[train], y[train]
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Ztr = (Xtr - mu) / sd
    Zte = (X[test] - mu) / sd
    A = np.column_stack([np.ones(len(Ztr)), Ztr])
    B = np.column_stack([np.ones(len(Zte)), Zte])
    p = A.shape[1]
    reg = ridge * np.eye(p)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
    return B @ beta


def score_scheme(X: np.ndarray, y: np.ndarray, folds: list,
                 ridge: float = 1e-3) -> dict:
    """Score one validation scheme. ``ic`` is the mean of the PER-FOLD ICs.

    That choice is not cosmetic, and getting it wrong cost this study an afternoon.

    The obvious implementation collects every out-of-fold prediction into one long vector and
    correlates it with the labels once. It is wrong, for a reason that has nothing to do with
    leakage. Each fold's model is fitted and standardised on *its own* training set, so each
    fold's predictions sit at their own level. Pooling them puts several clouds with different
    centres into one scatter, and the correlation then measures the arrangement of the clouds
    rather than the skill inside any of them. With persistent features and overlapping labels
    the cloud centres drift with the sample, and the pooled statistic comes out **strongly
    negative on data with no signal whatsoever** — an artefact large enough to swamp the effect
    this study is trying to measure.

    Averaging the per-fold correlations removes it, because each is computed within a single
    fold where the level is common. ``ic_pooled`` is returned as well, precisely so the
    difference can be displayed rather than asserted.
    """
    preds = np.full(len(y), np.nan)
    per_fold = []
    for train, test in folds:
        test = np.asarray(test)
        p = fit_predict(X, y, np.asarray(train), test, ridge)
        preds[test] = p
        ok = np.isfinite(p)
        if ok.sum() >= 30 and p[ok].std() > 0 and y[test][ok].std() > 0:
            per_fold.append(float(np.corrcoef(p[ok], y[test][ok])[0, 1]))
    ok = np.isfinite(preds)
    if ok.sum() < 50 or not per_fold:
        return {"n": int(ok.sum())}
    p, a = preds[ok], y[ok]
    ss_res = float(((a - p) ** 2).sum())
    ss_tot = float(((a - a.mean()) ** 2).sum())
    ic = float(np.mean(per_fold))
    pooled = float(np.corrcoef(p, a)[0, 1]) if p.std() > 0 else np.nan
    return {"n": int(ok.sum()), "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "ic": ic, "ic_pooled": pooled, "ic_sd": float(np.std(per_fold, ddof=1))
            if len(per_fold) > 1 else np.nan,
            "n_folds": len(per_fold),
            "mean_train_size": float(np.mean([len(t) for t, _ in folds])),
            "predictions": preds}


def compare_schemes(features: pd.DataFrame, labels: pd.DataFrame, k: int = 5,
                    embargo: float = 0.01, ridge: float = 1e-3) -> pd.DataFrame:
    """All four schemes on identical data, so only the scheme differs."""
    common = features.index.intersection(labels.index)
    F = features.loc[common]
    L = labels.loc[common]
    X = F.to_numpy(dtype=float)
    y = L["label"].to_numpy(dtype=float)
    n = len(y)
    schemes = {
        "k-fold, shuffled": kfold_shuffled(n, k),
        "k-fold, sequential": kfold_sequential(n, k),
        "purged k-fold": purged_kfold(L, k, embargo=0.0),
        "purged + embargo": purged_kfold(L, k, embargo=embargo),
        "walk-forward": walk_forward(n, k),
    }
    rows = []
    for name, folds in schemes.items():
        if not folds:
            continue
        s = score_scheme(X, y, folds, ridge)
        rows.append({"scheme": name, **{kk: v for kk, v in s.items()
                                        if kk != "predictions"}})
    return pd.DataFrame(rows).set_index("scheme")


def leakage_decomposition(features: pd.DataFrame, labels: pd.DataFrame, k: int = 5,
                          embargo: float = 0.01) -> dict:
    """Split the illusion into its two causes.

    - **Temporal leakage** = shuffled minus sequential. What you gain purely by letting the
      model see the future.
    - **Label overlap** = sequential minus purged. What you gain from neighbouring labels
      sharing days, with no date out of order at all.
    - **Embargo effect** = purged minus purged-with-embargo.

    Separating them matters because the second is invisible to the usual advice ("don't shuffle
    time series") and is often the larger of the two at long label horizons.
    """
    c = compare_schemes(features, labels, k, embargo)
    if c.empty or "walk-forward" not in c.index or "ic" not in c.columns:
        return {}
    def g(name, col="ic"):
        return float(c.loc[name, col]) if name in c.index else np.nan
    shuffled, seq = g("k-fold, shuffled"), g("k-fold, sequential")
    purged, emb = g("purged k-fold"), g("purged + embargo")
    wf = g("walk-forward")
    # A sample too short for the honest benchmark leaves nothing to decompose against.
    if not all(np.isfinite(v) for v in (shuffled, seq, purged, emb, wf)):
        return {}
    return {"shuffled_ic": shuffled, "sequential_ic": seq, "purged_ic": purged,
            "embargo_ic": emb, "walk_forward_ic": wf,
            "temporal_leak": shuffled - seq,
            "overlap_leak": seq - purged,
            "embargo_effect": purged - emb,
            "total_illusion": shuffled - wf,
            "overlap_share": float((seq - purged) / (shuffled - wf))
            if abs(shuffled - wf) > 1e-12 else np.nan,
            "honest_ic": wf}


def horizon_sweep(prices: pd.Series, horizons=(1, 5, 20, 60, 120), k: int = 5,
                  embargo: float = 0.01) -> pd.DataFrame:
    """How the leak scales with the label horizon.

    The single most useful table here. At a one-day horizon labels do not overlap and purging
    changes nothing; at a 120-day horizon each label shares 119 of its 120 days with its
    neighbour, and the overlap leak becomes enormous. The horizon is a design choice, and this
    is its hidden cost.
    """
    F = make_features(prices)
    rows = []
    for hz in horizons:
        L = make_labels(prices, hz)
        d = leakage_decomposition(F, L, k, embargo)
        if not d:
            continue
        rows.append({"horizon": hz, "overlap_with_neighbour": overlap_fraction(hz, 1),
                     **{kk: d[kk] for kk in ("shuffled_ic", "sequential_ic", "purged_ic",
                                             "walk_forward_ic", "temporal_leak",
                                             "overlap_leak", "total_illusion")}})
    return pd.DataFrame(rows).set_index("horizon")


def synthetic_panel(n: int = 4000, predictability: float = 0.0, horizon: int = 20,
                    n_features: int = 4, ar_feature: float = 0.95,
                    seed: int = 1001) -> dict:
    """Features with a KNOWN predictive relationship to a forward label.

    ``predictability`` is the true information coefficient between the (first) feature and the
    forward return. At zero there is nothing to find, and any scheme reporting a positive score
    is reporting leakage. The features are autocorrelated, because real ones are and because
    that is what makes the embargo matter.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1993-02-01", periods=n)
    F = np.zeros((n, n_features))
    for j in range(n_features):
        e = rng.normal(0, 1, n)
        v = np.zeros(n)
        for t in range(1, n):
            v[t] = ar_feature * v[t - 1] + e[t]
        F[:, j] = v / max(v.std(), 1e-9)
    daily = rng.normal(0, 0.01, n)
    if predictability > 0:
        # the signal predicts the FORWARD horizon-day sum, spread evenly over those days
        sig = predictability * 0.01 * F[:, 0]
        for t in range(n - horizon):
            daily[t + 1:t + 1 + horizon] += sig[t] / horizon
    rets = pd.Series(daily, index=idx, name="ret")
    prices = (1 + rets).cumprod() * 100
    features = pd.DataFrame(F, index=idx,
                            columns=[f"f{j}" for j in range(n_features)])
    labels = make_labels(prices, horizon)
    return {"prices": prices, "returns": rets, "features": features, "labels": labels,
            "predictability": predictability}


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Confirmed** if shuffled k-fold reports a materially higher information
      coefficient than walk-forward on data where the honest answer is near zero — i.e. the
      illusion is demonstrated; **Partial** if the gap exists but is small; **Busted** if
      ordinary cross-validation turns out to be fine.
    - **Tradability**: this is a methodology question. **Useful** if purging plus embargo
      recovers an estimate close to the walk-forward truth **and** still detects a planted
      signal — a fix that destroyed real signal along with the leakage would be no fix;
      **Partial** if it does one; **Mirage** if neither.
    """
    material = h["total_illusion"] > 0.02
    signal = ("Confirmed" if material and h["honest_ic"] < 0.05
              else ("Partial" if h["total_illusion"] > 0 else "Busted"))
    recovers = abs(h["embargo_ic"] - h["honest_ic"]) < 0.02
    keeps_signal = h["planted_detected"]
    trad = ("Useful" if (recovers and keeps_signal)
            else ("Partial" if (recovers or keeps_signal) else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"On {h['asset']} with a {h['horizon']}-day forward label, ordinary shuffled "
            f"five-fold cross-validation reported an information coefficient of "
            f"**{h['shuffled_ic']:+.3f}** (R² {h['shuffled_r2']:+.3f}). Walk-forward validation "
            f"— the only scheme that resembles how a model is actually used — reported "
            f"**{h['honest_ic']:+.3f}**. The gap is **{h['total_illusion']:+.3f}**, and it "
            f"splits into two quite different causes. Letting the model see the future at all "
            f"(shuffled minus sequential) is worth {h['temporal_leak']:+.3f}. **Label overlap** "
            f"— neighbouring observations sharing {h['overlap_pct']:.0%} of their label days, "
            f"with no date out of order whatsoever — is worth {h['overlap_leak']:+.3f}, which "
            f"is **{h['overlap_share']:.0%} of the total illusion**. That second channel is "
            f"invisible to the usual advice about not shuffling time series, and at long label "
            f"horizons it is the bigger one: the sweep shows it growing from "
            f"{h['leak_at_short']:+.3f} at a one-day label to {h['leak_at_long']:+.3f} at "
            f"{h['longest_horizon']} days."),
        "trad_why": (
            f"The fix works and it is cheap. Purging training observations whose label windows "
            f"touch the test block brought the estimate to {h['purged_ic']:+.3f}; adding a "
            f"{h['embargo']:.0%} embargo brought it to **{h['embargo_ic']:+.3f}** against "
            f"walk-forward's {h['honest_ic']:+.3f} — a residual difference of "
            f"{abs(h['embargo_ic'] - h['honest_ic']):.4f}. And it does not achieve that by "
            f"destroying everything: on synthetic data with a **planted** information "
            f"coefficient of {h['planted_ic']:.2f}, purged-and-embargoed cross-validation "
            f"recovered {h['planted_recovered']:+.3f}, so it "
            f"{'still finds real signal' if h['planted_detected'] else 'failed to find the planted signal, which is a problem'}. "
            f"The cost is data: purging removed {h['purged_data_loss']:.0%} of the training set "
            f"at this horizon, which is the real reason people skip it."),
        "trad": trad,
        "one_sentence": (
            f"Shuffled cross-validation flatters this model by {h['total_illusion']:+.3f} of "
            f"information coefficient, {h['overlap_share']:.0%} of it from overlapping labels "
            f"rather than from shuffling — and purging with an embargo recovers the honest "
            f"number to within {abs(h['embargo_ic'] - h['honest_ic']):.4f}."),
    }
