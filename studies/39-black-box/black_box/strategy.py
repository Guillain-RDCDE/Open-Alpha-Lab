"""The book — a small MLP classifier predicting next-day direction, traded two ways.

Two evaluation modes, and the gap between them *is* the study:

  1. :func:`in_sample_predictions` — **the trap.** Fit the net on the WHOLE series and predict the SAME
     series. The learner has seen every label it is now "predicting", so an overparameterised MLP can
     memorise noise and report a gorgeous accuracy / Sharpe. This is what a careless backtest does.
  2. :func:`walk_forward_predictions` — **the honest test.** Expanding-window walk-forward: fit on
     ``[0, split)``, predict the next ``step`` days it has never seen, advance, concatenate the
     out-of-sample predictions. This is the only number a live trader could ever have earned.

We trade the predicted sign (long the predicted-up days, short the predicted-down days, unit gross),
charge ``cost_bps`` per change of position, and report the same :func:`summary` the rest of the desk uses.
The MLP carries a fixed ``random_state`` so every run is reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_features

TRADING_DAYS = 365  # crypto trades every calendar day


def _make_mlp(hidden=(32, 16), max_iter: int = 400, alpha: float = 1e-4, seed: int = 39):
    """A deliberately *capable* (mildly over-parameterised) MLP with a fixed seed — the kind of net whose
    flexibility lets it shine in-sample and overfit out-of-sample. Lazy sklearn import."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter, alpha=alpha,
                        random_state=seed, early_stopping=False)
    return make_pipeline(StandardScaler(), clf)


def in_sample_predictions(close: pd.Series, n_lags: int = 5, hidden=(32, 16),
                          max_iter: int = 400, seed: int = 39) -> pd.Series:
    """Fit on the whole series, predict the whole series — **the in-sample trap**. Returns a +1/−1
    position Series aligned to the (current-day) target the net was trained on."""
    X, y = build_features(close, n_lags=n_lags)
    if len(X) < 30:
        return pd.Series(dtype=float)
    model = _make_mlp(hidden=hidden, max_iter=max_iter, seed=seed)
    model.fit(X.to_numpy(), y.to_numpy())
    pred = model.predict(X.to_numpy())
    pos = pd.Series(np.where(pred > 0, 1.0, -1.0), index=X.index, name="position")
    return pos


def walk_forward_predictions(close: pd.Series, n_lags: int = 5, hidden=(32, 16), max_iter: int = 400,
                             min_train: int = 252, step: int = 63, seed: int = 39) -> pd.Series:
    """Expanding-window walk-forward out-of-sample positions — **the honest test**.

    Fit on the first ``min_train`` rows, predict the next ``step`` rows (unseen), then expand the training
    window by ``step`` and repeat. Concatenates the out-of-sample +1/−1 positions; every prediction is
    made by a net that never saw the day it is trading. Returns an empty Series if too short.
    """
    X, y = build_features(close, n_lags=n_lags)
    if len(X) < min_train + step:
        return pd.Series(dtype=float)
    Xa, ya = X.to_numpy(), y.to_numpy()
    preds, idxs = [], []
    start = min_train
    while start < len(X):
        stop = min(start + step, len(X))
        model = _make_mlp(hidden=hidden, max_iter=max_iter, seed=seed)
        model.fit(Xa[:start], ya[:start])
        p = model.predict(Xa[start:stop])
        preds.append(np.where(p > 0, 1.0, -1.0))
        idxs.append(X.index[start:stop])
        start = stop
    pos = pd.Series(np.concatenate(preds), index=np.concatenate(idxs), name="position")
    return pos


def book_returns(close: pd.Series, positions: pd.Series, cost_bps: float = 10.0) -> pd.Series:
    """Net daily return of trading ``positions`` (+1/−1) on the price series ``close``: the position
    earns that day's simple return, minus ``cost_bps`` per unit of position *change* (a flip costs 2×).

    Positions are aligned to the day whose return they target; the cost is charged on the day the position
    is established. Crypto round-trip costs are higher than equities (~10 bp here is gentle)."""
    close = pd.Series(close).astype(float)
    ret = close.pct_change().reindex(positions.index)
    pos = positions.astype(float)
    gross = pos * ret
    turn = pos.diff().abs().fillna(pos.abs())
    cost = (cost_bps * 1e-4) * turn
    return (gross - cost).rename("strategy").dropna()


def accuracy(close: pd.Series, positions: pd.Series) -> float:
    """Directional hit-rate of the positions vs the realised sign of the day they target."""
    close = pd.Series(close).astype(float)
    ret = close.pct_change().reindex(positions.index)
    realised = np.sign(ret)
    hit = (np.sign(positions) == realised) & (realised != 0)
    valid = realised != 0
    return float(hit.sum() / valid.sum()) if valid.sum() else np.nan


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown, Calmar, skew for a daily return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "calmar", "skew", "n_days")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "cagr": float(cagr), "vol_ann": float(std * np.sqrt(periods_per_year)),
            "max_drawdown": float(dd), "calmar": float(cagr / abs(dd)) if dd < 0 else np.nan,
            "skew": float(r.skew()), "n_days": int(len(r))}
