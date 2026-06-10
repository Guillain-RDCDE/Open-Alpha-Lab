"""Beat-7 worked complement — the **shuffled-label control**, the smoking gun for overfitting.

The in-sample Sharpe looks impressive. Is the net *learning structure*, or *memorising noise*? The
cleanest test: destroy any real relationship between the features and the target by **randomly shuffling
the labels**, then re-fit and re-evaluate **in-sample**. If the in-sample Sharpe survives a shuffled
target — where, by construction, there is *nothing* to learn — then the in-sample number was never
measuring predictive skill; it was measuring the net's capacity to fit whatever it is handed, signal or
noise. A López de Prado / Bailey-Borwein-López de Prado-Zhu staple.

  * :func:`shuffled_label_control` — fit-and-predict in-sample on the true labels and on ``n_shuffles``
    independent label permutations; report the in-sample Sharpe in each case.
  * :func:`insample_vs_oos` — the headline gap: in-sample Sharpe vs walk-forward OOS Sharpe (and OOS
    directional accuracy) on the same series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_features
from .strategy import (
    _make_mlp,
    accuracy,
    book_returns,
    in_sample_predictions,
    summary,
    walk_forward_predictions,
)


def _insample_fit_with_labels(close: pd.Series, y: pd.Series, X: pd.DataFrame,
                              hidden=(32, 16), max_iter: int = 400, seed: int = 39) -> dict:
    """Fit the MLP on (X, y), predict in-sample. Return both the **in-sample training accuracy** (how
    well the net reproduces the labels it was handed) and the in-sample trading Sharpe. ``y`` may be the
    true labels or a shuffled permutation. Training accuracy is the robust memorisation signal; the
    Sharpe is the (noisier) money translation."""
    model = _make_mlp(hidden=hidden, max_iter=max_iter, seed=seed)
    yv = y.to_numpy()
    model.fit(X.to_numpy(), yv)
    pred = model.predict(X.to_numpy())
    train_acc = float((pred == yv).mean())                  # how much of the label set it memorised
    pos = pd.Series(np.where(pred > 0, 1.0, -1.0), index=X.index, name="position")
    sharpe = summary(book_returns(close, pos, cost_bps=0.0))["sharpe"]
    return {"train_accuracy": train_acc, "insample_sharpe": sharpe}


def shuffled_label_control(close: pd.Series, n_shuffles: int = 5, n_lags: int = 5,
                           hidden=(32, 16), max_iter: int = 400, seed: int = 39) -> pd.DataFrame:
    """In-sample fit on the TRUE labels vs ``n_shuffles`` random label permutations.

    The smoking gun for overfitting: if the net's **in-sample training accuracy stays well above the 0.5
    coin-flip baseline even on randomly shuffled labels** — where, by construction, there is nothing to
    learn — then the in-sample number measures the net's capacity to memorise whatever it is handed, not
    predictive skill. Returns a 'true' row and one row per shuffle, each with train accuracy and the
    in-sample trading Sharpe.
    """
    X, y = build_features(close, n_lags=n_lags)
    rng = np.random.default_rng(seed)
    rows = {"true": _insample_fit_with_labels(close, y, X, hidden, max_iter, seed)}
    for i in range(n_shuffles):
        y_shuf = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
        rows[f"shuffle_{i+1}"] = _insample_fit_with_labels(close, y_shuf, X, hidden, max_iter, seed + 1 + i)
    out = pd.DataFrame(rows).T
    out.index.name = "labels"
    return out


def insample_vs_oos(close: pd.Series, n_lags: int = 5, hidden=(32, 16), max_iter: int = 400,
                    min_train: int = 252, step: int = 63, cost_bps: float = 10.0,
                    seed: int = 39) -> pd.DataFrame:
    """The headline gap: in-sample vs walk-forward OOS — Sharpe (gross), net Sharpe and accuracy."""
    pos_is = in_sample_predictions(close, n_lags=n_lags, hidden=hidden, max_iter=max_iter, seed=seed)
    pos_oos = walk_forward_predictions(close, n_lags=n_lags, hidden=hidden, max_iter=max_iter,
                                       min_train=min_train, step=step, seed=seed)
    rows = {
        "in_sample": {
            "sharpe_gross": summary(book_returns(close, pos_is, cost_bps=0.0))["sharpe"],
            "sharpe_net": summary(book_returns(close, pos_is, cost_bps=cost_bps))["sharpe"],
            "accuracy": accuracy(close, pos_is),
        },
        "walk_forward_oos": {
            "sharpe_gross": summary(book_returns(close, pos_oos, cost_bps=0.0))["sharpe"],
            "sharpe_net": summary(book_returns(close, pos_oos, cost_bps=cost_bps))["sharpe"],
            "accuracy": accuracy(close, pos_oos),
        },
    }
    out = pd.DataFrame(rows).T
    out.index.name = "evaluation"
    return out
