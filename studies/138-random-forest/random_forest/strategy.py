"""Strategy and honest controls — Study 138 (Random-Forest).

The pitch: train a Random Forest on standard technical features (lagged returns, RSI,
realised vol, momentum, volume ratio) to predict next-day direction/return of SPY and
other large-cap equities.  We evaluate it two honest ways:

1. **Walk-forward (the honest test).**  A rolling training window (``train_window`` days)
   fits the forest; it predicts the *next* ``step`` days it has never seen; the window
   then rolls forward by ``step``.  This is the only equity curve a live trader could
   have earned: every prediction is made on unseen data.

2. **Shuffled-label control (the falsification).**  Identical architecture, identical
   features, identical walk-forward splits — but with the *target labels randomly
   permuted* within each training window before fitting.  If the real model scores
   materially better than the shuffle, the features carry real out-of-sample information.
   If they score the same, any apparent in-sample "alpha" is pure overfitting.

No look-ahead: all features use data up to close of day *t*; the position enters at the
open of day *t+1* (enforced by the ``shift(1)`` in ``build_features`` and the walk-forward
loop never training on indices it also predicts).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(
    bars: pd.DataFrame,
    lags: int = 5,
    rsi_n: int = 14,
    vol_n: int = 21,
    mom_n: int = 20,
    vol_ratio_n: int = 20,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a feature matrix X and binary next-day direction target y.

    All features use data available at the **close of day t** — no look-ahead.
    The target ``y[t] = 1`` if ``close[t+1] > close[t]`` else ``0`` — direction of
    the *next* bar, which is the quantity the forest is asked to predict.

    Features:
    - ``ret_lag_{k}`` for k in 1..lags: log returns at t-k (lagged; t-1 is yesterday's return)
    - ``rsi``: Wilder RSI(rsi_n) at t
    - ``rvol``: realised log-return volatility over the last vol_n days
    - ``mom``: cumulative log return over the last mom_n days (momentum)
    - ``vol_ratio``: today's volume / vol_ratio_n-day average volume (turnover proxy)

    Returns ``(X, y)`` aligned on the date index, with NaN rows dropped.
    """
    close = bars["close"]
    volume = bars["volume"]

    log_ret = np.log(close / close.shift(1))

    # Lagged returns (ret_lag_1 = yesterday's return; no look-ahead)
    feature_dict: dict[str, pd.Series] = {}
    for k in range(1, lags + 1):
        feature_dict[f"ret_lag_{k}"] = log_ret.shift(k)

    # RSI (Wilder's smoothed)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=rsi_n - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=rsi_n - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    feature_dict["rsi"] = (100 - 100 / (1 + rs)) / 100.0  # scaled to [0, 1]

    # Realised volatility
    feature_dict["rvol"] = log_ret.rolling(vol_n).std(ddof=1)

    # Medium-term momentum
    feature_dict["mom"] = log_ret.rolling(mom_n).sum()

    # Volume ratio (today / n-day average)
    feature_dict["vol_ratio"] = volume / volume.rolling(vol_ratio_n).mean()

    X = pd.DataFrame(feature_dict, index=bars.index)

    # Target: direction of NEXT day's return — shift(-1) so t maps to t+1 outcome
    y = (log_ret.shift(-1) > 0).astype(int)
    y.name = "up_tomorrow"

    # Drop any row where X or y is NaN (beginning of series or last row)
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]


# ---------------------------------------------------------------------------
# Random Forest model
# ---------------------------------------------------------------------------
def _make_rf(n_estimators: int = 200, max_depth: int | None = 6, seed: int = 138):
    """A Random Forest classifier with fixed seed for reproducibility.

    Deliberately regularised (``max_depth=6``, ``min_samples_leaf=10``) to reduce
    variance and avoid trivially memorising the training set — but not so constrained
    that it can't learn a real pattern if one exists.  The pipeline does NOT include a
    feature scaler (trees are invariant to monotone transforms).
    """
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------
def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    train_window: int = 504,
    step: int = 63,
    n_estimators: int = 200,
    max_depth: int | None = 6,
    seed: int = 138,
    shuffle_labels: bool = False,
    shuffle_seed: int = 42,
) -> pd.DataFrame:
    """Rolling walk-forward predictions — the only evaluation a live trader could earn.

    For each fold:
    - Training set: the most recent ``train_window`` rows ending at the fold boundary.
    - Test set: the next ``step`` rows (never seen during training).

    If ``shuffle_labels=True``, the *training labels* are randomly permuted before
    fitting (within each fold, with ``shuffle_seed``), while the test set is left
    intact.  This is the shuffled-label control: if the shuffled model matches the
    real model, the features carry no genuine information.

    Returns a DataFrame with columns ``date, y_true, y_pred, proba`` aligned on the
    test indices.
    """
    Xa = X.to_numpy(dtype=float)
    ya = y.to_numpy(dtype=int)
    n = len(X)

    rng = np.random.default_rng(shuffle_seed)

    rows = []
    start = train_window
    while start < n:
        stop = min(start + step, n)
        train_idx = slice(start - train_window, start)
        test_idx = slice(start, stop)

        X_train, y_train = Xa[train_idx], ya[train_idx].copy()
        X_test = Xa[test_idx]

        if shuffle_labels:
            rng.shuffle(y_train)

        model = _make_rf(n_estimators=n_estimators, max_depth=max_depth, seed=seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]  # P(up)

        for i, date in enumerate(X.index[start:stop]):
            rows.append({
                "date": date,
                "y_true": int(ya[start + i]),
                "y_pred": int(preds[i]),
                "proba": float(probas[i]),
            })
        start = stop

    return pd.DataFrame(rows).set_index("date")


def feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    train_window: int = 504,
    n_estimators: int = 200,
    max_depth: int | None = 6,
    seed: int = 138,
) -> pd.Series:
    """Mean impurity-decrease feature importance from a single in-sample fit.

    Fit on the first ``train_window`` rows (the initial training period).  Returns
    a Series of importances sorted descending.  This is an **in-sample** diagnostic —
    it shows what the tree *used*, not necessarily what is predictive out of sample.
    """
    Xa = X.to_numpy(dtype=float)
    ya = y.to_numpy(dtype=int)
    n_train = min(train_window, len(X))
    model = _make_rf(n_estimators=n_estimators, max_depth=max_depth, seed=seed)
    model.fit(Xa[:n_train], ya[:n_train])
    imp = pd.Series(model.feature_importances_, index=X.columns, name="importance")
    return imp.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Trading the predictions
# ---------------------------------------------------------------------------
def trade_predictions(
    bars: pd.DataFrame,
    preds: pd.DataFrame,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Translate walk-forward predictions into a daily P&L ledger.

    Position: long (+1) when ``y_pred == 1`` (model predicts up), flat (0) otherwise.
    We do not short on a predicted-down day to keep the comparison simple (long-only
    vs buy-and-hold).  Entry at open of *t+1*, exit at close of *t+1*.

    Returns a DataFrame with ``ret_gross``, ``ret_net``, ``position``, ``bh_ret`` aligned
    on ``preds.index``.
    """
    close = bars["close"].reindex(preds.index)
    open_ = bars["open"].reindex(preds.index)

    # Open-to-close return on the predicted day (the day AFTER the signal)
    ret_oc = (close - open_) / open_

    position = preds["y_pred"].astype(float)
    ret_gross = position * ret_oc

    # Cost charged on changes in position (transition 0->1 or 1->0 costs one way)
    turn = position.diff().abs().fillna(position.abs())
    cost = turn * cost_bps * 1e-4

    ret_net = ret_gross - cost

    # Buy-and-hold close-to-close for the same window
    bh_close = bars["close"]
    bh_ret = bh_close.pct_change().reindex(preds.index)

    return pd.DataFrame({
        "position": position,
        "ret_gross": ret_gross,
        "ret_net": ret_net,
        "bh_ret": bh_ret,
    })


# ---------------------------------------------------------------------------
# Summarise results
# ---------------------------------------------------------------------------
def summarize(
    preds: pd.DataFrame,
    ledger: pd.DataFrame,
    periods_per_year: int = 252,
) -> dict:
    """Headline metrics for one walk-forward evaluation run.

    Returns accuracy, HAC t-stat on daily strategy net returns (Newey-West), annualised
    Sharpe vs buy-and-hold, and the number of out-of-sample observations.
    """
    y_true = preds["y_true"].to_numpy(dtype=int)
    y_pred = preds["y_pred"].to_numpy(dtype=int)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else float("nan")

    r_net = ledger["ret_net"].to_numpy(dtype=float)
    r_net = r_net[np.isfinite(r_net)]
    n = r_net.size

    # HAC t-stat on daily net returns (Newey-West)
    tstat = float("nan")
    if n > 10:
        mu = r_net.mean()
        e = r_net - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        tstat = float(mu / se) if se > 0 else float("nan")

    # Annualised Sharpe
    sharpe = float("nan")
    if n > 1 and r_net.std() > 0:
        sharpe = float(r_net.mean() / r_net.std(ddof=1) * np.sqrt(periods_per_year))

    # Buy-and-hold Sharpe
    bh = ledger["bh_ret"].dropna().to_numpy(dtype=float)
    bh_sharpe = float("nan")
    if len(bh) > 1 and bh.std() > 0:
        bh_sharpe = float(bh.mean() / bh.std(ddof=1) * np.sqrt(periods_per_year))

    return {
        "n_oos": int(n),
        "accuracy": accuracy,
        "sharpe_net": sharpe,
        "tstat_net": tstat,
        "bh_sharpe": bh_sharpe,
        "mean_net_bps": float(r_net.mean() * 1e4) if n else float("nan"),
    }
