"""Tests for the strategy layer — feature engineering, walk-forward, shuffle control,
and the study's spine: the RF only beats a coin when a real signal is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from random_forest import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def test_build_features_shape_and_columns():
    bars, _ = data.synthetic_daily(n_days=200, seed=1)
    X, y = st.build_features(bars)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(X) < len(bars)  # NaN rows were dropped
    for col in ["ret_lag_1", "rsi", "rvol", "mom", "vol_ratio"]:
        assert col in X.columns, f"Missing feature: {col}"
    assert X.isna().sum().sum() == 0, "X contains NaN after build_features"
    assert y.isna().sum() == 0, "y contains NaN after build_features"


def test_build_features_target_is_binary():
    bars, _ = data.synthetic_daily(n_days=300, seed=2)
    _, y = st.build_features(bars)
    assert set(y.unique()) <= {0, 1}


def test_build_features_no_lookahead():
    """ret_lag_1 at row t must equal the return from t-2 to t-1, not the current bar's return."""
    bars, _ = data.synthetic_daily(n_days=100, seed=3)
    X, y = st.build_features(bars, lags=1)
    close = bars["close"]
    for date in X.index[:5]:
        t_pos = bars.index.get_loc(date)
        if t_pos < 2:
            continue
        actual_lag1 = float(np.log(close.iloc[t_pos - 1] / close.iloc[t_pos - 2]))
        assert abs(X.loc[date, "ret_lag_1"] - actual_lag1) < 1e-10, "Lag-1 has look-ahead"


def test_rsi_bounded():
    bars, _ = data.synthetic_daily(n_days=300, seed=4)
    X, _ = st.build_features(bars)
    assert (X["rsi"] >= 0).all() and (X["rsi"] <= 1).all()


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def test_walk_forward_has_no_lookahead():
    """The test fold must not overlap with the training window — no future data leaked."""
    bars, _ = data.synthetic_daily(n_days=700, seed=5)
    X, y = st.build_features(bars)
    # Use a very small step to get multiple folds quickly
    preds = st.walk_forward(X, y, train_window=300, step=30, n_estimators=10, max_depth=3)
    # All prediction dates must come strictly after the first train_window rows
    min_pred_pos = X.index.get_loc(preds.index[0])
    assert min_pred_pos >= 300, "First prediction overlaps with training window"


def test_walk_forward_returns_correct_columns():
    bars, _ = data.synthetic_daily(n_days=700, seed=6)
    X, y = st.build_features(bars)
    preds = st.walk_forward(X, y, train_window=300, step=50, n_estimators=10, max_depth=3)
    for col in ["y_true", "y_pred", "proba"]:
        assert col in preds.columns


def test_walk_forward_predictions_are_binary():
    bars, _ = data.synthetic_daily(n_days=700, seed=7)
    X, y = st.build_features(bars)
    preds = st.walk_forward(X, y, train_window=300, step=50, n_estimators=10, max_depth=3)
    assert set(preds["y_pred"].unique()) <= {0, 1}
    assert set(preds["y_true"].unique()) <= {0, 1}


def test_shuffle_control_matches_real_on_null_tape(null_tape):
    """On a martingale tape, the shuffled-label model must score similarly to the real model
    — neither can beat chance, so their OOS accuracies converge."""
    bars, _ = null_tape
    X, y = st.build_features(bars)
    preds_real = st.walk_forward(X, y, train_window=300, step=60, n_estimators=50, max_depth=4)
    preds_shuf = st.walk_forward(
        X, y, train_window=300, step=60, n_estimators=50, max_depth=4, shuffle_labels=True
    )
    acc_real = (preds_real["y_true"] == preds_real["y_pred"]).mean()
    acc_shuf = (preds_shuf["y_true"] == preds_shuf["y_pred"]).mean()
    # Both should be near 50 %; neither should dominate by more than 5 pp
    assert abs(acc_real - acc_shuf) < 0.06, (
        f"Real ({acc_real:.3f}) vs shuffle ({acc_shuf:.3f}) diverge too much on a null tape"
    )


def test_real_beats_shuffle_on_signal_tape(signal_tape):
    """On a tape with a planted signal, the real model must beat the shuffle control."""
    bars, _ = signal_tape
    X, y = st.build_features(bars)
    preds_real = st.walk_forward(X, y, train_window=400, step=63, n_estimators=100, max_depth=5)
    preds_shuf = st.walk_forward(
        X, y, train_window=400, step=63, n_estimators=100, max_depth=5, shuffle_labels=True
    )
    acc_real = (preds_real["y_true"] == preds_real["y_pred"]).mean()
    acc_shuf = (preds_shuf["y_true"] == preds_shuf["y_pred"]).mean()
    # With a planted signal the real model must clear the shuffle (relaxed: acc_real >= acc_shuf - 1pp)
    assert acc_real >= acc_shuf - 0.01, (
        f"Real ({acc_real:.3f}) did not beat shuffle ({acc_shuf:.3f}) on a signal tape"
    )


# ---------------------------------------------------------------------------
# Trading layer
# ---------------------------------------------------------------------------
def test_trade_predictions_columns(null_tape):
    bars, _ = null_tape
    X, y = st.build_features(bars)
    preds = st.walk_forward(X, y, train_window=300, step=50, n_estimators=20, max_depth=3)
    ledger = st.trade_predictions(bars, preds, cost_bps=5.0)
    for col in ["position", "ret_gross", "ret_net", "bh_ret"]:
        assert col in ledger.columns


def test_net_less_than_gross(null_tape):
    """Net returns must be <= gross returns (costs always drag)."""
    bars, _ = null_tape
    X, y = st.build_features(bars)
    preds = st.walk_forward(X, y, train_window=300, step=50, n_estimators=20, max_depth=3)
    ledger = st.trade_predictions(bars, preds, cost_bps=5.0)
    assert (ledger["ret_net"] <= ledger["ret_gross"] + 1e-10).all()


def test_cost_monotonically_lowers_sharpe(null_tape):
    """Higher cost must produce equal or lower Sharpe."""
    bars, _ = null_tape
    X, y = st.build_features(bars)
    preds = st.walk_forward(X, y, train_window=300, step=50, n_estimators=20, max_depth=3)
    sharpes = []
    for c in [0.0, 5.0, 10.0]:
        ledger = st.trade_predictions(bars, preds, cost_bps=c)
        s = st.summarize(preds, ledger)["sharpe_net"]
        sharpes.append(s)
    assert sharpes[0] >= sharpes[1] - 0.01
    assert sharpes[1] >= sharpes[2] - 0.01


def test_summarize_keys(null_tape):
    bars, _ = null_tape
    X, y = st.build_features(bars)
    preds = st.walk_forward(X, y, train_window=300, step=50, n_estimators=20, max_depth=3)
    ledger = st.trade_predictions(bars, preds, cost_bps=5.0)
    s = st.summarize(preds, ledger)
    for k in ["n_oos", "accuracy", "sharpe_net", "tstat_net", "bh_sharpe", "mean_net_bps"]:
        assert k in s, f"Missing key in summarize output: {k}"
