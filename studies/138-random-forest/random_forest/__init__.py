"""Study 138 — Random-Forest.

Does a Random Forest, trained on standard technical features (lagged returns, RSI,
realised volatility, momentum, volume ratio), predict next-day SPY direction better
than chance?  We evaluate with strict walk-forward cross-validation (never touch the
future) and a shuffled-label control (train on permuted targets — if the shuffled
model scores the same, the 'edge' was overfitting).  Distinct from Study 39
(neural net on crypto): tree ensemble on equities, explicit feature-importance
transparency, and the shuffle test as the primary falsification tool.

Expected verdict: NONE / MIRAGE — out-of-sample accuracy hovers at ~50 % and the
shuffled-label control matches it, confirming that any in-sample signal is pure
overfitting.
"""
