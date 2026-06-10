"""Study 39 — Black-Box: a neural net fed crypto OHLCV, the in-sample-vs-out-of-sample trap.

Kakushadze & Serur's *151 Trading Strategies* §18.2 catalogues a **neural-network crypto trader**: feed
an MLP a handful of price-derived features and let it predict tomorrow's direction. Fit-and-predict on
the same tape and it scores beautifully — high accuracy, a gaudy Sharpe. The desk's question (the
backtest-trap lineage of Study 22 Crystal-Ball): does that edge survive **walk-forward** out-of-sample,
where the net must trade days it never saw while fitting? The honest finding: the in-sample Sharpe is a
mirage of an overparameterised learner memorising noise — the walk-forward edge collapses to ~0 and is
negative after costs.
"""

from . import costs, data, extension, features, strategy  # noqa: F401
