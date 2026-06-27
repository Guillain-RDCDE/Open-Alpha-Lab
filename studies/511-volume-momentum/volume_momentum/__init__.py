"""Study 511 -- Volume-Momentum (Lee-Swaminathan).

Lee & Swaminathan (2000), "Price Momentum and Trading Volume" (Journal of Finance): trading
volume predicts both the **magnitude** and the **persistence** of price momentum. They sort
stocks first on past return (winners / losers), then on past **trading volume** (turnover), and
find a "momentum life cycle": momentum is strongest among **high-volume winners** and
**low-volume losers**, and -- their sharpest, most distinctive prediction -- **high-volume past
performers (high-volume winners AND high-volume losers) reverse faster**, so volume forecasts
*when* the momentum drift flips to reversal.

We rebuild it cleanly on a fixed ~40-name large-cap **survivor** basket: per name we form the
12-1 momentum signal and a trailing **share-turnover** measure (avg daily volume / shares,
proxied by avg dollar volume rank from yfinance OHLCV), double-sort momentum x volume into a
winners-minus-losers (WML) book inside each volume half, and ask (1) does the Lee-Swaminathan
**high-volume WML > low-volume WML** ordering hold and is it statistically real; (2) does the
spread survive costs; (3) does the **volume-conditioned reversal** show up -- do high-volume
winners give back their drift at longer horizons faster than low-volume winners?

Inference is the desk's shared spirit: a robust one-sample **HAC t**, a seed-robust
**label-shuffle placebo** null, one **forward execution lag**, one-way costs x NAV x turnover
plus borrow on the short leg, and a deterministic **synthetic positive control** (a planted
volume-conditioned momentum effect) that proves the engine is faithful and that zero edge cannot
fake significance.

See :mod:`volume_momentum.data` (real basket loader with OHLCV + deterministic synthetic control
with a volume-conditioned-momentum knob) and :mod:`volume_momentum.strategy` (12-1 momentum, the
turnover measure, the momentum x volume double-sort, the WML books, the volume-conditioned
reversal term-structure, HAC inference, the placebo null, costs).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
