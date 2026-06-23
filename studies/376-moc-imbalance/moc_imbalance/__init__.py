"""Study 376 — MOC-Imbalance 🔔 (closing-auction pressure & overnight reversal).

The folklore of the **market-on-close (MOC) auction**: large closing-auction *order
imbalances* — especially on index-rebalance days, when passive funds must trade the
official close — are said to **push the last print** away from fair value, which then
**reverses overnight** (the close→open gap leans back the other way). The believers'
trade: fade the imbalance into the close, capture the overnight snap-back.

There is **no free auction-imbalance feed** (the NYSE/Nasdaq MOC imbalance messages are a
paid data product). So we build a **transparent end-of-day-pressure PROXY** from yfinance
daily bars: each day's **intraday displacement** (where the close sits inside the day's
range, signed by the day's direction) stands in for "how hard price was pushed into the
close." We then ask the believers' question directly: does a big into-the-close push
**reverse overnight** (negative close→open return the next morning)? We focus extra
attention on **index-rebalance days** (the third Friday of quarter-end months), when real
MOC imbalances are largest.

The decisive object is the **close→open reversal coefficient**: regress the next overnight
gap on today's signed displacement, with an autocorrelation-robust (HAC) t-stat, a
placebo null, a win-rate vs base rate, a 1-day execution lag, and one-way costs. A
deterministic synthetic control plants a *known* reversal edge so the engine can be
validated (edge=0 must not manufacture significance; a large planted edge must light up).

See :mod:`moc_imbalance.data` (real-tape loader + deterministic synthetic control) and
:mod:`moc_imbalance.strategy` (displacement proxy, overnight-reversal regression, HAC t /
placebo null, win-rate, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
