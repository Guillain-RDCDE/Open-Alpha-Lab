"""Study 388 — Lumber-Gold-Ratio (a risk-on/risk-off switch for stocks and bonds?).

The folk claim (Gayed & Bilello, 2015): the **lumber/gold ratio** is a clean intermarket
regime gauge. Lumber is cyclical (housing, growth); gold is defensive (fear). So when lumber
out-runs gold the market is *risk-on* — hold **stocks (SPY)**; when gold out-runs lumber it is
*risk-off* — hold **long bonds (TLT)**. We rebuild that rotation switch, race it against
buy-and-hold SPY, a static 60/40 blend, and a same-exposure **random-timing** control, all on an
excess-of-cash basis with one execution lag and one-way costs, and ask whether the *timing*
beats the passive mix by more than luck.

True lumber-futures history (``LBS=F``) ends in May 2023 (contract discontinued), so for a
current 2008→2026 window we use the **WOOD** timber-equity ETF as a transparent **lumber proxy**
— named on the Signal axis.

See :mod:`lumber_gold_ratio.data` (real tape loader + deterministic synthetic control with a
planted-edge knob) and :mod:`lumber_gold_ratio.strategy` (the rotation switch, the excess-of-cash
race, HAC inference, block-bootstrap CIs, and costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
