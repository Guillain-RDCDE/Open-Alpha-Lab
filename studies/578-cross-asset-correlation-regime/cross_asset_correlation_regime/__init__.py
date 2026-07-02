"""Study 578 — Cross-Asset-Correlation-Regime.

The folklore: *"When everything starts moving together, the market is about to break."* A rising
**average cross-asset correlation** is read as a fragility indicator — diversification quietly
evaporates just before a stress event, so a high/rising correlation regime should *predict* lower
forward returns and higher forward volatility on a risk asset (here SPY).

We build the classic measure — the mean pairwise correlation across a multi-asset ETF panel over a
trailing window — split the tape into a HIGH-correlation regime and a LOW-correlation regime, and
test whether SPY's forward return / forward volatility differ across the two. A label-shuffle
placebo null, a multi-window robustness sweep, cost-honest labelling, and a seed-robust synthetic
positive control complete the machinery.

Distinct from this desk's neighbours: [Study 245 — Oil-Equity-Correlation](../245-oil-equity-correlation/)
tracks a *single pair's* correlation; [Study 502 — Betting-Against-Correlation](../502-betting-against-correlation/)
is a *cross-sectional* stock sort on correlation-to-market. Study 578 is a **time-series regime**
signal from the *panel-average* correlation, tested as a fragility/drawdown predictor.
"""
