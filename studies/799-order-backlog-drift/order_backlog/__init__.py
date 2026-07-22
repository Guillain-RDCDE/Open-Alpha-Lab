"""Study 799 — Order-Backlog Drift (Remaining Performance Obligations / RPO).

Does growth in a company's order backlog — the ASC-606 ``RevenueRemainingPerformance
Obligation`` (RPO): contracted revenue a firm has *signed* but not yet recognised — lead
its forward stock returns? Rank names on YoY RPO growth, go long the fastest growers,
short the slowest, and measure what the market pays.

Offline & deterministic once cached. See ``data.py`` (EDGAR RPO + prices + a seeded
synthetic positive control) and ``strategy.py`` (portfolio sorts + HAC inference + a
costed timer).
"""
