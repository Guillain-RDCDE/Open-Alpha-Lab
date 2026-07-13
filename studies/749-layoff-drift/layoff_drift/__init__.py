"""Study 749 — Layoff-Drift (does a mass-layoff announcement pop, then drift?).

Market folklore frames a mass-layoff announcement as a *bullish* catalyst: the market
cheers the cost discipline (the "restructuring pop"), and — in the 2022–2024 tech
"efficiency" narrative — the stock keeps drifting up as margins improve (a PEAD-style
continuation). The bear reads it the opposite way: layoffs signal distress, so the stock
sags. We pin the question down with a clean event study: around a dated layoff
announcement, what is the stock's *abnormal* return (market-model adjusted) over a short
**pop** window and a longer **drift** window — and does either clear |t| ≥ 2?

There is no free, survivorship-clean database of mass-layoff dates, so we hardcode a
transparent table of ~28 well-known large-cap layoff announcements (ticker, date,
approximate cut) and compute event-window abnormal returns from yfinance daily closes vs
SPY. The decisive finding is statistical, not directional: with ~two dozen heterogeneous
events, both the pop and the drift are swamped by their own standard error — a
small-sample event-study illusion (cf. Study 391 CEO-Turnover, another rare-catalyst
event study, and Study 389 Name-Change-Effect, the pop-then-fade legend).

See :mod:`layoff_drift.data` (hardcoded event table + yfinance loader + deterministic
synthetic positive control with plantable pop/drift edges) and
:mod:`layoff_drift.strategy` (market-model abnormal returns, pop/drift CAR windows,
Welch t + Newey-West HAC t on the pooled daily drift, placebo null, one-day execution
lag, one-way costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
