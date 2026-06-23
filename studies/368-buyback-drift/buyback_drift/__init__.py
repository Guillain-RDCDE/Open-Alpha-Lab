"""Study 368 — Buyback-Drift (does a stock drift up for months after a big repurchase OK?).

The folklore: when a company announces a large share-**buyback authorization**, the stock is
supposed to drift *up* for months — management is "buying its own stock," so the market should
follow. We test the *announcement* (the board OK'ing a $X program), explicitly distinguished
from *execution* (shares actually bought, which happens slowly over years and is not what the
headline reacts to).

We hard-code a transparent table of ~30 notable, well-documented buyback authorizations
(ticker, announcement date, program size) and measure the **abnormal** post-announcement drift —
the event stock's forward return **minus SPY** over 1/3/6 months — against a matched placebo
(the same names on random dates). True point-in-time press-release timestamps for thousands of
authorizations are not on yfinance, so the table is an explicit, named sample; the lesson is
statistical (a few-dozen events, dominated by their own variance) and survives that caveat.

See :mod:`buyback_drift.data` (the event table, real abnormal-return loader, and a deterministic
synthetic positive control with a planted-edge knob) and :mod:`buyback_drift.strategy` (abnormal
forward returns with a 1-day execution lag, Welch t vs a same-names placebo null, win-rate vs the
base rate, and one-way costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
