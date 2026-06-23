"""Study 391 — CEO-Turnover (is firing the CEO good news or bad news?).

Market folklore cuts both ways: a "forced out" CEO is sometimes cheered ("activists
won, the underperformer is gone") and sometimes feared ("chaos at the top"). We pin
the question down with a clean event study: around a dated CEO-change announcement,
what is the stock's *abnormal* return (CAR over a short window, market-model adjusted)
— and does it differ between **forced** departures and **planned/orderly** ones?

True CEO-change databases (ExecuComp, BoardEx) are not free, so we hardcode a
transparent table of ~25 notable large-cap CEO changes (ticker, announcement date,
forced vs planned) and compute event-window abnormal returns from yfinance daily
closes vs SPY. The decisive finding is statistical, not directional: with ~a dozen
events per bucket, the forced-minus-planned CAR difference is swamped by its own
standard error — a sample-size event-study illusion (cf. Study 249 Index-Inclusion,
where a *frequent* corporate event leaves a measurable but tiny print; here the event
is rare *and* heterogeneous).

See :mod:`ceo_turnover.data` (hardcoded event table + yfinance loader + deterministic
synthetic positive control with a planted CAR edge) and :mod:`ceo_turnover.strategy`
(market-model abnormal returns, CAR windows, Welch t / placebo null, win-rate vs base
rate, 1-day execution lag, one-way costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
