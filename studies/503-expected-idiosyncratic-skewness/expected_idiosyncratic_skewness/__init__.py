"""Study 503 — Expected Idiosyncratic Skewness (Boyer-Mitton-Vorkink 2010).

The behavioural claim: investors **over-pay for lottery-like upside**, so stocks with high
*expected idiosyncratic skewness* — names whose own-return distribution is fat on the right
tail once you strip out the market — earn **lower** future returns. Boyer, Mitton & Vorkink
(*Review of Financial Studies*, 2010) model expected idiosyncratic skewness and show it
negatively predicts the cross-section: high expected idio-skew → low subsequent return.

We proxy *expected* idiosyncratic skewness the simplest honest way: each month, regress each
name's daily returns on the market over a trailing window, and take the **skewness of the
residuals** (the part of the return not explained by the market). Trailing residual skew is the
standard, transparent predictor of next-period idio-skew. We sort the cross-section, go **long
the low-skew quintile, short the high-skew quintile**, and ask the desk's only question: does a
long-low / short-high idio-skew book clear *t* ≥ 2, net of costs, on the tape it actually ran?

This is **distinct** from two neighbours the brief flags. (1) [Study 365 — Lottery-MAX]
(../365-lottery-max-effect/) sorts on the single highest *daily* return (a one-day pop), not on
the *shape* of the whole residual distribution. (2) **Coskewness** is a *systematic* (market-
co-movement) tail measure; idiosyncratic skewness is the *own-name*, market-orthogonal tail.

True expected idio-skew is a CRSP-universe object (thousands of names incl. small-caps, where
the lottery effect is strongest). yfinance gives only a large-cap slice, so we run the sort on a
fixed ~66-name S&P-100-style basket and call it a **proxy** throughout — survivorship-tilted,
named on the Signal axis everywhere.

See :mod:`expected_idiosyncratic_skewness.data` (real basket loader + market proxy + a
deterministic synthetic cross-section with a *planted* skewness-preference knob) and
:mod:`expected_idiosyncratic_skewness.strategy` (the residual-skew signal, quintile sort,
long-short spread, Newey-West / placebo inference, 1-day lag, one-way costs + borrow)."""

from . import data, strategy

__all__ = ["data", "strategy"]
