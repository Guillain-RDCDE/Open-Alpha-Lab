"""Study 504 — Coskewness (Harvey-Siddique 2000).

The risk-pricing claim: a stock's **coskewness** with the market — how much it contributes to the
*market's* own (left-tail) skewness — is a systematic, undiversifiable risk. Harvey & Siddique
(*Journal of Finance*, 2000) show coskewness is **priced**: names with **low (more negative)**
coskewness sell off hardest exactly when the market is most volatile / crashing, so they are poor
hedges and must offer a **premium**; high-coskewness names act like insurance and earn less. The
natural trade is **long low coskewness, short high coskewness**.

We measure coskewness the simplest honest way: each month, take Harvey-Siddique's **direct
standardised coskewness** of the name's daily returns with the market's, over a trailing window —
``E[e_i * e_m^2] / (sqrt(E[e_i^2]) * E[e_m^2])`` on demeaned daily returns. We sort the
cross-section, go **long the low-coskew quintile, short the high-coskew quintile**, and ask the
desk's only question: does a long-low / short-high coskewness book clear *t* ≥ 2, net of costs,
on the tape it actually ran?

This is **distinct** from [Study 503 — Expected Idiosyncratic Skewness]
(../503-expected-idiosyncratic-skewness/), which sorts on the skew of the market-model *residual*
— the *idiosyncratic*, diversifiable tail a behavioural lottery-buyer over-pays for. Coskewness
is the opposite axis of the same regression: the **systematic** co-movement of a name's tail with
the market's tail, which a diversified investor *cannot* shed and is *rewarded* for bearing.

True coskewness pricing is a CRSP-universe object (thousands of names). yfinance gives only a
large-cap slice, so we run the sort on a fixed ~79-name S&P-100-style basket and call it a
**proxy** throughout — survivorship-tilted, named on the Signal axis everywhere.

See :mod:`coskewness.data` (real basket loader + market proxy + a deterministic synthetic
cross-section with a *planted* coskewness-premium knob) and :mod:`coskewness.strategy` (the
direct-coskewness signal, quintile sort, long-short spread, Newey-West / placebo inference,
1-day lag, one-way costs + borrow)."""

from . import data, strategy

__all__ = ["data", "strategy"]
