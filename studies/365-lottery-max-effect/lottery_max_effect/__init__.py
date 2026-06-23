"""Study 365 — the Lottery / MAX effect (Bali-Cakici-Whitelaw 2011).

The behavioural claim: investors over-pay for **lottery-like** stocks — names with a
big, recent, eye-catching one-day pop — and those names then *underperform*. Operationalised
as **MAX**: each month, rank the cross-section by the *highest single-day return over the
prior month*; the highest-MAX stocks are the "lottery tickets," the lowest-MAX the "boring"
names. The strategy is **long low-MAX, short high-MAX** (or, equivalently, buy the dull tail
and avoid the flashy one).

True MAX is a CRSP-universe, thousands-of-names result. yfinance gives us only a large-cap
slice, so we run the sort on a **fixed ~90-name S&P-100-style basket** and call it a
**proxy** throughout: a survivorship-tilted, large-cap-only stand-in. The decisive question
the desk asks is not "does the published anomaly exist" (it does, in CRSP small-caps) but
"does *this* tape — large, liquid, survivor names — certify a tradable MAX spread at t ≥ 2,
net of costs?" Spoiler in the verdict; the work is in the notebooks.

See :mod:`lottery_max_effect.data` (real basket loader + a deterministic synthetic
cross-section with a *planted* lottery-preference knob) and
:mod:`lottery_max_effect.strategy` (the MAX signal, quintile sort, long-short spread,
Newey-West / placebo inference, win-rate, 1-day lag, one-way costs + borrow)."""

from . import data, strategy

__all__ = ["data", "strategy"]
