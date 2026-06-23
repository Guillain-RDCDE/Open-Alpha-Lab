"""Study 364 — FX Carry Trade (borrow low-yield, hold high-yield currencies).

The textbook macro "free lunch": short a low-interest-rate currency (JPY, CHF),
go long a high-interest-rate one (AUD, NZD, ...), and pocket the interest-rate
differential (the *carry*) while spot does roughly nothing on average. Uncovered
interest parity says the high-yielder should depreciate enough to wipe the carry
out; empirically it doesn't — for years at a time — so the carry basket earns a
positive average return and a high Sharpe.

The catch is the *shape* of the distribution. Carry is short volatility: it grinds
up in calm regimes and crashes violently in risk-off episodes (the "carry crash" —
1998 LTCM/yen, 2008 GFC, 2015 CHF de-peg). Average return looks like alpha; the
left tail looks like a sold insurance policy. This study measures the realised
carry premium on a yfinance FX tape, then quantifies the **crash skew** that turns
the "free lunch" into a premium for bearing crash risk.

True overnight deposit rates are not on yfinance, so we attach a **transparent,
fixed carry-by-currency proxy** (a long-run average short-rate differential vs USD)
to each pair, named as a proxy throughout. The synthetic positive control plants a
known carry premium *and* a known fat-tailed unwind so the engine's skew/Sharpe
machinery can be checked against ground truth.

See :mod:`fx_carry_trade.data` (real FX loader + carry proxy + deterministic
synthetic control) and :mod:`fx_carry_trade.strategy` (carry ranking, the long-high
/ short-low basket, Sharpe/skew inference, costs, and the crash-conditional null)."""

from . import data, strategy

__all__ = ["data", "strategy"]
