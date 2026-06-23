"""Study 367 — Closed-End Fund Discount (does a deep discount pay you to wait?).

Closed-end funds (CEFs) issue a fixed share count and trade on an exchange like a
stock, so their market **price** can wander away from the **net asset value (NAV)** of
the securities they hold. A fund trading below NAV is "at a discount"; the folklore —
and a real strand of the academic literature — says buying the *widest* discounts pays
you to wait, because the discount mean-reverts back toward NAV.

True per-fund daily NAV is **not** on yfinance (it serves market price only). So we build
a **transparent NAV proxy**: each CEF is mapped to a published benchmark for its mandate
(e.g. an S&P-500 fund -> SPY, a utilities fund -> XLU), and the proxy discount is the
fund's price path measured *relative to* that benchmark, demeaned per fund. We label it a
proxy everywhere. We then sort funds by proxy-discount each month, buy the widest-discount
basket, and test whether it out-earns the narrowest (and the equal-weight CEF universe),
with a Welch t, a block-bootstrap null, costs, and a survivorship caveat named on the
Signal axis.

See :mod:`closed_end_fund_discount.data` (real CEF loader + NAV proxy + deterministic
synthetic control with a mean-reverting discount and a PLANTED edge knob) and
:mod:`closed_end_fund_discount.strategy` (discount signal, long-wide / long-short
portfolios, Welch t, block-bootstrap / placebo null, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
