"""Study 370 — the 0DTE-options boom and the SPX tape.

Believers' claim: since same-day (0DTE) options went *daily* in 2022, the SPX/SPY tape is
more **mean-reverting / more pinned** intraday — dealers hedging a wall of at-the-money,
same-day gamma supposedly damp and reverse moves into the close.

There is no free intraday SPX option tape, so the intraday phenomenon cannot be measured
head-on here. We test the claim on **explicit daily proxies**: the lag-1 autocorrelation of
the daily open→close ('intraday') leg (a mean-reversion gauge), the same-day realised range,
and a one-shot nearest-expiry option-chain snapshot (the only direct 0DTE evidence yfinance
gives — a current at-the-money volume cluster, labelled a snapshot). We split SPY at 2022 and
ask whether the post-boom regime is *measurably* more mean-reverting.

The decisive finding is methodological: the cross-2022 change on the proxy is small and
statistically indistinguishable from a random split of the same tape — and a daily feed
*cannot* see the intraday microstructure the story is about. A deterministic synthetic
control with a planted pinning regime switch confirms the engine detects a *real* break and
does not invent one when none exists.

See :mod:`zero_dte_options.data` (real SPY/option loader + synthetic pinning control) and
:mod:`zero_dte_options.strategy` (autocorr signal, block-bootstrap Welch t, placebo-break
null, the naive pin trade, option-chain concentration)."""

from . import data, strategy

__all__ = ["data", "strategy"]
