"""Study 383 — SOFR-Repo-Stress (do funding-market spikes warn risk assets?).

The folklore: when the overnight repo / SOFR rate *spikes* — most famously the
September 2019 blow-up, when SOFR jumped to ~5% and overnight repo briefly printed
near 10% — it is a flashing red light that the *plumbing* of the financial system is
seizing, and risk assets (equities, credit) are about to crack. "Watch the repo
market" became a permanent fixture of macro-Twitter after Sept 2019.

We test it the only honest way the public data allows. There is **no free, clean
daily SOFR-stress tape on yfinance**, so we work from a **hardcoded, sourced table of
the named repo-stress episodes** (Sept 2019, the year-end turns, March 2020, the
2023 banking stress, etc.) and measure the forward reaction of risk assets — SPY
(equities) and HYG / LQD (credit ETFs) — around each event, vs the unconditional base
rate, with a 1-day execution lag, a Welch *t*, and a placebo / randomization null
sized to the event count.

The decisive finding is statistical, not directional: a "signal" with **a dozen-ish
named lifetime episodes** — several of which are *not even bearish for stocks*, and
the one true crash (March 2020) is COVID, not repo — cannot be a tradable warning. It
is the same sample-size + confounding pathology the desk has seen in every rare-event
"oracle" (cf. the rare-breadth Zweig / Hindenburg studies).

See :mod:`sofr_repo_stress.data` (real risk-asset loader + hardcoded episode table +
deterministic synthetic control) and :mod:`sofr_repo_stress.strategy` (event-window
forward returns, Welch *t*, placebo null, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
