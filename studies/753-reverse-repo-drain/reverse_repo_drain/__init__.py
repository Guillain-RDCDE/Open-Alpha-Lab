"""Study 753 — Reverse-Repo-Drain 💵 (does a draining ON RRP mark risk-on?).

The pitch, straight off liquidity-plumbing FinTwit: the Fed's **Overnight Reverse Repo
(ON RRP) facility** is a giant parking lot for idle money-market cash. When that balance
is **draining** — falling from its ~$2.55T Dec-2022 peak back toward zero — the cash is
(the story goes) flowing out of the RRP and into risk assets, so a draining RRP marks a
**risk-on** regime and you should be long stocks. A *filling* RRP is liquidity leaving
markets: be cautious.

The ON RRP balance isn't on yfinance, so we ship a **small, clearly-labelled, hardcoded
monthly proxy** transcribed from the public FRED series ``RRPONTSYD`` / NY Fed operating-
desk prints (see :mod:`reverse_repo_drain.data`), aligned to month-end SPY. We then split
next-month SPY returns by whether the RRP was draining, with a one-month execution lag.

The decisive caution is on the Signal axis: the RRP's entire meaningful life is **one
fill-then-drain episode** — the 2021 ramp, the 2022 peak, the 2023-25 drain — that happens
to straddle the 2022 bear and the 2023-24 bull. A "drain = risk-on" spread is an n=1 macro
coincidence, and a block-bootstrap null that respects the long regimes says exactly how
easily one big drain lining up with one big rally arises by chance.

See :mod:`reverse_repo_drain.data` (the RRP proxy + SPY loader + a deterministic synthetic
positive control with a PLANTED drain edge) and :mod:`reverse_repo_drain.strategy` (the
drain regime split, Welch t / block-bootstrap placebo, and a net-of-cost timing backtest)."""

from . import data, strategy

__all__ = ["data", "strategy"]
