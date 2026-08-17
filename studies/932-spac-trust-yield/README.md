# Study 932 — Trust Yield 🏦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Real](https://img.shields.io/badge/Real-1e8449?style=flat-square) | Buying below trust and redeeming paid **+1.34%** per 302-day position, cluster CI over 25 shells **[+1.04%, +1.63%]**, positive in **both** rate eras. **That headline is an identity** — with the payoff imposed at the accrued trust, the excess *is* the entry discount (residual 3e-04), so its *t* of +17.9 measures the discount, not the redemption. The stamp rests on what is **not** assumed: the deadline-day quote sat at or above the assumed trust line in **14/25** shells and never >2.9% below it; the assumption-free sell-at-quote version clears *t* = **+2.54** one-bet-per-name; and letting the tape veto the $10.00 assumption shell by shell (`market_floor`) still leaves **+0.83%**, CI [+0.28%, +1.28%]. **Survivorship:** successor-ticker, no-reverse-split shells only — liquidations and split names are absent, so the *count of chances* is flattered (the payoff is not: liquidation pays the trust too). |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Fragile-e67e22?style=flat-square) | **+1.2%/yr over T-bills as a book** — the per-position yield annualises to +1.6%, and the +2.31% mean-of-annualised is an artefact of short holds, not a rate of return. That is for a ~10-month lock-up in a $200-300m shell trading a few hundred thousand dollars a day. It survives 100 bps of one-way friction (**+0.30%/yr**) and **dies at 200 bps** (−0.78%/yr); it more than halves if the trust was **$9.90** rather than the assumed $10.00 (+0.56%/yr); a broker's voluntary corporate-action fee on the redemption election is not modelled at all; it assumes a deadline that 2022-23 sponsors kept extending; and the opportunity set fell from **16** live pre-deal shells in 2021 to **1** by 2024. |

> **In one sentence:** The pre-deal SPAC discount was exactly what the arithmetic promised — and *arithmetic* is the word, because the study pays itself the trust by assumption, so what the tape really establishes is narrower: the $10.00-accreted line was where the market actually was when the put expired, the discount to it was reliably there, and capturing it was worth about **one and a quarter percent a year over bills** in a market that has since closed behind it.

## What we tested

At each **month-end**, buy every SPAC still in its pre-deal window quoted **below** its
accrued trust value with 30-730 days left; execute at the **next** close (one lag, 15 bps
one-way × NAV); hold to the redemption deadline and **redeem at trust**, never selling into
the market. Raced **excess-of-cash** against BIL over identical windows, on **31 hardcoded
2019-2022-vintage SPACs** read as **unadjusted** closes off their successor tickers
(2019-03-27 → 2024-04-10). PROXIES/ASSUMPTIONS, all swept: trust **$10.00** at IPO accreting
at BIL, deadline = deal close **− 30 days**, fee drag **0 bp**, deep-quote guard 12%. The
deadline and the 30-730-day horizon filter are built from the **realised** deal date — a
look-ahead, named and shown inert (lifting the cap to 3000 days: +1.34% → +1.36%).
**Dedup:** distinct from **931-cef-ipo-decay** (the mirror $10 wrapper — there you *pay* a
load into a fund that slides to a discount; here you *buy* a discount a contract must repay),
**929-rights-offering-discount** and **928-odd-lot-tender** (discounts granted by an *issuer*,
not by the market against a put), **930-when-issued-spinoff** (a two-price window with no
trust), and **921-bill-ladder-vs-etf** / **922-frn-vs-fixed-front-end** (the cash legs this
study measures a spread *over*).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what you actually own before a deal closes, why the discount opened in 2022, the four things that could have gone wrong, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | position-level excess, **the identity teardown** and the three unassumed reads that replace it, cluster bootstrap vs the over-sized naive *t*, accrual vs mark-to-market books, the three annualisations, era cut, four assumption sweeps, two cross-checks, the yield path, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`trust_yield/`](trust_yield/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
