# Study 928 — Odd-Lot Priority

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | **Real on the announcement run-up · None on the give-back the claim needs.** The pop takes **+7.41%** of the premium before an outside buyer can act (abnormal +6.87%, HAC *t* = **+9.06**) — pure tape, no proxy. The post-expiry give-back odd-lot priority feeds on is **absent**: −0.50%, *t* = −1.17, median 0.00%. Every *profitable* number here is the **assumed** 13% clearing premium meeting the tape — the +3.52% hedged round trip (*t* = +4.32) **and** the +3.39% "value of priority" (*t* = +4.33) alike; they flip sign at 9% and 7.4% respectively. Premium-free, the breakeven clearing premium is **10.17%**, rising to **11.99%** post-2018 (a change that is itself significant, Welch *t* = +2.49) and to **12.91%** on a micro-cap 100 bps touch, where the round trip is +0.97% (*t* = +1.19). Survivorship: acquired / taken-private issuers drop out of the ticker map. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The rule that creates the edge caps it at **99 shares**: ~**$176 per event**, ~**$1,859 a year** per account at a $50 share price — and **$118 a year** at $10, where the flat broker corporate-action fee alone is 300 bps and *t* falls to +1.37. Scaling is a legal constraint, not a capital one, and the scaled fantasy sleeve loses the excess-of-cash Sharpe race to SPY (**+0.715 vs +0.799**) at 66% vol. |

> **In one sentence:** The announcement pop takes 7.4 points of a low-teens premium before you can buy, the post-expiry give-back that odd-lot priority feeds on has been statistically absent for fifteen years, and what the entitlement is "worth" turns out to be the premium you assumed rather than a number the tape printed — capped, either way, at ninety-nine shares.

## What we tested

Buy an odd lot at the close of **t+1** (the offer is public at the close of **t** — the one
execution lag), tender all of it, be filled **in full** under the priority clause, and be
out at expiry. Same **180 filed modified-Dutch-auction self-tenders** as Study 927 (SC TO-I,
EDGAR full-text search, accession numbers included) — 178 after the recycled-symbol
exclusion, 128 of 129 issuer tickers with a recoverable tape, 167 fully covered windows,
2010-01-28 → 2025-11-21, total-return closes vs **SPY** and **BIL**, as-of 2026-06-30. Four
inputs are **not on the tape** and are declared PROXIES, swept end to end: the clearing
premium (13%), the offer length (21 trading days, the Rule 14e-1 minimum), the round-lot
proration fill (0.35) and the flat broker fee ($30 on a $5,000 lot). One-way cost swept
0–100 bps × NAV (charged three times per hedged round trip, and the list is a third
micro-cap); the SPY hedge pays borrow (swept to 300 bps). **Dedup:** shares its event list
with **927-dutch-auction-buyback** deliberately — 927 asks whether a self-tender marks the
issuer's bottom; 928 asks what the odd-lot *entitlement* is worth to the holder who
tenders, and reports a **breakeven premium** instead of inventing clearing prices. Distinct
from **929-rights-offering-discount** and **931-cef-ipo-decay** (other corporate actions,
different mechanics) and from **926-t-plus-one-settlement** (plumbing, no event list).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the forum trade in plain language, why the pop is not yours, why priority needs a give-back that never comes, and the ninety-nine-share ceiling |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the two round-trip identities, the premium-free breakeven and what it still assumes, every proxy swept, the era cut with the *contrast* tested, the calendar-time excess-of-cash race, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`odd_lot/`](odd_lot/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
