# Study 927 — Dutch Auction 🔨

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Two claims, two answers. The **announcement repricing is real and large**: **+4.72%** abnormal (issuer − SPY) on the SC TO-I session, *t* = **+7.93**, positive in **83%** of **145** events, jackknife LOO [+7.81, +8.86], placebo *p* = **0.0005**, **date-locked** to a single session (+0.43% at *t+1*), present in **both** eras and in the liquid subset. The **"it marks the bottom" claim is absent**: the tender window is **+0.12%** (*t* = +0.17, placebo *p* = 0.91) and the six-month post-expiry drift is +2.76% on a **−2.36% median** (*t* = +0.71). Named biases: SEC's *current* CIK→ticker register (**survivorship**); the screens need **148 sessions of tape after** the event, dropping any issuer taken out within ~7 months of its own tender — the very outcome that would print a big positive drift, so the flat drift is an **upper bound**; and the "modified Dutch auction" phrase filter (**visibility**). |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The paying session is the one you cannot be positioned for. Buy one session after the filing and hold to expiry: **−0.28% net** per full-size trade at 10 bps; the calendar-time long/short sleeve compounds to **−69.2%** (Sharpe **−0.302**, HAC *t* −1.14) and is **−0.169 gross**, so the tape kills it before costs do; the long-only basket trails plain SPY by **0.70 excess-of-cash Sharpe**, HAC *t* **−2.00** on the daily mean difference — right on the two-sigma line. Costs are charged at each slot's real portfolio weight, not a full-NAV round trip per event; cost, borrow and expiry sweeps only make it worse. |

> **In one sentence:** across **145** issuer modified-Dutch-auction self-tenders filed on form SC TO-I (2010–2025), the tape reprices the stock **+4.72% versus SPY in the single session the filing lands** — one of the cleanest event effects on this desk — and then does **nothing whatsoever** for the buyer who arrives one day late, so the premium goes to the holders who tender, not to the tape.

## What we tested

Rather than a remembered list, the event set is the machine-readable output of **one EDGAR
full-text-search query** (`q="modified Dutch auction"`, `forms=SC TO-I`, 2010–2025),
clustered per registrant, with each row carrying its **SEC accession number** so the table
is checkable filing by filing: **180** offers, **145** surviving mechanical screens (cached
tape, ≥252 sessions before, ≥148 after, price ≥ $5) on **109** issuers. We measure four
things and never blend them — the (−5, −1) leakage run-up, the announcement-day abnormal
return, the **tradable** tender window (enter at the close of *t+1*, **one** execution lag,
hold to the expiry proxy) and the post-expiry drift at 1 / 3 / 6 months — all excess of
**SPY** on daily **total-return** closes. Inference: cross-event *t*, jackknife, block
bootstrap, a same-names random-date placebo, and a **calendar-time portfolio** with a HAC
*t* for the overlapping-event problem, plus an era cut, a $10m/day liquidity cut, and sweeps
of every non-tape input (cost, borrow, the **expiry PROXY** at Rule 14e-1's 20-session
minimum, the event date ±5 sessions). Costs are booked at each slot's **real portfolio
weight** on the days it opens and closes, not as a full-NAV round trip per event — the
shortcut would have overstated the drag 1.55× and let the cost model carry the verdict. **Dedup:** distinct from **368-buyback-drift** (open-market
repurchase *authorisations* — no obligation to buy — on 32 remembered mega-caps),
**564-short-report-event** (same machinery, short side) and **390-activist-13d** (an
*external* blockholder, not the issuer itself).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Dutch-auction self-tender is, why the whole effect is one day wide, why the biggest pops are on the names you cannot trade, and the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the window decomposition, jackknife / bootstrap / placebo / date-shift, the calendar-time portfolio, the excess-of-cash race, every sweep, and the Jensen-term finding hiding in the synthetic null |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (return fp `c39ea6c65ccb`, as-of 2026-06-30): [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`dutch_auction/`](dutch_auction/). The event set is an EDGAR filing search, not a point-in-time universe (survivorship, post-event-survival and phrase-visibility all named on the Signal axis). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
