# Study 800 — High-Frequency (Weekly) Reversal ⚡

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do last week's losers beat last week's winners next week? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The raw weekly loser-minus-winner spread is **+17.5 bps/wk (+9.1%/yr)**, HAC *t* = **+2.47** — it *clears* the bar and is near-market-neutral (beta **0.29**, so not disguised beta). But it is **not robust**: inserting a one-week gap between forming and holding drops it to **−1.7 bps, *t* = −0.28** (a clean sign-flip to zero), and even the *raw* effect has decayed to nothing since ~2021 (**+2.9 bps, *t* = +0.26**). A reversal that lives only when the same Friday close both forms the signal *and* prices the entry is **bid-ask bounce**, not an edge. Survivorship-biased upper bound (current S&P 500 projected backwards; delisted losers absent). |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Each leg turns over **~78% one-way every week**; break-even flat cost is **~5.6 bps** all-in. The empirical **Corwin-Schultz bid-ask-bounce haircut** — each leg pays its *own* effective spread — is **−28.8 bps/wk even at half-spread** (losers are illiquid and pay most). A weekly-refreshed long-short book of a signal that already evaporates one week out cannot be traded. |
| **Is this really reversal, or bid-ask bounce?** | ![Confirmed](https://img.shields.io/badge/Bid--ask%20bounce%3F-Confirmed-8b949e?style=flat-square) | The single decisive control: a **one-week skip** removes **100%** of the spread (*t* +2.47 → −0.28). The synthetic control proves the machinery — a *planted real* reversal survives the skip (*t* +16 → +21) while a *planted pure bounce* dies (*t* +11 → −1). The real tape shows the **bounce** signature. |

> **In one sentence:** last week's losers really do out-return last week's winners next
> week — **+17.5 bps/wk, HAC *t* = +2.47** — but a one-week gap kills the entire spread
> (*t* = −0.28) and the losers' own bid-ask spread costs more than the edge is worth, so it
> is **bid-ask bounce dressed as alpha**, not a tradable weekly reversal.

## What we tested

The pitch that a **weekly** reversal is an amplified, cleaner cousin of the one-month
reversal: each **Friday**, rank the S&P 500 cross-section on **last week's** five-day
return, long the bottom quintile (losers) and short the top quintile (winners), hold one
week, rebalance — **total-return weekly closes, 2010-01-08 → 2026-05-29, ~476 names/week,
~95 per quintile**. We measure the loser-minus-winner spread with a Newey-West HAC *t*,
then run the microstructure autopsy the horizon demands: **Killer #1**, a one-week *skip*
so the same close can't both form and price the trade; **Killer #2**, an **empirical
Corwin-Schultz (2012) bid-ask-bounce haircut** that charges each leg its own effective
spread; plus a flat cost sweep with short borrow, a beta decomposition, a random-portfolio
null, and a McLean-Pontiff sub-period decay split. A deterministic synthetic panel with
independent *reversal* and *bounce* knobs proves the machinery separates the two. **Dedup:**
[329-one-month-reversal](../329-one-month-reversal/) is the same idea on a **4×-slower
monthly clock**; [196-long-term-reversal](../196-long-term-reversal/) is the **multi-year**
De Bondt-Thaler reversal (different mechanism, no microstructure issue);
[538-industry-relative-reversal](../538-industry-relative-reversal/) fades the **industry-
relative** move. This is the **weekly** horizon with a **bid-ask-bounce haircut** as the
headline test. **Survivorship** named on the Signal axis. As-of **2026-05-29**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why last week's losers "bounce back," why a one-week wait makes the bounce vanish, and why the losers' own trading costs eat the edge |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC spread, the skip=0→skip=1 bid-ask-bounce killer, the empirical Corwin-Schultz haircut, the cost sweep + borrow, the beta decomposition, the decay split, and the two-knob synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`hf_reversal/`](hf_reversal/). Weekly total-return S&P 500 panel (current
membership projected backwards — **survivorship-biased upper bound**, named on the Signal
axis). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
