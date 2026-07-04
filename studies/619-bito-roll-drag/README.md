# Study 619 — BITO-Roll-Drag 🎢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the futures ETF pay a real toll vs spot? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | On the tape twice over: **−34.5 bps/month** vs spot over 56 months (HAC *t* = **−2.32**, 69.6% of months negative, **−20.2 pp** cumulative — a BITO dollar is worth **77 cents** of the spot dollar) and, on the clean matched-close ruler, **−5.14%/yr vs IBIT** (HAC(21) ***t* = −7.68**, lag-robust). The arithmetic closes: 0.70% fee gap + **~4.4%/yr carry ≈ the +4.9%/yr median annualized front basis**. No survivorship (single live instrument). |
| **Tradability** — can you harvest the toll? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Long IBIT / short BITO nets **+2.90%/yr (HAC *t* = +4.34, Sharpe 1.51)** at 2% borrow — real but thin, capacity-bounded, and **dead at ~5% borrow** (−0.10%/yr). The toll itself **flipped sign in the 2022 backwardation** (+3.6 pp that year): a contango phenomenon, not a constant. The risk-free fix — just hold the spot ETF — is cost avoidance, not alpha, and didn't exist before 2024. |
| **Any reason BITO survives the spot ETFs?** | ![Busted](https://img.shields.io/badge/Reason_to_survive%3F-Busted-8b949e?style=flat-square) | Same exposure (beta on IBIT = **1.003**) at **−5.14%/yr** (*t* = −7.68). The "monthly income" is your own capital handed back (the drag is measured *after* crediting every distribution); the 40-Act access and options rationales migrated to the spot ETFs by Nov-2024. What survives is habitat, not performance. |

> **In one sentence:** the futures-based bitcoin ETF really does pay a toll — **−5.1%/yr against the spot ETF at *t* = −7.7**, fee plus a carry that matches the measured CME front basis almost to the decimal — but the toll accrues **daily as the premium converges, not in the roll week** (Welch *t* = −0.3/−0.9), it flipped sign in the 2022 backwardation, and harvesting it short-vs-long survives only while borrow stays cheap — **Real, Fragile**, and since Jan-2024 there is no performance reason to hold BITO at all.

## What we tested

The desk claim: *"the futures-based bitcoin ETF pays a toll every month it rolls — measurable daylight between BITO and spot."* We race **total-return** BITO (its huge monthly distributions reinvested — price-only would wildly overstate the drag) against spot `BTC-USD` sampled on BITO's trading days since inception (Oct-2021), and against **IBIT** matched-close once the spot ETF listed (Jan-2024), with an IBIT-vs-spot control proving the crypto-midnight-vs-4pm timestamp offset adds noise but no bias. The daily gap gets a **Newey-West HAC *t*** (monthly HAC *t* vs spot where the timestamp noise washes out); the **roll-window attribution** (5 trading days into each CME last-Friday expiry) asks *where* the toll is paid — answer: nowhere in particular, the carry bleeds daily; the **contango split** (front basis from `BTC=F`, prior-day sign — the study's one lag) plus calendar-year basis levels tie the drag's sign to the curve (2022 backwardation → BITO *beat* spot), suggestive at Welch *t* ≈ 1, not certified. Tradability charges the long-IBIT/short-BITO spread borrow on the short and one-way costs on turnover across 0/2/5% borrow. A deterministic synthetic world with a **planted basis + fee** proves the detector recovers a known toll, invents none from basis noise — and exposed why daily basis-sign splits are artifact-prone. Distinct from the mechanical-decay siblings: [61-slow-burn](../61-slow-burn/) (leverage variance drag), [100-melting-ice](../100-melting-ice/) (commodity contango), [375-vxx-roll-decay](../375-vxx-roll-decay/) (VIX ETP) — new instrument, monthly CME roll, and a live spot-ETF ruler. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why an ETF that legally can't hold bitcoin buys next month's bitcoin at a premium, watches the premium melt, and does it again every month — and what that cost you vs just holding the spot ETF, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC inference on the daily/monthly gap, the matched-close IBIT race + timestamp-bias control, roll-window attribution, basis measurement and the contango split (with the artifact warning), borrow-sensitivity of the spread, and the planted-basis synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`bito_roll_drag/`](bito_roll_drag/). All legs total-return; the toll is measured after crediting BITO's distributions. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
