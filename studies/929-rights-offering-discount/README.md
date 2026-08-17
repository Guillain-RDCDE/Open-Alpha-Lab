# Study 929 — Rights Offering 🎟️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The announcement window is **+0.27% (*t* = +0.34)** and the post-expiry window **+0.21% (*t* = +0.25)** — the discount is not compensated, and deep-discount deals do no better (label-permutation *p* = 0.75). The one nominally significant number, the −2.06% subscription drift (*t* = −2.03), sits in the window contaminated by **mechanical ex-rights dilution**; it fails the fair, era-matched placebo (*z* = −1.60, *p* = **0.09**), the leave-one-issuer-out jackknife (*t* → −1.14), the era cut (*t* → −1.05), the assumed timetable and the ±10-day anchor jitter. Sample is **survivor-only** (five deals have no tape), the event list is **month-precision** — a look-ahead channel that could only *help* find an effect — and 39 deals come from 20 heavily overlapping issuers. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Buying the discount earns an excess-of-cash Sharpe of **+0.098** over the full window (beta-adjusted alpha **+0.03%/yr**, HAC *t* +0.01). Hold only the clean, ex-rights-free span and the Sharpe rises to **+0.448** — but that is rented equity beta: alpha **+2.98%/yr, HAC *t* +1.79**, still under the bar, still behind SPY's **+0.575**, in a book of 1.1 illiquid closed-end funds live an eighth of the time. The mirror-image short — credited its cash collateral, so the costs are not one-sided — is negative at **every** borrow rate 0–800 bps on both spans. |

> **In one sentence:** A rights offering's deep subscription discount is neither a gift nor a warning — it is a *pro rata* bookkeeping choice, and on 39 US deals the tape shows no announcement reaction, no post-expiry compensation, no gradient in the depth of the discount, and nothing tradable in either direction.

## What we tested

Thirty-nine US rights offerings by twenty listed issuers (mostly closed-end funds —
Cornerstone, Gabelli, Oxford Lane — plus two BDCs and a REIT), 2013–2023. Market-model
abnormal returns against **SPY** (alpha/beta on the `(−250, −31)` window) cumulated over
an **announcement**, a **subscription** and a **post-expiry** window; a deep-vs-shallow
discount split with a label permutation; three placebos (whole-tape, era-matched,
clustered); and a costed calendar-time portfolio raced **excess-of-cash on both legs**,
long and short, over the full and the ex-rights-free holding span, with one execution
lag. The event list, the rights timetable and the discount bands are
**PROXIES / ASSUMPTIONS** and are each swept.
**Dedup:** distinct from **563-secondary-offering-drift** (underwritten follow-ons, no
pro-rata right), **519-net-share-issuance** and **790-composite-equity-issuance** (the
slow annual issuance factor, not a dated deal), **367-closed-end-fund-discount** and
**910-managed-distribution-cef** (a fund's standing discount and payout policy, not an
issuance event), **927-dutch-auction-buyback** (the exact mirror: a pro-rata tender to
*retire* shares at a premium) and **569-sbc-dilution** (continuous, non-pro-rata dilution).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a discount you are *given* cannot make you richer, what the tape says, and the one number that looked real until we tried to break it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the market-model CARs, the three placebos (and why the flattering one is not the one to quote), issuer jackknife, permutation test, anchor jitter, timetable sweep, the costed long/short book with its beta-adjusted alpha, and the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`rights_offering/`](rights_offering/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
