# Study 569 — SBC-Dilution 🧾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do lean, low-SBC firms beat the heavy diluters? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On a **31-name large-cap survivor** basket over **9** annual rebalances (formation 2016 → 2024, one execution lag), the long-short (long low-dilution, short high-SBC) spread is **−15.2%/yr at *t* = −2.01** — the **wrong sign** for the hidden-cost anomaly, and if anything significant *against* it. A **label-shuffle placebo** beats the real sort **99.8%** of the time; the win-rate is **22%** (2/9). Wrong sign in each leg (SBC-only −41%, dilution-only −8%) and at every quantile width (*t* −1.9 to −2.0). The faithful 25-seed synthetic control shows a *real* edge would print a clearly **positive** *t*, so this is a genuine **inversion on this tape**, not a hidden sign-flip. **Survivorship** + a **shallow SBC snapshot** named on this axis. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A spread that points the wrong way and sits near *t* −2 is not a book — and the short leg you'd hold is exactly the expensive-to-borrow mega-cap tech (NVDA, META, AMZN, MSFT, CRM). One-way **10 bps × 2 legs + 50 bps** short borrow only deepens it (net **−15.9%/yr**). Nothing to size. |
| **Hidden-cost anomaly replicates here?** | ![Busted](https://img.shields.io/badge/Replicates_here%3F-Busted-8b949e?style=flat-square) | A canonical accounting factor (SBC / dilution under-pricing, a cousin of Pontiff-Woodgate issuance) replicated honestly on a small survivor basket over an AI-driven melt-up lands where the desk expects **most** such replications to: the sign reverses (the heavy-SBC growth mega-caps *led*), *t* never clears +2, placebo unbeaten. The effect that lives in the full survivorship-free cross-section **does not survive** a 31-name survivor sort. |

> **In one sentence:** the stock-based-comp / dilution anomaly — *the companies handing out the most
> equity quietly dilute you into worse returns* — is real and documented in the full US
> cross-section, but replicated honestly on **31 large-cap survivors** over **9** years with a
> 1-year lag, split-adjusted shares and a shallow SBC snapshot, the long-lean / short-diluters
> spread comes out **−15.2%/yr at *t* = −2.01** — the *wrong sign*, with a label-shuffle placebo
> that beats it 998 times in 1000 and a faithful synthetic control confirming a genuine edge would
> have printed a positive *t* — because the AI melt-up rewarded exactly the heavy-SBC growth names
> the anomaly says should lose, and a survivor basket has already deleted every firm that diluted
> itself to death. **None × Mirage**, replication **Busted**.

## What we tested

A point-in-time, survivorship-free SBC/issuance universe is a Compustat product, not a free
yfinance feed, so we fix a **transparent 31-name large-cap survivor basket** and, for each
formation year-end, build a **dilution score** = `z(SBC / revenue) + z(share-count growth)`, each
z-scored across the basket. The SBC leg is `Stock Based Compensation` / `Total Revenue` from
yfinance's cash-flow + income statements (a **shallow ~4-year snapshot**, named on the Signal
axis); the dilution leg is year-over-year growth in **split-adjusted** shares outstanding (so a
4-for-1 split isn't 300% dilution). We go **long the bottom 30%** (lean / low-SBC), **short the top
30%** (heavy-SBC / fast-diluting), and hold the **next** year (one execution lag — the score is
public at year-end *t*, we trade *t→t+1*). Inference is a one-sample *t* against zero plus a
**label-shuffle placebo null**, with one-way costs × turnover + a short-leg borrow, a
quantile-width sweep, an each-leg-alone cut, and a deterministic **synthetic control averaged over
25 seeds** confirming the engine recovers a planted edge (with the *right* sign) and refuses to
manufacture significance at the null. This is the **SBC-flavoured dilution factor** — distinct from
[Study 519 — Net-Share-Issuance](../519-net-share-issuance/) (the pure share-count factor) and
[Study 368 — Buyback-Drift](../368-buyback-drift/) (discrete *announcements*).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "the companies printing the most stock quietly dilute you" is a real Wall-Street idea that *inverts* on a few dozen big survivors — the AI melt-up, the wrong sign, and why 9 years can't settle it, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the annual cross-sectional sort, long-short vs zero, a one-sample *t* + label-shuffle placebo null, quantile-width & each-leg robustness, costs + borrow, and a 25-seed synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`sbc_dilution/`](sbc_dilution/). The basket is an explicit **large-cap survivor sample** with a **shallow SBC snapshot**, not a point-in-time universe (both named on the Signal axis). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
