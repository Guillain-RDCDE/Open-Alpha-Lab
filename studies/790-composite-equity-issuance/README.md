# Study 790 — Composite Equity Issuance 🧾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do low/negative 5y composite issuers (buyback-ers) beat high issuers (diluters)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On a **36-name large-cap survivor** basket over **11** annual rebalances, the long-short (long low-issuance, short high-issuance) spread is **−13.22%/yr at one-sample *t* = −2.29** (Newey-West *t* = **−2.69**) — the **wrong sign** for the Daniel-Titman factor, and *significantly* so: high composite issuers **outperformed**. A **label-shuffle placebo** beats the real sort **99.7%** of the time; the win-rate is **36%** (Wilson [15%, 65%]). The faithful synthetic control recovers a *real* low-issuance edge at a clearly **positive** *t* = +7.8, so this is a **survivorship-driven sign flip**, not noise the tape can't resolve. **Survivorship** named on this axis, its direction (against the claim) argued. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A wrong-sign spread is no book long the buyback-ers, and one-way **10 bps × 2 legs + 50 bps** short borrow only deepens it (net **−13.92%/yr**, *t* = −2.41). The *inverse* bet is a **pre-2020-only** artifact (early *t* = −10.2, late *t* = −0.45) — nothing to size in either direction. |
| **Does the published factor replicate here?** | ![Busted](https://img.shields.io/badge/Replicates_here%3F-Busted-8b949e?style=flat-square) | A canonical academic factor, honestly rebuilt on a small survivor basket with **point-in-time filed shares**, one execution lag and real costs, lands where the desk expects most of them to: the sign flips, the placebo goes unbeaten, the "edge" is a small-sample / survivorship illustration. The effect that lives in the full CRSP/Compustat cross-section **does not survive** a 36-name survivor replication. |

> **In one sentence:** composite equity issuance — *serial issuers underperform, net buyback-ers
> outperform* — is real in the **full** US cross-section (Daniel-Titman 2006), but rebuilt
> honestly on **36 large-cap survivors** over **11** years with point-in-time EDGAR shares, a
> 1-year lag and real costs, the long-short comes out **−13.22%/yr at *t* = −2.29** — the *wrong
> sign*, with a label-shuffle placebo that beats it 997 times in 1,000 and a faithful synthetic
> control confirming a genuine edge would have printed a **positive** *t* — so on this tape it is
> **None × Mirage**, a textbook **survivorship** inversion rather than a tradable edge.

## What we tested

A point-in-time, survivorship-free issuance universe is a CRSP/Compustat product, not a free
feed, so we fix a **transparent 36-name large-cap survivor basket** and compute each firm's
**5-year composite equity issuance** (Daniel-Titman 2006): `ι = log(ME_t/ME_{t−5}) − r(t−5,t)`,
the part of 5-year log market-cap growth **not** explained by the stock's own cumulative
total return — net equity issuance in log terms. `ME = shares × raw price`, where **shares** are
SEC EDGAR `CommonStockSharesOutstanding` (`dei:EntityCommonStockSharesOutstanding` fallback)
taken **point-in-time** by their `filed` date (no look-ahead into an unfiled 10-K), and the
return leg is yfinance adjusted closes; splits cancel between the legs. Each formation year-end
we sort the cross-section, go **long the bottom 30%** (low/negative issuance), **short the top
30%** (high issuance), and hold the **next** year (one execution lag). Inference is a one-sample
*t* + a Newey-West HAC *t* (overlapping 5y windows) + a **20,000-draw label-shuffle placebo**,
with one-way costs × both legs + short borrow, a quantile-width sweep and an era split. A
deterministic **25-seed synthetic control** confirms the engine recovers a planted low-issuance
edge with the **right** sign. This is the **5-year log composite** measure — distinct from
[Study 519 — Net-Share-Issuance](../519-net-share-issuance/) (the *1-year raw share-count*
change), [Study 368 — Buyback-Drift](../368-buyback-drift/) (the announcement *event*) and
[Study 250 — Reverse-Split](../250-reverse-split/) (a corporate-action signal). As-of
**2025-12-31**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "companies that print stock lose to companies that buy it back" *flips sign* on a basket of big survivors — the two baskets, the year-by-year ladder, and how survivorship (not a broken detector) inverts a real factor, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the 5-year Daniel-Titman measure on point-in-time EDGAR shares, the annual cross-sectional sort, one-sample + Newey-West *t*, a 20,000-draw label-shuffle placebo, quantile-width & era cuts, costs + borrow, and a 25-seed synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`composite_issuance/`](composite_issuance/). The basket is an explicit **large-cap survivor sample**, not a point-in-time universe (survivorship named on the Signal axis, its direction argued against the claim). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
