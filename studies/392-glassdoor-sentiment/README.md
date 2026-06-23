# Study 392 — Glassdoor-Sentiment 🙂

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do the happiest workforces out-earn? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The academic premium is real *in the literature* (Edmans 2011), but real employer-review data isn't free, so we can only test the claim on a **constructed, return-independent sentiment proxy** — on which the happiest-minus-grumpiest long-short is **+3.3%/yr at t = 1.16** (placebo *p* = **0.16**, Sharpe **0.25** with a CI **through zero**). It **fails t ≥ 2**. Proxy/synthetic-only support ⇒ **Weak**, never Real. (Survivorship + proxy both named on this axis.) |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | A static-score quintile long-short trades rarely, so a **70 bps/yr** all-in cost only trims Sharpe 0.25 → 0.20. But a long-short whose **gross Sharpe is 0.25 with a CI straddling zero** is not a NAV-scale strategy. Costs aren't the killer — there is no edge to deploy. |
| **Free lunch?** | ![Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square) | "Buy the happiest workplaces and beat the market" is, on a return-independent proxy, a **coin-flip** (54% monthly win-rate; a random relabelling of who's "happy" matches it ~1 time in 6). The alt-data romance does the work the data can't. |

> **In one sentence:** the "happiest employees beat the market" pitch descends from a real academic premium (Edmans 2011), but employer-review data is paywalled, so on a transparent constructed satisfaction proxy the happiest-minus-grumpiest long-short returns +3.3%/yr at t = 1.16 with a Sharpe CI through zero — a faithful engine finding an honest null, which a proxy can never upgrade to a Real stamp.

## What we tested

Real Glassdoor / employer-review ratings sit behind paid APIs, so we **construct a transparent, clearly-labelled employee-satisfaction proxy**: each name in a fixed **40-name** large-cap basket gets a fixed-seed 1.0–5.0 "stars" score, assigned **independently of returns** (it is *not* scraped data, and we say so throughout). We then run the textbook factor sort — go long the happiest quintile, short the grumpiest, equal-weight, one-period entry lag — and measure the monthly long-short spread over **21.5 years** (2005–2026, **257** months) with a one-sample *t*, a bootstrap Sharpe CI, a sign-flip placebo null, and one-way costs plus short borrow. A deterministic synthetic control with an *injected* happiness→return edge confirms the engine recovers a real edge when one exists — so the real-tape result is an **honest null**, not a broken detector.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "happy companies beat the market" is a real *paper* but an unproven *trade*, what a happiness long-short is, and why a proxy can't settle the question — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the quintile sort, the long-short with a one-sample *t* + bootstrap Sharpe CI + sign-flip placebo null, costs, and a synthetic faithful-engine / planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`glassdoor_sentiment/`](glassdoor_sentiment/). The satisfaction score here is an explicit **constructed proxy** (a fixed-seed placeholder), not real Glassdoor data; prices are public adjusted closes. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
