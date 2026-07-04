# Study 628 — Buffett's Alpha 🎩

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is Buffett's 40-year alpha statistically real? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Full-sample CAPM alpha **+9.46%/yr** at **HAC *t* = 3.47** (1980-04 → 2026-06, 555 months) at beta **0.70**; on the FKP-era window (→ 2011) alpha **+12.36%/yr**, *t* = **3.40** — the paper's *t* > 3 replicates on free public data. Carries an explicit **selection-on-success** caveat: Berkshire is on the bench *because* it won. |
| **Tradability** — can you harvest it today? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Buying BRK-B is trivially cheap and unlimited-capacity — but the alpha **faded −10.98 pp/yr after 2010** (HAC *t* = −2.38 on the difference), **no 10-year window since 2002** clears *t* ≥ 2, and today's holder owns beta 0.75 with statistically-zero alpha. Real, no longer harvestable — not INVESTABLE. |
| **"Any alpha left in the last 15 years?"** | ![Busted](https://img.shields.io/badge/Any_alpha_left%3F-Busted-8b949e?style=flat-square) | 2011-07 → 2026-06: alpha **+2.66%/yr** at *t* = **0.89**, Sharpe **0.74 vs the market's 0.90**, and $1 in BRK grew to **$6.45 vs $7.37** in the index. Berkshire has been a lagging min-vol fund (USMV loading **+0.89**, *t* = 3.77), exactly as FKP's fade predicted. |

> **In one sentence:** the single most famous alpha in the world is *real* — $1 → $2,880 vs the
> market's $220 since 1980, CAPM alpha +9.46%/yr at HAC *t* = 3.47 on beta 0.70 — but it lives
> almost entirely in the 1980s (+23.4%/yr, *t* = 3.55), has faded significantly since (−11 pp/yr
> post-2010, *t* = −2.38), and the last 15 years show zero alpha and an index-lagging dollar —
> Frazzini-Kabiller-Pedersen confirmed on both halves: the alpha was real, and it's gone.

## What we tested

We audit Frazzini, Kabiller & Pedersen's *Buffett's Alpha* (FAJ 2018) on free public data:
BRK-A monthly total returns (yfinance; Berkshire pays no dividend, so price = total return)
against an honestly-spliced US-market total return (^GSPC + Shiller dividend yield to 1993-01,
SPY thereafter — labeled per row), both in excess of the previous-month-end 13-week T-bill.
The Signal axis is the full-sample excess-vs-excess CAPM alpha with Newey-West HAC *t*; the
decay story runs through a per-decade table, a 436-window rolling 10-year alpha curve, and a
post-2010 dummy regression putting a HAC *t* on the alpha *change* itself. A factor-lite
regression on investable proxies (QUAL + USMV, ETF era only, honestly labeled — not academic
QMJ/BAB) checks FKP's quality/low-beta mechanism. A 20-seed synthetic control with a planted
alpha proves the machinery. Selection-on-success is named on the Signal axis. As-of
**2026-06-30**, fingerprint `96d18b13a5da`.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "alpha" means, why Buffett's is the one legend the tape certifies, where it lived (the 1980s), and why buying Berkshire today buys the past — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC CAPM full-sample + FKP-era replication, decade table, rolling 10-y alpha, the fade *t*-test on the difference, QUAL/USMV factor-lite loadings, and the 20-seed planted-alpha control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`buffetts_alpha/`](buffetts_alpha/). Siblings: the factors FKP explain Buffett with —
[238-betting-against-beta](../238-betting-against-beta/), [242-quality-minus-junk](../242-quality-minus-junk/) —
plus [264-buffett-indicator](../264-buffett-indicator/) (same name, different claim) and
[627-thirteen-f-cloning](../627-thirteen-f-cloning/) (copying disclosures). **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
