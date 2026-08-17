# Study 930 — When-Issued Window

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

When a company splits in two, is the freshly cut-loose half a bargain in its first days?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | One of the five pre-specified legs clears the bar, and with the **sign inverted**: the child under-performs SPY by **−4.54%** gross over its first **5** regular-way sessions (*t* = −2.73, HAC *t* = −3.02, bootstrap CI [−7.67, −1.42]% clear of zero, 8/26 positive; −4.77% net of a long position's frictions, which is why the *gross* number is the headline). Negative in **both** eras, versus **IWM and MDY** too, jackknife *t* ∈ [−3.36, −2.44], de-clustered *t* = −2.88. The parent run-up (−1.45%, *t* = −0.26) and the sum-of-the-parts pop (+1.88%, *t* = +1.28) are **null**. **It survives narrowly:** the Westfall-Young family-wise *p* across all five legs is **0.045**. *Survivorship named:* 26 curated liquid spins, plus four lost because the child was acquired and delisted. |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | To bank a negative alpha you must **short a five-day-old spin-off**: +4.32%/trade gross of borrow (*t* = +2.60), +3.33% at 50%/yr borrow, but **+2.33% with a CI through zero** at 100%/yr — and a fresh child is the textbook name where borrow is rationed or simply absent. ~2 trades a year, 8.5 pp per-trade dispersion, and a window so narrow that starting three sessions late turns **+4.3% into −2.2%**. |

> **In one sentence:** Greenblatt's forced-*seller* story predicts the spun-off child is dumped and cheap in its first days — the tape says the opposite, that the child slides for three sessions after it starts trading regular way and only claws most of it back by the end of the month, which looks far more like index funds forced to *buy* it on distribution day than like anyone forced to sell.

## What we tested

A hardcoded table of **26 liquid US spin-offs (2012–2025)**, cut into three windows: the
**parent** from the day after the separation was announced to the last close before the
child trades regular way; the **child** over its first **5 / 10 / 21** regular-way sessions;
and the **parent + child** combination held 21 sessions. Every leg is an alpha versus SPY
(long asset, short index, **1:1 in dollars — no beta is estimated**; cash-neutral, so
excess-of-cash *is* the spread), total-return closes, 10 bps one-way × NAV plus borrow on
the short leg, one execution lag. The announcement date, the regular-way date and the
distribution ratio are **assumptions** compiled from the public record and are each swept.
**Dedup:** distinct from **[239-spinoffs](../239-spinoffs/)**, which buys the child and
holds it for **6–24 months** (the Cusatis-Greenblatt long drift) on a *different* 14-event
table, and from **[452-spinning-top](../452-spinning-top/)**, which shares only the word.
Study 930 looks at the opposite end of the same clock — the wait before the distribution
and the first month after it — and adds the parent and combined legs that 239 never measures.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a when-issued window is, why everyone expects a bargain, the three-day slide the tape actually shows, and why you probably still cannot trade it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the three-leg panel, HAC *t* and bootstrap CIs, the family-wise max-\|*t*\| test across all five legs, era and benchmark cuts, the jackknife, the three proxy sweeps, the short-side borrow economics, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`when_issued/`](when_issued/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
