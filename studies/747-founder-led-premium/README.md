# Study 747 — Founder-Led-Premium 🧑‍💼

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do founder-led firms earn a real abnormal return? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The founder basket beat professional-CEO peers by a huge **+20.5%/yr** — but **67%** of that spread is plain **market beta** (a high-beta tech long vs a low-beta staples short). The residual market-model alpha is **+52 bps/mo at Newey-West *t* = 0.76** — zero. Placebo *p* = **0.33**; drop **NVDA** alone and the alpha halves. **Survivorship** named here, and it points *for* the claim. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The basket is **hindsight-selected** — the founders who *turned into* Nvidia, picked because we know that now. Two originals (**SQ, FIT**) delisted outright. Net **+128 bps/mo** is mostly free beta plus a one-name coin-flip; you could never have formed it ex-ante. |
| **"Founder premium?"** | ![Misattributed](https://img.shields.io/badge/Founder_premium%3F-Misattributed-8b949e?style=flat-square) | The outperformance is **real** and **mis-explained**: survivorship (we only remember the winners) plus tech-sector beta concentrated in ~2 names — not a leadership characteristic you can screen for. |

> **In one sentence:** a basket of the founder-led firms we remember in 2024 — Amazon, Nvidia, Meta, Tesla — thrashed a basket of professionally-run blue chips by ~20 points a year, but **two-thirds of that is just market beta**, the leftover "founder alpha" is a statistically-zero **+52 bps/mo (*t* = 0.76)** that **halves when you drop Nvidia**, and a random relabelling of the same names matches it a third of the time — so the premium is real, un-tradable, and entirely **misattributed** to founders rather than to survivorship and sector beta.

## What we tested

The founder thesis is well-cited: **[Fahlenbrach (2009)](docs/references.md)** reports S&P 500 founder-CEO firms earn positive abnormal returns, and Bain's *"Founder's Mentality"* turned it into management and VC folklore. We take the strongest form — a hardcoded **founder-led** basket (long) vs a matched **professional-CEO** basket (short), equal-weighted, rebalanced monthly over 108 months (2016–2024, yfinance total-return proxy) — and compute the spread's **market-model abnormal return** (Jensen alpha, Newey-West HAC *t*), then attack it with a beta/alpha decomposition, a leave-one-out jackknife, and a label-shuffle placebo. There's no free founder-CEO database, so the basket is a transparent, **hindsight/survivor-biased** stand-in — named loudly on the Signal axis, where the bias points *for* the claim, so an insignificant alpha is a conservative refutation. A deterministic synthetic control confirms the engine separates alpha from beta and won't fabricate a premium.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the flattering chart, why "they won" isn't "founders win", the beta hiding inside, and the hindsight trap — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | CAPM alpha with a Newey-West HAC *t*, the beta/alpha decomposition, a leave-one-out jackknife, a label-shuffle placebo, costs + borrow, and a synthetic plant-and-recover control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`founder_led_premium/`](founder_led_premium/). The baskets are an explicit **hardcoded, hindsight-labelled** stand-in; the priced tape is **survivor-biased** (the founder flame-outs delisted; two of our own names dropped out), named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
