# Study 649 — Gold-Seasonality 🪙📅

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is September really gold's best month? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | September's monthly return is **+0.03%** — a rounding error — vs **+0.88%** for the other 11 months (**wrong sign**, Welch *t* = **−0.64**), bootstrap CI **[−3.23%, +1.60%]**. Zero of 12 calendar months clear the Bonferroni bar in September's favor; the one that does (January, HAC *t* = +3.77) is a different, untested claim. The summer "lull" points the right way but isn't significant (*t* = −1.24), and September was never certified in either the pre- or post-2013 era. |
| **Tradability** — can you harvest the calendar? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | "Own gold only in September" gives up almost the entire return: **+1.3–1.5%/yr net vs +8.6%/yr buy-and-hold**, negative excess-of-cash Sharpe (−0.02 to −0.05 vs **+0.47**), hit rate **42.9%** (Wilson [24.5%, 63.5%]) — below a coin flip. |
| **Gold's "best month"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | September isn't statistically distinct from average, isn't even the sample's actual best month (January is — a different claim), and a strategy built to catch it loses to simply holding gold. |

> **In one sentence:** the "September is gold's best month, summer is a lull" story sounds
> plausible — it even has a real physical-demand mechanism (Indian wedding-season and pre-Diwali
> buying) — but on 21+ years of GLD, September is statistically indistinguishable from an average
> month (Welch *t* = −0.64, wrong sign), no month clears Bonferroni in its favor, and owning gold
> only in September forfeits almost the entire buy-and-hold return: **None, and a Mirage.**

## What we tested

The claim: gold has a calendar — **September strength** (Indian wedding season + pre-Diwali
physical demand, year-end jeweller restocking) and a **summer lull** (the quiet stretch between
spring Akshaya Tritiya buying and the autumn wedding season). We test it on **GLD** (SPDR Gold
Shares, 2004→2026), the tradable, physically-backed spot-gold proxy anyone can actually buy: a
12-cell month-of-year table (naive + Newey-West *t*, **Bonferroni**-corrected for 12 simultaneous
tests), a Welch *t* on September vs the other 11 months with a circular block-bootstrap CI, a
Welch *t* on summer vs the rest, a pre/post-**2013** era contrast (the 2013-04-12/15 gold crash,
an externally-dated, justified split), and an "own gold only in September" calendar timer raced
excess-of-cash against buy-and-hold, gross and net of costs. A 20-seed synthetic positive control
with a tunable planted September/summer effect proves the machinery is unbiased (never cited for
the stamp). **Dedup:** [289-diwali-muhurat](../289-diwali-muhurat/) (the same festival, but the
Indian *equity* Muhurat session, not gold's monthly calendar), [69-safe-haven](../69-safe-haven/)
(gold's inflation/crash-hedge behavior, no calendar), [580-gold-lease-rate](../580-gold-lease-rate/)
(a borrow-cost lead-lag, synthetic-only), [640-gold-overnight](../640-gold-overnight/) (the daily
overnight/intraday clock, not month-of-year), [305-gold-oil-ratio](../305-gold-oil-ratio/) and
[113-gold-silver-ratio](../113-gold-silver-ratio/) (cross-asset ratio signals, no calendar axis)
never test gold's own **monthly** seasonality — this study's own axis. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "gold's best month" sounds so convincing, what the calendar actually shows, and why owning gold only in September is a worse trade than doing nothing |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the 12-cell Bonferroni table, the Welch/HAC splits, the block-bootstrap CI, the 2013 era contrast, the excess-of-cash timer race, and the 20-seed synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`gold_seasonality/`](gold_seasonality/). GLD is a single, continuously-listed
physically-backed ETF (no survivorship panel, no futures roll). **Not investment advice** —
research & education. See [LICENSE](../../LICENSE).*
