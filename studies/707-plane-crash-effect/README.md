# Study 707 — Plane-Crash-Effect ✈️📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the market dip when a major air crash hits the news? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Crash-day abnormal SPY return **+0.41 bps**, one-sample *t* = **+0.02**, hit rate **47.2%** (Wilson [32.0%, 63.0%]) — a coin flip. Random-calendar placebo **p = 0.51** over 20,000 draws. The airline basket (AAL/DAL/UAL/LUV) shows no extra drop either (*t* = +1.11, wrong sign). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | "Buy the dip" after a crash **underperforms** simply holding SPY at every horizon tested (1/3/5/10 days), gross and net of costs — worst case *t* = −0.91 at 10 bps costs. There is no edge to charge costs against in the first place. |
| **Does the market flinch at a plane crash?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | 36 major disasters, 2000→2025, a proper event study, a random-calendar placebo and a paired airline-extra-drop test all agree: no detectable market-wide dread dip, no airline-sector overreaction, on the modern tradable tape. |

> **In one sentence:** across 36 of the most famous commercial aviation disasters since 2000, the S&P 500 moves by a statistically indistinguishable-from-zero **+0.41 bps** on the news (*t* = 0.02, placebo *p* = 0.51), airline stocks (AAL/DAL/UAL/LUV) don't fall any harder than the market, and "buying the dip" afterward loses to just holding the index — the Kaplanski-Levy "aviation-disaster sentiment" effect does not survive on today's tradable instruments, at least not at a detectable size given 36 events.

## What we tested

Kaplanski & Levy (2010, *JFQA*) found that major aviation disasters trigger a small,
sentiment-driven market-wide dip — the "fear/dread" mood shock from a vivid tragedy,
not a fundamentals story — that fades within days; the tradable corollary is that
airline stocks, with genuine exposure, should drop *harder*. We steelman it on a
hand-curated table of **36 major commercial-aviation disasters, 2000→2025** (9/11 and
MH17 deliberately excluded — both already sit in
[313-geopolitical-shock](../313-geopolitical-shock/)'s shock table as terror/war
events), run an event study on SPY around each crash's first tradable session
(abnormal returns, CAR path [−1..+5], a random-calendar placebo), pair each event's
SPY move against a 4-carrier airline basket's move to isolate a sector-specific extra
drop, and put a "buy the dip" timer through a cost sweep — stated with the obvious
ethical caveat, kept entirely clinical: this is a statistical test of public
market-price data, not a recommendation to trade on tragedy. A deterministic
synthetic tape with a *planted* crash-day dip is the positive control. **As-of
2026-06-30.**

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a plane crash *should* move markets if the "fear index" story is right, what the tape actually shows, and why the intuitive "airlines drop harder" idea doesn't hold up |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the event-study anatomy, the random-calendar placebo, the paired airline-extra-drop test, the look-elsewhere caveat on the one nominally significant offset, the costed buy-the-dip timer, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`plane_crash_effect/`](plane_crash_effect/). The disaster calendar is
hand-curated from public aviation-safety reporting; SPY and AAL/DAL/UAL/LUV are
fetched via yfinance, no survivorship on the market-wide axis, airline-basket
coverage is named honestly (full 4-carrier only from 2008 onward). **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
