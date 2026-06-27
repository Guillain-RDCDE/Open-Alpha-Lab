# Study 515 — Earnings-Announcement Premium 📅

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there an announcement premium? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On the tradable **`[D−1, D+1]`** window the premium is **+0.53 bps/day** (Welch *t* = **0.16**, per-event *t* = **−0.11**, random-calendar placebo *p* = **0.39**) — indistinguishable from zero, and announcement days carry only **0.50×** their share of return. The one honest nuance is a faint **day-of** bump (+5.5 bps, *t* = **0.90**, 1.45× concentration) with the right Frazzini–Lamont *shape* — but it never clears **t ≥ 2** and **vanishes** at any tradable window. A **powered** synthetic control (*t* ≈ 7.5 on a planted premium) proves the detector would have seen a real effect. **Survivorship** is named, and here it biases *toward* a premium — we still find none. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The long-the-announcers overlay **underperforms buy & hold at every cost level** (net +3.0 bps/day at 5 bps vs B&H +5.8; **negative** at 10 bps) and is in-market only ~5% of the year — you sacrifice the other 95% of returns to chase a premium that isn't there. Nothing to deploy. |
| **"Most of the return is earned around earnings"?** | ![Busted](https://img.shields.io/badge/Earnings_days_carry_it%3F-Busted-8b949e?style=flat-square) | Announcement-window days carry **2.38%** of cumulative return on **4.79%** of days — *half* their proportional share, the exact **opposite** of the claim. Only the single announcement day shows a >1× concentration (**1.45×**), and even that is a sub-*t*-2 whisper that the surrounding days erase. |

> **In one sentence:** the famous Frazzini–Lamont earnings-announcement premium has the right *shape* on the announcement day itself (a stock earns ~double its normal return on the day it reports, 1.45× return concentration) but it is a sub-significance whisper (Welch *t* = 0.90) that completely **dissolves** the moment you widen to a tradable `±1`-day window (+0.53 bps/day at *t* = 0.16) — on a liquid large-cap survivor basket the premium is **gone**, announcement days carry *less* than their share of return, and a long-the-announcers calendar loses to buy & hold at every cost level.

## What we tested

We replicate the **Frazzini–Lamont (2007) / Beaver (1968)** announcement premium as a clean
calendar study on a fixed **30-name large-cap basket**: for every trading day we tag whether it
sits inside an announcement window `[D−1, D+1]` around any scheduled earnings date, then compare
the mean daily return on announcement days against the rest — pooled (Welch *t*) and per-event
with a within-name baseline (one-sample *t*). We add a **share-of-return** decomposition (do the
~5% of announcement days carry more than 5% of the return?), a **random-calendar placebo** (20,000
re-taggings of the same density), a **window-width robustness** sweep (`D±0` / `±1` / `±2`), and
a tradable **long-the-announcers overlay** charged one round trip per window. A deterministic,
**powered** synthetic control plants a known announcement premium and confirms the engine lights
up at *t* ≈ 7 — so the real-tape null is a genuine absence, not a blunt detector. This is the
*level* premium on the announcement days themselves and is **distinct from PEAD**
([../363-pead-drift](../363-pead-drift)), which is the signed-surprise *drift*. Survivorship (the
basket is names still trading in 2026) is named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the "earnings premium" is, why a stock might pay you for reporting, why the day-of bump looks real but the tradable window is flat, and why holding only-while-reporting loses to just owning the stock — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | announce-vs-rest per-day premium, within-name per-event one-sample *t*, a random-calendar placebo, the share-of-return decomposition, the window-width sweep, the overlay net of costs, and a powered synthetic faithful-engine control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`earnings_premium/`](earnings_premium/). The tag is the announcement window `[D−1, D+1]`
around each scheduled `Ticker.get_earnings_dates` print; the premium is announce-day return minus
the within-name baseline. Basket is **survivors** — named on the Signal axis. **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
