# Study 563 — Secondary-Offering-Drift 💧

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — after a company sells a fresh slug of shares, does the stock keep sliding? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The claim predicts a **negative** drift (dilution + a bearish issuance signal). On **34** notable follow-on / secondary offerings the abnormal (stock − SPY) drift is **positive at every horizon** — **+10.6 / +7.8 / +8.2 / +29.4%** at 1 / 3 / 6 / 12 months — the **wrong sign**, never clearing *t* ≥ 2 (best **+1.78**), and the same-names placebo puts the real set on the *high* side of the random-date distribution (left-tail *p* 0.53–0.99). Not an edge, and not even the predicted direction. Survivorship + visibility selection named on this axis. |
| **Tradability** — can you short the issuer? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The tradable trade — **short the issuer** — is the wrong sign *before* costs: short gross **−8% to −11%** at 1/3/6m, net **−9.8%** at 6m after a 10 bps round-trip + a punitive **300 bps/yr borrow** (these are hard-to-borrow high-fliers). Nothing to harvest. |
| **"Dilution drift on the tape?"** | ![Busted](https://img.shields.io/badge/Dilution_drift%3F-Busted-8b949e?style=flat-square) | On this basket the issuers *out*-ran the market for a year afterward — the drift is **inverted**. The genuine effect (the negative *announcement-day* reaction, and the net-issuance factor on full universes — see [519](../../519-net-share-issuance/)) does not survive a hand-picked roster of headline growth-story raisers (Tesla, crypto miners, pandemic names). |

> **In one sentence:** the dilution folklore — sell fresh shares and the stock keeps sliding — is real in the long-run new-issues literature and as an *announcement-day* pop, but across **34** notable follow-on/secondary offerings the *subsequent* abnormal drift is **positive at every horizon** (+10.6% at one month, +29% at a year), never clears *t* = 2, and a same-names placebo can't reject luck — because the recognisable, survived issuers we can name are growth-story high-fliers that raised cash at strength and *kept ripping*, so the tradable "short the issuer" trade is the wrong sign before you even pay the borrow.

## What we tested

Clean, point-in-time SEC-424B feeds of seasoned offerings aren't free on yfinance, so we hard-code a
**transparent table of 34 notable follow-on / secondary equity offerings** (ticker, public pricing
date, headline $bn size) and measure the **abnormal** drift — the event stock's forward return
**minus SPY** over 1 / 3 / 6 / 12 months, so a market move can't masquerade as offering drift. We
enter **one day after** the pricing, so we measure the *drift* the claim is about, not the
documented (negative) announcement-day jump. Inference is a one-sample Welch *t* against zero plus a
**same-names left-tail placebo null** (re-enter each name on random dates, ask how often chance is
*as negative*), with a size-split robustness cut and a **short-the-issuer** cost model (one-way cost
+ 300 bps borrow). A deterministic synthetic control with an *injected negative* drift confirms the
engine is faithful — flat at the null, past *t* = −2 only for an implausibly large planted slide —
**and** that ~34 single-name events can't reach significance for any drift of plausible magnitude.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a secondary offering is, why dilution "should" push the stock down, and why on our basket the issuers actually *rose* — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | abnormal-return event study, forward drift vs zero, a one-sample *t* + same-names left-tail placebo null, a size-split cut, short-the-issuer costs + borrow, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (34
offerings, panel fp `75dc79931e2c`, as-of 2026-06-30): [docs/results.md](docs/results.md).

---

*Engine: [`secondary_offering_drift/`](secondary_offering_drift/). The event set is an explicit
**hand-curated sample** of notable offerings, not a point-in-time universe (survivorship + visibility
named on the Signal axis). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
