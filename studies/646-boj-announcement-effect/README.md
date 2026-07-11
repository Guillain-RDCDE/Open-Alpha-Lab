# Study 646 — BoJ Announcement Effect 🇯🇵🏦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do Japanese equities/the yen move systematically on BoJ decision days? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Across **247** BoJ decisions 2005→2026: EWJ decision-day return **+0.008%** vs +0.015% other days (Welch *t* = **−0.06**), yen **−0.063%** vs −0.006% (Welch *t* = **−0.85**) — both hit rates sit *below* 50% (47.8%, 47.4%), both random-calendar placebos are unremarkable (*p* = 0.92, 0.16), no offset in the [−5..+3] event window clears the bar, and the NIRP/YCC "surprise-regime" era split doesn't rescue it (diff *t* = +0.13 / +0.75). |
| **Tradability** — can you harvest it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | With no average edge, one round trip of costs turns both legs decisively negative: EWJ **−9.2 / −19.2 bps net** at 5/10 bps, yen **−16.3 / −26.3 bps net**, and both carry an **8%-scale worst-day tail** (2011 earthquake session, 2008/2020 crisis clusters). |
| **"BoJ days swing MORE than an average day?"** | ![Confirmed](https://img.shields.io/badge/Bigger_swings%3F-Confirmed-8b949e?style=flat-square) | Realized range is genuinely elevated: EWJ (H−L)/open **1.164%** vs 0.995% (Welch *t* = **+2.49**), yen **1.059%** vs 0.760% (Welch *t* = **+6.30**). Decision days are louder — the amplitude claim is true even though the direction claim is not. |

> **In one sentence:** across 247 BoJ decisions since 2005 there is no systematic directional
> reaction in EWJ (Welch *t* = −0.06) or the yen (*t* = −0.85) — not even inside the NIRP/YCC
> "surprise regime" — so there is nothing to trade net of costs (Mirage); but decision days are
> genuinely **louder** (realized-range *t* = +2.49 / +6.30), and the honest anatomy of that
> loudness is a handful of huge, sign-flipping tail events (the Dec-2022 "Kuroda shock" alone
> moved the yen +3.8%, *z* = +3.69) that cancel on average rather than a repeatable calendar
> edge.

## What we tested

We hardcode **248 BoJ Monetary Policy Meeting decision/statement dates 2005-01-19 → 2026-06-16**
(Bank of Japan official archives; **every** decision on record, scheduled and inter-meeting
alike — unlike the FOMC sibling, this claim is explicitly about the surprise-driven eras) and
split daily EWJ (unhedged Japan equities, USD) and yen returns (minus the USDJPY change, same
sign convention as [615-yen-safe-haven](../615-yen-safe-haven/)) into decision days vs all
5,157 other trading days: Welch *t*, a Newey-West dummy-regression *t*, a Wilson-bounded hit
rate and a two-sided 20-seed × 1,000-draw random-calendar placebo. An event window [−5..+3]
finds no build-up and no persistence on either instrument. A **named data quirk** — yfinance's
`JPY=X` daily `Close` silently duplicates `Open` on >95% of 2023-2025 rows — is fixed by dating
the yen return `Open[D+1]/Open[D] − 1` uniformly across the whole sample (documented in
[docs/results.md](docs/results.md)), because the broken field would otherwise mute the very
tail events (Dec-2022 "Kuroda shock") this study exists to measure. The grey third axis asks a
different, honest question — **is the day just louder, even if the sign is unpredictable?** —
and the answer is yes (realized range *t* = +2.49 EWJ / +6.30 yen). Tradability charges the
honest bill: hold the decision day only, enter at the prior close (the BoJ's yearly schedule is
public — zero look-ahead), exit at the decision close, net of one-way costs × 2. A 20-seed
synthetic null plus a planted-shift world proves the machinery. **Dedup:**
[615-yen-safe-haven](../615-yen-safe-haven/) (the yen's *general* risk-off reaction to *any*
equity-crash day, not a BoJ-specific calendar), [645-ecb-announcement-effect](../645-ecb-announcement-effect/)
(the same question for the ECB) and [637-fomc-vol-crush](../637-fomc-vol-crush/) (the Fed's
decision-day effect on *implied vol*, scheduled meetings only) never test what **EWJ/the yen
do specifically on a BoJ decision day** across the full NIRP/YCC surprise history — this study
does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "the BoJ moves markets" sounds obvious, the seven surprises everyone remembers, why the average day is still a coin flip, and what "louder but not directional" actually means |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch/HAC splits, the two-sided placebo, the event-window anatomy, the era contrast, the realized-range myth-check, the named `JPY=X` data quirk, the tail-event z-scores, the capture test with costs, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`boj_announcement_effect/`](boj_announcement_effect/). EWJ is unhedged (USD)
Japan-equity exposure, price-only unless noted; yen return = **minus** the `JPY=X` change,
labeled everywhere; no survivorship on the Signal axis (a fund and an FX rate, not a
survivor-conditioned basket). **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
