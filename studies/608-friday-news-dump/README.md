# Study 608 — Friday-News-Dump 🗑️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does Friday-filed bad news drift more (DellaVigna-Pollet)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On 1,027 negative 8-Ks (Item 4.02 non-reliance, 2.06 impairments, 5.02 CEO exits, 2004-2026) the Friday-vs-Mon-Thu drift gap CAR[+1..+10] is **+10.7 bps at Welch *t* = +0.08** (winsorized +0.02, label-permutation **p = 0.523**); all six class/period robustness cells sit below \|*t*\| = 1.4 with signs both ways, and even the pooled bad-news drift (−54 bps/10d) reads *t* = −1.12. **Survivorship named**: only 39.5% of filers map to a live ticker, so drift *levels* are understated — but the Friday *gap* the claim needs is absent on the visible tape. |
| **Tradability** — short the Friday dump? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Enter at the day-0 close (one documented lag), cover +10, SPY-hedged: **+3.8 bps/event net (*t* = +0.03)** at ~10 trades/yr in hard-to-borrow small caps (10 bps/side + 5%/yr borrow); the Friday-after-close variant is net **negative**. Nothing to deploy. |
| **"Firms dump bad news Friday night"?** | ![Mixed](https://img.shields.io/badge/Friday_night_dump%3F-Mixed-8b949e?style=flat-square) | The dump is real but it is an **after-hours** dump, not a Friday one: on the full survivorship-free panel (3,122 bad vs 696 earnings filings) **57.9% of bad 8-Ks land after 16:00** (z = +4.94) and Friday-after-close runs **12.8% vs 1.4%** for earnings (z = +15.15) — yet the weekday mix of bad news is flat (Fri 21.3% vs a 20.3% uniform null, **z = +1.40**). Friday only looks like trash day because *routine* news avoids it (7.9%). |

> **In one sentence:** companies genuinely wait for the closing bell to file restatements, impairments and CEO exits — but they do not particularly wait for Friday, and the market does not stay fooled either way: Friday-filed bad news shows **zero** extra drift (gap *t* = +0.08, placebo p = 0.52), so the DellaVigna-Pollet trade on 8-K bad news is a **None / Mirage**, while the "hiding" folklore survives only as an *after-hours* effect.

## What we tested

We rebuilt the disclosure-timing claim from the SEC's own logs: a stratified panel of **3,122 unambiguously-negative 8-K filings** (Item 4.02 non-reliance, Item 2.06 material impairments, Item 5.02 CEO resignations — the classes created by the 2004-08-23 expanded 8-K regime) harvested from the EDGAR full-text search API with a documented, weekday-orthogonal cap (first 12 eligible pure-8-K hits per class × quarter), each carrying its EDGAR **acceptance timestamp to the second**, plus 696 Item 2.02 earnings 8-Ks as the timing control. Day 0 is the first session whose close reflects the filing (Friday-PM filings land Monday); abnormal returns are daily stock-minus-SPY with a ≥$1 prior-close screen. The Signal axis is the **Welch *t* on the Friday-vs-Mon-Thu drift gap** CAR[+1..+10], guarded by a seeded 2,000-draw label-permutation placebo and 1%-winsorization; the third axis tests the *hiding* margin (Friday / after-close / Friday-PM shares vs the earnings control **and** vs a uniform-calendar null) on the full, survivorship-free panel. Tradability charges 10 bps/side + borrow on a short-the-dump overlay. A deterministic synthetic control with a *planted* Friday drift and a *planted* hiding propensity proves the machinery. Survivorship (39.5% ticker-mapped) is named on the Signal axis. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a "Friday news dump" is, the weekday chart that shows bad news is flat across the week while earnings desert Friday, why the after-hours dump is real, and why the market doesn't stay fooled — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch gap test with permutation placebo and winsorized robustness, class/period slices, the two-proportion timing tests vs both nulls, the cost stack on the short overlay, and the planted-effect synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`friday_news_dump/`](friday_news_dump/). Siblings: [565](../565-filing-readability/) / [566](../566-earnings-call-tone/) / [567](../567-uncertainty-word-count/) are text-**content** studies — this one is the **timing** of disclosure with the news class fixed bad; [90-weekend](../90-weekend/) is the unconditional calendar effect. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
