# Study 720 — Super-Bowl-Advertiser 📺

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do Super Bowl advertisers drift up after the game? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The post-game "drift" is **+0.4%** vs the market (Welch *t* = **0.31**, placebo *p* = **0.64**) on a **47%** win-rate (*below* a coin flip), and the day-one Monday reaction is actually **−0.33%** (*t* = −0.83). Fehle-Tsyplakov-Zdorovtsov (2005) found it on a 2000–04 sample; on this modern ~32-event tape it's gone. **Survivorship** named here: the loudest advertisers (Pets.com & co.) went bust and left the tape. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Buying the ad basket every February earns **+0.39% gross / +0.19% net** over five days — a rounding error inside its own error bar, indistinguishable from a week of large-cap beta. Nothing to size. |
| **"Big-ad signal"?** | ![Not supported](https://img.shields.io/badge/Big--ad_signal%3F-Not_supported-8b949e?style=flat-square) | A **much-cited 2005 paper + a memorable sock puppet** keep the idea alive: Pets.com is remembered *because* it was extreme, and it **delisted**. Across a representative survivor table the drift is a coin-flip and the day-one move is negative. |

> **In one sentence:** the legend that a company buying a ~$7-million Super Bowl commercial gets a burst of attention and drifts up in the days after the game rests on a real 2005 finance paper (2000–04 sample) and a famous sock puppet — but across a transparent table of ~32 real listed advertisers (2015–2024) the post-game drift is a faint **+0.4%** (Welch *t* = 0.31, placebo *p* = 0.64) on a sub-50% win-rate, the Monday reaction is *negative*, and the ad-calendar basket earns **+0.19% net** — so it's real-as-a-2005-paper, absent-as-a-modern-edge, and untradable at any size.

## What we tested

We hardcode a **transparent table of ~32 real *listed* Super Bowl advertisers** as advertiser-year events (2015–2024) — Wix, Coinbase, Temu, e.l.f., Bud Light, GM, Rocket, Salesforce, Ulta — and run a textbook short-window **event study**: the **abnormal (excess-of-SPY) return** on a short **drift** leg (`[+1…+5d]`, the "big-ad signal") and a longer **hold** leg (`[+6…+25d]`) after the game, plus the single-day **Monday reaction**, with a one-day entry lag (you act Monday, the Sunday-night ad already public). The steelman is a peer-reviewed paper — [Fehle, Tsyplakov & Zdorovtsov (2005)](docs/references.md) found significantly positive abnormal returns after the game — so we test whether that 2000–04 result still breathes. We judge the drift with a Welch *t*, a placebo null sized to the event count, and the cost of the basket; we name the central honesty problem loudly: the loudest advertisers that **went bust** (Pets.com, Computer.com, Kozmo.com) delisted and leave no series, biasing the survivor tape *toward* the story. A deterministic synthetic control with an injected drift confirms the engine is faithful **and** that ~32 events can't reach significance unless the planted effect is implausibly large.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why we only remember the sock puppet, what an "abnormal return" is, and why the survivors didn't drift up after all — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | abnormal-return event windows on an advertiser table, the drift leg vs zero, a Welch *t* + placebo randomization null, the costed ad-calendar basket, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`super_bowl_advertiser/`](super_bowl_advertiser/). The advertiser table is hardcoded & transparent; the priced tape is **survivor-biased** (the loudest advertisers went bust / private), named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
