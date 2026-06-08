# Study 04 — Social-Oracle 🔮

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | 🔴 `NONE` | A mention has **no abnormal edge over a random day**: excess +0.08% / +0.05% / **−0.66%** at 1d / 1wk / 1mo (p≈0.23 / 0.40 / 0.94 — i.e. *negative* by a month), and the clustering bootstrap straddles zero at every horizon. Not one lucky name (jackknife is flat). |
| **Tradability** — does it survive costs, capacity, scale? | 🔴 `MIRAGE` | Gross is **pure beta**: +0.72%/trade but only **+5 bps abnormal**; the **median trade is −1.3%**, net hits zero at a 25 bps spread and goes negative beyond, and the equal-weight sleeve runs **−44% with an −84% drawdown** (the 2022 pile-in). |
| **Pump-and-fade?** — does the pop reverse? | ⚪ `CONFIRMED` | The month-ahead abnormal return is significantly *negative* vs a random day, the share of up-names falls to **45.7%** (vs 51.4% random), and a mention does **worse than a name that was simply already hot** (−1.06% at 1mo). The follower buys the bleed. |

> **In one sentence:** on 1,468 real WallStreetBets viral surges, buying what the
> crowd screams carries **no abnormal edge** — a tiny, insignificant one-day flicker
> that fades to a *negative* month, gross "gains" that are just market beta the costs
> erase, a median trade of −1.3%, and 42 of the most-viral names that literally
> delisted. It's a pump you're late to, dressed as a signal.

## What we tested

A retail-investing folk hero goes viral — here, *白毛股神* **Serenity**
([@aleabitoreddit](https://twitter.com)) — and a wave of open-source repos now scrape
her timeline, distil it into `$SYMBOL`s, and score them 0–100 as tradeable "signals"
([`haskaomni/serenity-signal-dashboard`](https://github.com) and friends). We steelman
the strongest version: *a public mention is, on average, followed by a positive
abnormal return you could have captured.* Since the single-guru feed lives behind X's
auth wall, we test the **same phenomenon on the crowd** — a purer, fully reproducible
instance built from daily r/WallStreetBets mention counts.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story + the stakes, plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full method: abnormal-return event study, the two nulls, the fade, costs |

Real run on **1,468** WSB surges (CC-BY `youyanggu/yolostocks-data`): tables in [docs/results_wsb.md](docs/results_wsb.md), feed provenance in [_data/PROVENANCE.md](_data/PROVENANCE.md), reproduce via [examples/verify_wsb.py](examples/verify_wsb.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
