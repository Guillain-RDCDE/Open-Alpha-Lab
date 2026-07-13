# Study 726 — Chicken-Wing-Index 🍗

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the Super-Bowl window beat the rest of the year? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The run-up-window-minus-rest spread is **+5.10%/month, Welch t = 1.62**, block-bootstrap 95% CI **[−0.31%, +10.36%]** — straddles zero. January alone shows a naive **t = 2.52**, but it's one snooped month of twelve and **off-thesis November is stronger (t = 2.70)**. |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The long-January timer's higher Sharpe (**0.63** vs buy-and-hold **0.57**) is a de-risking illusion — it sits in T-bills 11 months and **forgoes more than half of WING's compound return** (9.0% vs **20.7%** CAGR). And a November-only timer beats it. |
| **Super-Bowl bump?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | **February — the actual game month — is *negative* (−2.01%)**. January's pop sits half on the ordinary January effect (WING-vs-SPY alpha t = 2.00 on 11 points, β ≈ 1), and the wing-price proxy is driven by avian-flu supply shocks, not a calendar. |

> **In one sentence:** the billion-wings-on-Sunday demand pulse is real, but as a trade it's a mirage — the Super-Bowl window on Wingstop is a coin-flip spread (t = 1.62) resting on a single data-snooped January that off-thesis November out-ranks, while February, the actual game, is *negative*.

## What we tested

The folklore, at full strength: Americans eat **~1.4 billion chicken wings** on Super Bowl Sunday
([National Chicken Council](https://www.nationalchickencouncil.org/)), so wing demand spikes into the
early-February game and **Wingstop (`WING`)** — the pure-play wing chain and the perennial "Super-Bowl
stock" — should rally into it. We test the strongest tradable version on **every calendar month of WING**
since its 2015 IPO (131 months, 11 Januaries, month-end from daily closes): per-month HAC t-stats; a
Welch **window-vs-rest** test on the January run-up; a block-bootstrap CI on the spread; a **12-month
placebo** (where does January rank?); a **January-alpha-vs-`SPY`** regression to net out the ordinary
January effect; and a long-January **calendar timer** vs buy-and-hold, gross and net. We also carry a
small **hardcoded, cited, approximate** wholesale-wing-price series — a *labelled proxy*, never a live
feed, and not itself tradable (no wing futures). (Same seasonality-mining shape as
[Study 307](../../307-coffee-seasonality/); same labelled-proxy discipline as [Study 358](../../358-watch-index/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a billion real wings makes a *terrible* calendar trade — the snooped month, the negative game month, the cash-drag illusion, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-month HAC t-stats, Bonferroni, window-vs-rest spread + block-bootstrap CI, the 12-month placebo, January-alpha-vs-SPY, the timer race, sub-period split |

The fingerprinted real-data run (WING + SPY + ^IRX, 2015–2026, fp `5742cb7dc999`) is in
[docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to
download); the offline machinery proof runs on the synthetic world in
[chicken_wing_index/data.py](chicken_wing_index/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). The wholesale-wing-price series is a
**hardcoded, cited, approximate proxy** — not a live feed and not tradable; WING is the tradable leg.
Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
