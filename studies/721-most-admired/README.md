# Study 721 — Most-Admired ⭐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does making Fortune's list predict returns? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The honest, no-look-ahead admired book *does* clear the bar (**+7.05%/yr**, HAC **t = 2.53**, market-model alpha **t = 2.00**) — but it's **significant-raw, fragile-to-selection**: drop **Apple & Nvidia** and the alpha collapses to **+1.8%/yr (t = 0.88)**, **~half** is the generic equal-weight-large-cap tilt a *random* basket also earns (placebo **+3.2%/yr**), and the roster is the winners that **stayed** admired. **Look-ahead + survivorship** named here. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The harvestable-looking book (**t = 4.9**) is **look-ahead selection**; costs are trivial (**0.02%/yr**), so the mirage is the hindsight, not the frictions. By the time Fortune crowns a firm, the run that earned the crown is already priced. |
| **Premium, or beta + hindsight?** | ![Misattributed](https://img.shields.io/badge/Premium%3F-Misattributed-8b949e?style=flat-square) | The out-performance is a levered (**beta 1.12**) large-cap/tech tilt concentrated in **Apple & Nvidia** — not a reward for *admiration*. And the contrarian reversal doesn't show either (admired *beat* spurned). |

> **In one sentence:** owning Fortune's Most Admired mega-caps *did* beat the market — but the "admiration premium" is two hindsight winners (Apple, Nvidia), half a generic equal-weight-large-cap tilt, and a roster hand-picked from the survivors that stayed on top; strip the look-ahead and the beta and there is no admiration edge to buy prospectively.

## What we tested

We steelman the strongest version of the claim — *the best-run firms compound faster, so
owning Fortune's [World's Most Admired Companies](https://fortune.com/ranking/worlds-most-admired-companies/)
beats the index* (the academic **premium** of [Antunovich–Laster–Mishra, 2000](docs/references.md)),
and its contrarian mirror, the [Statman–Fisher–Anginer](docs/references.md) **reversal**. We
hardcode a transparent, cited table of the survey's 15 perennial All-Stars, build an
equal-weight admired book against the market (`SPY`), fit a market-model **alpha**, and judge
it with a **Newey–West (HAC)** *t* — under a strict publication lag (own a name only *after*
it is crowned) and against a random-large-cap placebo. The central honesty problem is named
loudly and on the Signal axis: a *current* most-admired list is **look-ahead + survivorship**,
so raw out-performance is selection and factor beta before it is "admiration."

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "great company" and "great stock" aren't the same, how a survey that crowns yesterday's winners fakes a premium, and the Apple-and-Nvidia tell — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the admired book's excess-over-SPY, a HAC *t* + market-model alpha, the leave-two-out robustness, a random-large-cap placebo, the survivor-biased long/short, and a synthetic power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`most_admired/`](most_admired/). The admired table is hardcoded & cited; the priced tape is **look-ahead + survivor-selected** (the roster is the winners that stayed admired; the spurned proxy is survivors of a delisted cohort), named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
