# Study 512 -- High-Volume-Return-Premium

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Do stocks that just traded on unusually high volume keep climbing -- and quiet ones lag?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The high-vol-minus-low-vol book earns **-5.47%/yr** (one-sample *t* = **-1.34**, HAC -1.33) -- the *wrong* sign for the premium -- and the label-shuffle placebo gives **p = 0.12** (seed-robust). The sort is indistinguishable from random labels; the GKM premium does not replicate on modern large-caps. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross Sharpe **-0.39**, net **-11.17%/yr** after 5 bps/leg + 50 bps borrow, **~65%/week** turnover. Nothing to trade even if the sign had been right. |
| **Survivorship-biased?** | ![Named](https://img.shields.io/badge/Named-8b949e?style=flat-square) | 40 current large-caps projected backwards. The blow-up / takeover names that trade on the biggest volume spikes are absent -- results are **upper bounds**, and the bound is already negative. |

> **In one sentence:** Gervais-Kaniel-Mingelgrin's high-volume return premium -- buy the names that just traded heavily, sell the quiet ones -- is real on old, broad NYSE tapes, but on a 12-year modern mega-cap survivor basket the long-short is the wrong sign, statistically a coin (placebo p=0.12), and badly negative after costs: None signal, Mirage tradability.

## What we tested

Gervais, Kaniel & Mingelgrin (2001): each Friday, rank the cross-section by **abnormal volume**
(this week's average daily volume ÷ its own trailing 8-week mean, minus 1); go long the
high-volume top quintile, short the low-volume bottom quintile, equal-weight and dollar-neutral;
hold the **next** week's return (one execution lag -- no same-bar fill, no look-ahead). Panel:
40 large-cap survivors, yfinance daily adjusted close + raw volume, 2014--2025 (622 weekly
observations). We charge 5 bps/leg + 50 bps short borrow, run a within-week label-shuffle placebo
(200 draws, seed-robust), sweep the holding horizon (1/2/4/8 weeks), and verify the engine with a
deterministic synthetic positive control. Universe is survivorship-biased -- named, treated as an
upper bound.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the attention/visibility story in plain language, the synthetic positive control, the real-tape long-short, the placebo, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | abnormal-volume construction, quintile legs, HAC inference, label-shuffle null distribution, horizon sweep, cost & turnover drag, equity curve |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`high_volume_return_premium/`](high_volume_return_premium/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
