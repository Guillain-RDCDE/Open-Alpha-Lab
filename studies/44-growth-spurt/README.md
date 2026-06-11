# Study 44 — Growth-Spurt 📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do low-asset-growth firms beat high-growth ones? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No measurable premium on tradable large caps: over 15 look-ahead-free July→June windows the hedge earns **−3.4%/yr** (*t* = −1.23, **95% CI [−9.3%, +2.5%]**) — indistinguishable from zero, and a modest premium can't be ruled out on 15 annual observations. |
| **Tradability** — can you trade the highest headline Sharpe on the list? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No. The vendor's **0.835** doesn't replicate where you can hold names at scale; the effect hides in micro-caps and is subsumed by the Fama-French investment factor. |
| **"Replicates on large caps"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | Not measurable on current large-cap **survivors** — and the panel is biased *against* the effect (censoring delisted high-growth blow-ups props up the short leg), so the negative point estimate is uninformative. Not a disproof; a failure to support. |

> **In one sentence:** the juiciest number on the vendor list (Sharpe 0.835) is a mirage for anyone trading liquid stocks — on large-cap survivors the asset-growth hedge is a statistical zero, on a panel whose survivorship leans *against* the effect — because the premium, to the extent it's real, lives in micro-caps and is already captured by a standard factor, i.e. exactly the illiquid, survivorship-fragile corner where headline Sharpes are manufactured.

## What we tested

The **asset-growth effect** (Cooper, Gulen & Schill 2008): firms that grow their total assets fast subsequently underperform. It carries the **highest headline Sharpe (0.835) on [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading)'s open list** — the most tempting thing in the backlog. We rebuild it on **real balance-sheet data** — fiscal-year total assets from **SEC EDGAR** for ~399 current S&P 500 members, monthly total returns from Yahoo — forming portfolios on **June 30 of y+1** from fiscal-year-y 10-Ks *already filed by then* (the Cooper-Gulen-Schill convention, EDGAR `filed`-enforced, so no look-ahead), long the slow growers, short the fast ones, July→June windows 2010–2025. The one question that matters: **does the famous Sharpe survive on names you could actually trade?** We flag the caveats openly — the universe is large-cap and survivorship-biased, and that bias runs *against* the strategy (it flatters the short leg). The offline control is a synthetic firm panel with a tunable growth penalty (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the best-looking number on the list can't be found on real, tradable stocks, and where the effect actually hides |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the look-ahead-free long-short construction, the CI that swallows the headline, the micro-cap / investment-factor / survivorship-direction explanations |

The fingerprinted real-data run (~399 S&P 500 names, July→June windows 2010–2025, as-of 2026-06-01, fp `8dec74717e92`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` crawls SEC EDGAR — slow); the offline machinery proof runs on the synthetic panel in [growth_spurt/data.py](growth_spurt/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
