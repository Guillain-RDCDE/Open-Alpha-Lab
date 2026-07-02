# Study 590 — Sharpe-Hacking 🎛️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there any real edge behind the juiced Sharpe? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On a tape we *built* to have an honest Sharpe of **0.24** and **zero alpha**, return-smoothing (θ 0.5) inflates the *reported* Sharpe **0.32 → 0.56** (+73%) while the autocorrelation-corrected **honest** Sharpe moves **0.242 → 0.244** — nothing. Leverage (×3) changes *neither* Sharpe (0.323 / 0.242, exactly). Vol-targeting *lowers* the naive Sharpe to **0.16**. A synthetic-only method demo — no real tape, so it can never earn `REAL`. |
| **Tradability** — is there anything to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A smoothed Sharpe is an accounting illusion you cannot spend; leverage buys volatility and drawdown, not risk-adjusted return; vol-targeting is below baseline and loses net of a 2 bps cost (**0.145**). By construction there is nothing there. |
| **Can pure financial engineering fake a Sharpe?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | *Yes* — for the **reported** metric. Smoothing scales the naive Sharpe with the staleness (0.32 → 0.98, a 3.0× inflation, as θ → 0.8; lag-1 autocorr 0.01 → 0.81), and the bootstrap band on the *honest* inflation straddles zero (**[−0.012, +0.013]**). The correction sees through it, and the control proves it still tracks a *planted* edge. |

> **In one sentence:** you *can* juice a strategy's headline Sharpe with pure financial engineering — return-smoothing (stale illiquid marks) balloons the *reported* Sharpe by up to 3× with zero added return, while the folk levers everyone reaches for (leverage, vol-targeting) do literally nothing or actively hurt — but every bit of the "gain" evaporates the moment you use the autocorrelation-corrected Sharpe that can't be gamed, which is exactly why a smooth, high-Sharpe track record is a *warning*, not a boast.

## What we tested

The most persuasive number in asset management — a high, smooth Sharpe ratio — is also the easiest to
manufacture. We take a synthetic daily return stream with a modest **honest Sharpe (≈ 0.24) and no
genuine edge**, and run three pieces of pure financial engineering over it while watching two
numbers: the **naive** Sharpe everyone quotes (mean/std × √252) and the **honest**,
autocorrelation-corrected Sharpe (Lo 2002; Getmansky-Lo-Makarov 2004) that *cannot be gamed by
smoothing*. **Return smoothing** (reporting AR(1)-stale, illiquid marks) is the big fake — it slashes
measured volatility and injects autocorrelation, inflating the reported Sharpe with zero added
return. **Naive leverage** provably changes *neither* Sharpe (mean and std scale together). **Vol
targeting** is no free lunch and here loses net of costs. A circular-block bootstrap shows the
honest-inflation band straddles zero while the naive one is firmly positive; a **seed-robust synthetic
positive control** (25 seeds) proves the honest Sharpe *tracks a genuinely planted edge* — so the
correction isolates real skill from measurement games. *A pure method demo on a synthetic world by
design — cousin of [344 Backtest-Overfitting](../344-backtest-overfitting/) and
[589 Genetic-Algo-Overfit](../589-genetic-algo-overfit/), which inflate a Sharpe by **searching**;
590 inflates it by **transforming the reported returns**, corrected by the autocorrelation adjustment
rather than a trial count.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Sharpe ratio is, how a smooth illiquid track record fakes a great one, why leverage does nothing, and why "vol-targeting" isn't magic — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the naive vs autocorrelation-corrected (Lo/GLM) Sharpe, the AR(1) smoothing model, the θ-inflation sweep, the leverage invariance, the vol-targeting costs, the block-bootstrap bands, and the seed-robust synthetic positive control |

The fingerprinted headline run (null tape fp `807d5cddc33f`, as-of 2026-06-30) is in
[docs/results.md](docs/results.md); the whole machinery runs offline and deterministic on the
synthetic world in [`sharpe_hacking/data.py`](sharpe_hacking/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`sharpe_hacking/`](sharpe_hacking/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
