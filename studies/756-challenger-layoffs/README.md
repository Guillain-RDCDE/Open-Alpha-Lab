# Study 756 — Challenger-Layoffs 🪓

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a job-cut spike precede weaker returns? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The *direction* is real — after a Challenger cut-spike, forward SPY returns are lower at every horizon, sharpest in the **announcement-drift month** (**+0.25%** vs base **+0.78%**, down-rate **44%** vs **36%**) — but the excess **fails t ≥ 2** everywhere (best HAC *t* = **−1.66**, placebo *p* = **0.08**), is fragile to the window, and **flips positive** for the biggest spikes. Real-as-lore, weak-as-edge. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The "cash when cuts spike" overlay **underperforms buy-and-hold** gross *and* net (**+5.3%** vs **+9.3%**/yr, Sharpe **0.51** vs **0.61**). Acting on the signal *destroys* return — there's nothing to allocate to. |
| **Early warning?** | ![Not_supported](https://img.shields.io/badge/Early_warning%3F-Not_supported-8b949e?style=flat-square) | The lead/lag scan puts the strongest *negative* correlation at **L = −2** — the cut spike *lags* the market by two months; at positive leads it's ~zero. Announced cuts are a **coincident-to-lagging** echo, not a leader. The "early" part is exactly what the data rejects. |

> **In one sentence:** a spike in Challenger's monthly announced-job-cuts really does precede modestly weaker equity returns — most visibly in the month right after the print — but the tilt never clears significance (best HAC *t* = −1.66), the cut spike actually *lags* the market by two months rather than leading it, and a "sell when cuts spike" rule loses to buy-and-hold — so the famous "layoffs warn you early" reads as a coincident recession echo dressed up as a crystal ball.

## What we tested

The macro-nowcasting folklore says the **Challenger, Gray & Christmas** monthly Job Cuts
Report — the widely-cited tally of layoffs *announced* by U.S. employers, out before the BLS
jobs number — is a *leading* indicator, so when announced cuts **spike** an equity downturn
is on the way (get defensive). We rebuild that signal on the monthly Challenger tape: cuts
are **spiking** when the month runs above its trailing-12-month average, and we measure
forward 1/3/6/12-month SPY returns in spike months vs the unconditional base rate, with a
strict **one-month release lag** (you only act *after* the report is public), a Welch *t*, a
**Newey-West HAC *t***, a placebo null, an explicit **lead/lag** scan (does the spike come
*first*?), and a tradable cash-on-spike overlay. (Challenger's series is proprietary with no
free feed, so the job-cut tape is a hardcoded, clearly-labelled **approximate proxy** of the
published monthly headlines — the COVID-2020 record spike included faithfully.) A
deterministic synthetic control with a *planted* spike→returns link confirms the engine
recovers a real edge and can't manufacture one from noise.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "layoffs warn the market" is mostly the market leading *layoffs*, what a cut spike really tells you, and why selling on it loses money — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | spike-conditioned split returns, a Welch *t* + Newey-West HAC *t* + placebo null, the decisive lead/lag cross-correlation, the timing overlay vs buy-and-hold, robustness (window / threshold / ex-COVID), and a synthetic planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`challenger_layoffs/`](challenger_layoffs/). Job cuts here are a hardcoded, labelled **approximate proxy** of the Challenger, Gray & Christmas monthly headlines (proprietary series, no free feed), named as such on the Signal axis. Prices are yfinance **total-return** (auto-adjusted) SPY closes. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
