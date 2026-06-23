# Study 385 — Jobless-Claims-Momentum 📰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do rising claims precede weaker returns? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The *direction* is real — when claims momentum is rising, forward SPY returns are lower at every horizon (12-month **+10.4%** vs base **+12.0%**, down-rate **24%** vs **18%**) — but the excess **fails t ≥ 2** (best Welch *t* = **−0.94**, placebo *p* = **0.10**), is fragile to the window, and **flips positive** for the biggest upticks. Real-as-lore, weak-as-edge. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The "cash when claims rise" overlay **underperforms buy-and-hold** gross *and* net (**+7.0%** vs **+11.4%**/yr, Sharpe **0.72** vs **0.77**). Acting on the signal *destroys* return — there's nothing to allocate to. |
| **Early warning?** | ![Not_supported](https://img.shields.io/badge/Early_warning%3F-Not_supported-8b949e?style=flat-square) | The lead/lag scan puts the strongest *negative* correlation at **L = −3** — claims momentum *lags* the market by a quarter; at positive leads it's ~zero. Claims are a **coincident-to-lagging echo**, not a leader. The "early" part is exactly what the data rejects. |

> **In one sentence:** rising jobless-claims momentum really does precede modestly weaker equity returns, but the tilt is statistically insignificant (Welch *t* = −0.94), the claims uptick actually *lags* the market rather than leading it (peak negative correlation three months *behind* the move), and a "sell when claims rise" rule loses to buy-and-hold — so the famous "claims call downturns early" reads as a coincident recession echo dressed up as a crystal ball.

## What we tested

The macro-nowcasting folklore says initial jobless claims are a *leading* indicator, so when the 4-week-MA of claims turns **up** an equity downturn is on the way — an early-warning you can get defensive on. We rebuild that signal on the monthly claims tape: claims momentum is **rising** when the 4-week-MA level is above its value three months prior, and we measure forward 1/3/6/12-month SPY returns in rising-claims months vs the unconditional base rate, with a one-month execution lag, a Welch *t*, a placebo null, an explicit **lead/lag** scan (does the uptick actually come *first*?), and a tradable cash-on-rising-claims overlay. (FRED is firewalled in this build, so the claims series is a hardcoded, never-revised monthly snapshot of `IC4WSA` — public and frozen, caveated on the Signal axis; the COVID-2020 spike is included faithfully.) A deterministic synthetic control with a *planted* claims→returns link confirms the engine recovers a real edge and can't manufacture one from noise.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "claims lead the market" is mostly the market leading *claims*, what a claims uptick really tells you, and why selling on it loses money — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | claims-momentum split returns, a Welch *t* + placebo null, the decisive lead/lag cross-correlation, the timing overlay vs buy-and-hold, robustness (window / threshold / ex-COVID), and a synthetic planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`jobless_claims_momentum/`](jobless_claims_momentum/). Claims here are a hardcoded **snapshot** of FRED `IC4WSA` (the settled print, not the real-time vintage), named as such. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
