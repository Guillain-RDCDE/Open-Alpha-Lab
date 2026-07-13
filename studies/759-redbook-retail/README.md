# Study 759 — Redbook-Retail 🛒

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do accelerating same-store sales precede stronger retail returns? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The tilt is right-signed only at 3–6 months and never significant — best case **6-month +9.1%** vs base **+6.3%**, Welch **t = +1.21** (placebo *p* = 0.08) — a dead tie at 1 month, near-nothing at 12, fragile to every spec, and **market beta** (retail-vs-market relative excess *t* = **+0.79** at 6m, *negative* at 1m/12m). Indistinguishable from noise. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | "Own retail when Redbook accelerates" **underperforms buy-and-hold** — **+8.2%** vs **+12.1%**/yr, $1 → **$4.01** vs **$6.06** over 20 years. It buys volatility reduction (Sharpe a near-tie, 0.50 vs 0.49), not reward. Acting on it *destroys* return. |
| **Leads retail?** | ![Not_supported](https://img.shields.io/badge/Leads_retail%3F-Not_supported-8b949e?style=flat-square) | The lead/lag scan peaks *positive* at **L = −3** — Redbook *lags* XRT by a quarter (ρ = **+0.33**); at positive leads ρ ≈ 0. A **coincident-to-lagging echo**, not a leader — and the one *significant* level-regime result (*t* = **−2.28**) points the **wrong way** (strong nominal 2021–22 sales preceded *weaker* returns). |

> **In one sentence:** accelerating Redbook same-store sales are followed by *slightly* better retail returns at one horizon, but the tilt is statistically absent (best Welch *t* = +1.21), it's really just market beta, the Redbook uptick actually *lags* retail stocks by a quarter rather than leading them, and a "buy retail when sales accelerate" rule loses to buy-and-hold — so the famous weekly consumer nowcast reads as a nominal, inflation-tangled echo of a sector the market already reprices in real time.

## What we tested

The consumer-nowcasting folklore says the weekly **Johnson Redbook Index** of same-store retail sales is the most timely read on the shopper, so when its year-over-year growth **accelerates** the retail sector (XRT) is about to climb — a nowcast you can tilt into. We rebuild that signal on the monthly Redbook YoY tape: Redbook is **accelerating** when same-store growth is above its value three months prior, and we measure forward 1/3/6/12-month **XRT** returns in accelerating vs decelerating months against the unconditional base rate, with a one-month execution lag, a Welch *t*, a placebo null, an explicit **lead/lag** scan (does the uptick actually come *first*?), a **retail-vs-market** (XRT−SPY) relative test, a level-regime split, and a tradable own-on-acceleration overlay. (The weekly Redbook series is proprietary and off FRED, so it's a hardcoded, clearly-labelled **approximate reconstruction** of the YoY same-store number — faithful in shape, approximate in level, caveated on the Signal axis; the 2021–22 double-digit *nominal* surge is included faithfully.) A deterministic synthetic control with a *planted* Redbook→returns link confirms the engine recovers a real edge and can't manufacture one from noise.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "the sales pulse leads retail" is mostly the market leading *sales*, why a nominal number is dangerous to trade on, and why buying on it loses money — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Redbook-acceleration split returns, a Welch *t* + placebo null, the decisive lead/lag cross-correlation, the retail-vs-market relative test, the wrong-way level regime, the timing overlay vs buy-and-hold, robustness, and a synthetic planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`redbook_retail/`](redbook_retail/). Redbook here is a hardcoded, clearly-labelled **approximate proxy** of the proprietary same-store series (faithful in shape, approximate in level), named as such. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
