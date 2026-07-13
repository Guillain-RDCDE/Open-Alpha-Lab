# Study 762 — Vegas-Gaming-Win 🎲

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does rising Strip GGR momentum lead the casino stocks? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No bullish tilt at any horizon — the rising-GGR forward mean sits *below* the base rate (12-month **+21.8%** vs **+25.1%**), best Welch *t* = **−0.72** (wrong sign, placebo *p* = 0.81), and the biggest accelerations are emphatically **contrarian** (*t* = **−4.08** at > +3%). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The own-when-rising overlay **ties** buy-and-hold on Sharpe (**0.50 vs 0.49**) while forfeiting **~10 pts/yr** of return (**+13.4%** vs **+23.0%**). It removes a third of the beta — less exposure, not more skill. |
| **Leading signal?** | ![Leading_signal%3F: Not_supported](https://img.shields.io/badge/Leading_signal%3F-Not_supported-8b949e?style=flat-square) | The lead/lag scan's only *positive* correlation sits at **L = −6** (GGR momentum lagging the stocks by ~half a year); at positive leads it's negative throughout. The stocks move first; the five-weeks-late GGR print echoes them. |

> **In one sentence:** rising Las Vegas Strip gaming-revenue momentum does *not* precede casino-stock rallies — the tilt is absent-to-contrarian (best *t* = −0.72, and −4.08 when GGR accelerates hardest), a "buy when the tape accelerates" overlay just gives up beta for the same Sharpe, and the lead/lag scan shows the forward-looking stocks turning ~half a year *before* the backward-looking GGR report, so the famous "casino revenue tells you where the stocks go" reads as a lagging echo of a move the market already made.

## What we tested

The gaming-sector folklore holds that Las Vegas **Strip gross gaming revenue** (GGR) is the fundamental pulse of the casino business, so when the monthly Strip-GGR run-rate *accelerates* the operators (MGM, Caesars, Las Vegas Sands, Wynn, Boyd, Penn) are about to run — a top-down, single-number sector-timing edge. We rebuild that signal on the monthly Strip-GGR tape (a hardcoded, clearly-labelled approximate reconstruction of the [Nevada Gaming Control Board](https://gaming.nv.gov/) "Las Vegas Strip" line — the NGCB PDFs aren't machine-fetchable here — with annual sums matched to the published totals) and measure forward 1/3/6/12-month returns of an equal-weight casino basket vs the base rate, with a strict one-month release lag, a Welch *t*, a placebo null, an explicit **lead/lag** scan, and a tradable timing overlay. A deterministic synthetic control with a *planted* GGR→returns link confirms the engine recovers a real edge and can't manufacture one from noise.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "casino revenue leads casino stocks" is mostly the stocks leading the revenue, what a GGR uptick really tells you, and why owning-on-acceleration just gives up return — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | GGR-momentum split returns, a Welch *t* + placebo null, the decisive lead/lag cross-correlation, the timing overlay vs buy-and-hold, robustness (window / threshold / ex-COVID), and a synthetic planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`vegas_gaming_win/`](vegas_gaming_win/). The Strip-GGR input is a hardcoded **approximate reconstruction** of the NGCB "Las Vegas Strip" line (annual sums matched to the published totals), named as such; the casino basket is a **surviving** set of listed operators. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
