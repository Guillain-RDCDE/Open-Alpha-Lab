# Study 755 — JOLTS-Quits 🧑‍🏭

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do falling quits precede weaker returns? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Tiny, insignificant, and **sign-unstable**: falling-quits forward SPY is *below* base at 1–3 months but *above* it at 6–12 (12m **+11.4%** vs **+10.8%**), the best Welch *t* is **−0.40**, the down-rate matches base everywhere, and the window choice flips the sign. The named **cyclicals** leg goes the *wrong* way (**+6.3% vs +3.6%**, *t* = +1.7). Indistinguishable from noise. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | "Cash when quits fall" turns $1 into **6.0×** vs **8.7×** for buy-and-hold (**+7.8%/yr** vs **+9.7%**). Its Sharpe (**0.66 vs 0.64**) is a rounding-tie bought only by sitting out ~29% of months — **beta you de-risked**, not alpha. |
| **Leading gauge?** | ![Not supported](https://img.shields.io/badge/Leading_gauge%3F-Not_supported-8b949e?style=flat-square) | Lead/lag peaks (positively) at **L = −4 months** — quits *lag* the market by a quarter — and is flat at the positive leads a real early-warning needs. Then JOLTS publishes the print **~6 weeks late**, so even the echo is stale. A coincident-to-lagging tell, not a leader. |

> **In one sentence:** the JOLTS quits rate is a genuine worker-confidence gauge but a non-signal for equities — falling quits are followed by returns that are *below* average for a quarter and *above* average by a year (best *t* = −0.4, sign set by your lookback), the quits drop lines up with a market move already three-to-four months old and only reaches you six weeks after that, and a "sell when quits fall" overlay ends a third poorer than buy-and-hold for a Sharpe it merely ties.

## What we tested

The labour-nowcasting folklore says the **JOLTS quits rate is worker confidence made visible** — people quit only when sure of something better — so a *falling* quits rate signals fading confidence and precedes equity, and especially **cyclical**, weakness ([BLS JOLTS](https://www.bls.gov/jlt/), FRED [`JTSQUR`](https://fred.stlouisfed.org/series/JTSQUR)). We rebuild that signal on the monthly quits-rate tape (a hardcoded snapshot of `JTSQUR`, Total Nonfarm, SA) and measure forward 1/3/6/12-month **SPY** and **cyclical-minus-defensive** (XLY − XLP) returns in falling- vs rising-quits months against the base rate, with a Welch *t*, a placebo null, an explicit **lead/lag** scan, and a tradable overlay — all under the honest **2-month JOLTS release lag** (the print for month *t* is only public in month *t+2*). A deterministic synthetic control confirms the engine recovers a *planted* quits→returns link and can't manufacture one from noise.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "quits = confidence" feels right, why the quits drop actually *follows* the market, and why selling on it loses money — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | quits-momentum split returns, a Welch *t* + placebo null, the decisive lead/lag cross-correlation, the cyclicals leg, the timing overlay vs buy-and-hold, the release-lag tax, and a synthetic planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`jolts_quits/`](jolts_quits/). The quits rate here is a hardcoded **snapshot** of FRED `JTSQUR` (the settled print, not the real-time vintage), named as such; SPY/XLY/XLP are total-return adjusted. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
