# Study 409 — Tweezer Tops & Bottoms 🥢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do aligned wicks forecast a turn? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The raw tweezer-bottom looks great — **+26.78 bps/5d, HAC *t* = +4.16** — until you notice that buying *any* bar earned **+32.35 bps**: the pattern *undershoots* the base rate. Net of each tape's own drift the bottom adds **−0.69 bps (*t* = −0.11)** and the top **+3.79 bps (*t* = +0.74)**; **no horizon clears *t* = 2**. The "significance" was equity beta. *(Survivorship: a survivor basket tilts bullish and still finds nothing.)* |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The only drift-free framing — a market-neutral long-bottom / short-top book — is **−3.39 bps/event gross** (*t* = −0.89) and sinks with spread + borrow. No positive break-even cost exists. |
| **Do two aligned wicks mark the turn?** | ![Busted](https://img.shields.io/badge/Two_aligned_wicks_mark_the_turn%3F-Busted-8b949e?style=flat-square) | A date-shuffle placebo (random entry days, same count & mix) matches the real signal at ***p* = 0.32**; random long-days beat the tweezer bottom **78%** of the time. The matched extremes are a coincidence, not a forecast — and a trend filter doesn't rescue it. |

> **In one sentence:** the tweezer's headline win is **borrowed beta** — buying the bottom makes money only because the market drifts up, and it actually trails the unconditional base rate; strip the drift and the two-aligned-wicks reversal is statistically nothing at every horizon, fails a date-shuffle placebo, and the prior-trend filter the lore swears by changes nothing.

## What we tested

A **tweezer bottom** is two consecutive candles whose **lows match** (within 10 bps) after a short down-leg — folklore says *buy the reversal*; a **tweezer top** is two matched **highs** after an up-leg — *sell*. We detect both by exact OHLC rules across a fixed basket of **31 liquid US large-caps + SPY** (10 years of total-return-adjusted daily bars, cache-first via yfinance), enter the **next bar's open** (one execution lag), and measure forward **1 / 3 / 5 / 10-day** returns **net of each tape's unconditional base rate** — because a long pattern on a rising market is significant by construction. The Signal axis tests the excess with a one-sample HAC *t* and a 5,000-draw date-shuffle placebo; the myth-check toggles the prior-trend filter; Tradability charges spread on both legs plus short borrow on a market-neutral book. A deterministic synthetic panel with a *planted* post-tweezer reversal confirms the detector banks an edge when one exists (and reads ≈ 0 when it doesn't). Survivorship (current-membership basket) is named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a tweezer is, why the headline win is just the market going up, the base-rate trap, and the date-shuffle — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | exact OHLC detector, base-rate-adjusted forward returns by horizon, one-sample HAC *t*, date-shuffle placebo, the trend-filter myth-check, cost sweep, and the synthetic planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`tweezer_tops_bottoms/`](tweezer_tops_bottoms/). Daily bars are total-return adjusted; basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
