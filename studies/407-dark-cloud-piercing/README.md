# Study 407 — Dark Cloud & Piercing 🌩️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do the twins predict the reversal? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The signed day-after return is **−0.078%** at HAC *t* = **−2.57** — *wrong-signed and significant*. Negative at every horizon, win-rate **below 50%**, label-shuffle placebo *p* ≈ **0.998** (a coin beats it almost every time). The only |*t*| ≥ 2 reading points the wrong way. **Survivorship** works *for* the bullish leg and still can't save it. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The gross signed edge is negative before a cent of cost; a 5 bps round trip + borrow on the short Dark Cloud leg only widens the loss to **−0.18% to −0.21%** per event. Nothing to scale. |
| **A reliable reversal?** | ![Busted](https://img.shields.io/badge/Reliable_reversal%3F-Busted-8b949e?style=flat-square) | The **Dark Cloud** short keeps *rising* (1-day leg *t* = −2.85); the **Piercing** long looks green at 5–10 days but **underperforms the unconditional drift** — it's beta, not reversal alpha. Trend/volume filters don't flip it; the synthetic control proves the test *would* see a real reversal if one existed. |

> **In one sentence:** the Piercing Line and Dark Cloud Cover are sold as failed-sell-off / failed-rally reversals, but on 21 years of liquid US large-caps the signed effect is **wrong-signed** (HAC *t* = −2.57 the day after, win-rate below 50%, placebo *p* ≈ 0.998) — the bearish leg keeps climbing and the "good" bullish leg is just market drift the pattern lags.

## What we tested

We rebuild the twins as a clean event study on a fixed **30-name basket** (29 liquid US large-caps + **SPY**, yfinance daily, total-return adjusted, 2005–2026): detect every **Piercing Line** (prior down day, gap-down open, close back above the prior body's midpoint, short of a full engulfing) and **Dark Cloud Cover** (the mirror) by the precise OHLC rule, enter the **next open** (one execution lag), and measure the forward **1 / 3 / 5 / 10-day** return **signed by the pattern direction** (long after a Piercing, short after a Dark Cloud). The Signal axis tests the signed mean against zero with a Newey-West HAC *t* and a plain one-sample *t*, against the unconditional base pool (Welch *t*), and against a 10,000-draw label-shuffle placebo; Tradability charges a round trip per event plus borrow on the short leg. A deterministic synthetic control with a *planted* day-after reversal confirms the harness can bank a real edge and that a random walk cannot fake one. Survivorship (the basket is names still trading in 2026) is named on the Signal axis — and works *for* the bullish claim.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the twins are, why a "failed gap" sounds like a reversal but often isn't, the leg split, and why the bullish leg's gain is a beta trap — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the precise OHLC detector, signed forward 1/3/5/10-day returns, HAC + one-sample *t* vs zero, a 10k-draw label-shuffle placebo, the Piercing-vs-Dark-Cloud leg split vs the unconditional drift, the trend/volume myth-check, and a synthetic planted-reversal control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`dark_cloud_piercing/`](dark_cloud_piercing/). Basket is **survivors** — named on the Signal axis. Real tape: yfinance daily, `auto_adjust=True` (total-return), as-of 2026-06-18, fingerprint `bf1d6cb7ca54`. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
