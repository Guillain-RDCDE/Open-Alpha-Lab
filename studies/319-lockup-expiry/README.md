# Study 319 — Lockup-Expiry

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Across a 29-IPO basket (2019–2024), **no** window around the 180-day unlock shows a significant negative abnormal return. The expiry bar is *positive* (AAR +62.9 bps, HAC *t* = +0.87), the post-expiry [0,+5] window is *positive* too (+322.7 bps, *t* = +1.75 — wrong sign); the only sag-leaning window is the run-up [−5,−1] (−316.7 bps, *t* = −1.98), below the bar with a CI touching zero. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The folklore "short the unlock" trade **loses money gross** (−322.7 bps over [0,+5], *t* = −1.75) — IPOs tend to *rise* vs the market just after the unlock. The only profitable variant (front-running the run-up) is wiped out by two-leg costs and would need hard-to-borrow short on recent IPOs on top. |
| **A predictable supply-shock sag?** | ![Not_supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | Field & Hanka (2001) found a real ≈ −1.5% expiry sag — but the date is in the S-1, known months ahead. On this recent basket the effect is gone: fully anticipated and arbitraged. |

> **In one sentence:** the 180-day lock-up expiry is the textbook "predictable" supply shock, and twenty years ago it really did print a small sag — but on a 2019–2024 basket the abnormal return at the unlock is flat-to-positive, the believers' short loses money before a penny of cost, and the only negative window (the run-up) can't clear the bar.

## What we tested

Trading desks and IPO-calendar blogs market a simple edge: *180 days after an IPO, insiders are finally free to sell, so the flood of supply pushes the price down — short it.* The early academic evidence is real (Field & Hanka 2001; Bradley et al. 2001 found a ≈ −1.5% abnormal return around expiry, bigger for VC-backed firms). We test whether any of it survives today, as a textbook **event study** on a hardcoded basket of 29 recent US IPOs: for each we compute daily **abnormal returns** (stock − SPY, a market-adjusted synthetic control), align them in event-time around the lock-up-expiry bar (first close + 125 trading days ≈ 180 calendar days), and measure the average abnormal return at the unlock and the cumulative abnormal return over windows around it — with cross-sectional HAC *t*-stats and block-bootstrap CIs — then run the believers' market-hedged short net of two-leg costs. A deterministic synthetic panel with a planted −300 bps sag is the positive control (it proves the harness can see a sag when one exists). *Distinct from [Study 219 — IPO-Pop](../219-ipo-pop/), which measures the first-day pop and the multi-year drift, not the unlock event.* **Survivorship bias is named on the Signal axis:** the basket holds only tickers still on yfinance.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why insiders unlocking *should* drop the price, the event-study picture, and why the sag isn't there anymore |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | AAR/CAR per window, cross-sectional HAC *t*, block-bootstrap CIs, the hedged short net of two-leg costs, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`lockup_expiry/`](lockup_expiry/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
