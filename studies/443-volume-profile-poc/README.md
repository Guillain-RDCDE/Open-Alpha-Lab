# Study 443 — Volume Profile POC 🧲

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the POC a magnet? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The prior-day POC is touched **49.7%** of the time — but a **distance-matched random level** (same gap from the open) is touched **53.3%**, *more* than the POC. Edge **−3.6 pp**, **HAC t = −1.32** (wrong sign, far from t ≥ 2), shuffle-POC placebo **p = 0.68**, negative at every touch tolerance and in no name. The "magnet" is volatile range touching nearby levels, not the POC. |
| **Tradability** — can you fade to it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Fading the open toward the POC is **gross-negative** (**−0.102%** per trade, t = −1.46) *before* any spread; break-even one-way cost **−5.1 bps**. There's no edge to pay even a zero spread out of — and intraday levels die to the spread. |
| **"Price returns to the POC"?** | ![Busted](https://img.shields.io/badge/Price_returns_to_POC%3F-Busted-8b949e?style=flat-square) | It returns no more (a touch **less**) than to a random level the same distance away. The claim mistakes a property of a volatile tape for a force. |

> **In one sentence:** the Volume Profile Point of Control is a real, observable level — but it is **not** a magnet: on a 6-name liquid basket of 5-minute bars it is touched *less* (49.7% vs 53.3%, HAC t = −1.32) than a distance-matched random level, fading toward it loses money before costs, and the effect is absent at every touch tolerance and in every name — a textbook case of a volatile tape touching *every* nearby level.

## What we tested

We build a **volume profile** for each session from yfinance **5-minute** bars (50 price bins, typical-price volume) on SPY/QQQ/AAPL/MSFT/NVDA/TSLA, take the **POC** (busiest bin), and stamp it onto the *next* session (known at the prior close — no look-ahead). The believer's claim is that price is **drawn back** to the POC. The honest test is not "is the POC touched?" — on a volatile day *any* nearby level is touched — but whether the POC beats a **distance-matched random control level** (same gap from the open, random side). We measure the touch-rate difference with paired and **HAC** *t* plus Wilson intervals, confront it with a **shuffle-the-POC placebo**, then fade the open toward the POC and charge a spread. A deterministic synthetic control with a *planted* pull proves the harness lights up when a magnet is real — it does not on the tape. **Loud caveats:** Yahoo caps 5m history at ~60 trading days (336 events), so this rejects a strong magnet on liquid names but can't certify a tiny one; and intraday levels die to the bid-ask spread (the capacity verdict).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a POC is, why "it came back to the level!" is a volatility illusion, the distance-matched control, and why the fade loses before costs — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | POC vs distance-matched control with paired/HAC *t* + Wilson intervals, the shuffle-POC placebo, tolerance & cross-section robustness, the fade net of spread, and a synthetic planted-magnet power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`volume_profile_poc/`](volume_profile_poc/). Real tape: yfinance 5m bars, ~60 trading days (2026-03-31 → 2026-06-23), short-span + spread caveats named on the Signal and Tradability axes. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
