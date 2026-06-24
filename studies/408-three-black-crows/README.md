# Study 408 — Three Black Crows 🐦‍⬛

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a bearish edge? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Shorting three black crows (enter next open, one lag) **loses at every horizon** — signed-short mean **−0.05% → −0.28%** (1→10 days), HAC *t* **negative** (never near +2), coin-flip placebo *p* ≈ **0.94–0.98** (the real signal sits in the *left* tail; ~95%+ of random picks did better). No bearish edge. **Survivorship** caveat actually tilts the test *toward* the claim — and it still fails. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Negative **gross**; net of a 5-bps × 2 round trip + 50 bps/yr borrow it is worse (**−0.28% net at 5d**). Nothing under the signal to charge costs against. |
| **"Crash incoming"?** | ![Busted](https://img.shields.io/badge/Crash_incoming%3F-Busted-8b949e?style=flat-square) | The "sell-off is just starting" reading is **backwards**: three big red days are a stock that already fell, and it drifts mildly **back up** (the short loses; the only borderline significance, 10d *t* = −2.06, is the *bounce*). Strict-crow and prior-uptrend filters don't rescue it. |

> **In one sentence:** the scariest-looking candle in the book — three long red days that "warn of a crash" — has **no predictive power** on 21.5 years of real tape; shorting it *loses money before costs* (signed-short −0.05% to −0.28%, HAC *t* < 0, placebo *p* ≈ 0.95), the post-pattern drift is mildly **up** (a bounce, not a crash), and no stricter recipe rescues it — on a survivor basket that we deliberately stacked in the lore's favour.

## What we tested

We rebuild three black crows as a clean signed-**short** event study on a fixed **30-name liquid US large-cap + SPY** basket (yfinance daily OHLCV, 2005→2026, 161,970 bars). A precise OHLC detector flags **every** occurrence (three stacked red bodies, each closing lower, each opening inside the prior body); we wait for the confirming close, enter the **next open** (one execution lag), and measure the forward **1 / 3 / 5 / 10-day** return held short. The Signal axis tests the signed-short mean against zero with a HAC + one-sample *t* and a 5,000-draw coin-flip placebo; Tradability charges a round-trip cost + short borrow. Two myth-checks ask whether the **strict long-bodied crow** or a **prior-uptrend** (true reversal) filter helps. A deterministic synthetic control with a *planted* post-pattern crash confirms the engine would catch a real one (it lights up at *t* = 3–7) and that zero edge cannot fake significance. Survivorship — the basket excludes firms that actually crashed and delisted, biasing the test *toward* the claim — is named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the pattern is, why "three down days" already means the drop happened, why shorting it loses, and why no stricter recipe saves it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the precise detector, signed-short forward 1/3/5/10-day event study, HAC + one-sample + Welch *t*, a coin-flip placebo, costs + borrow, the strict-crow & prior-uptrend myth checks, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`three_black_crows/`](three_black_crows/). Detector is the precise real-body three-black-crows (strict + prior-uptrend variants for the myth-check). Basket is **survivors** — named on the Signal axis (and it cuts *toward* the claim). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
