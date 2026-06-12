# Study 45 — Vanishing-Act 🎩

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do small-caps beat large-caps? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No. Russell 2000 − S&P 500 over **39 years** earns **+0.03%/yr, Sharpe 0.00 (Lo t = 0.02)** — zero — and on the *same months* small-caps trail large risk-adjusted (Sharpe 0.47 vs 0.62). No pair clears |t| = 2. |
| **Tradability** — is there a size premium to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Not today, and not detectably ever on this bench. The tradable ETF spreads (IWM−SPY, IJR−IVV) are insignificant; what little survives (S&P 600) looks like a **quality screen**, not size. |
| **"Survived publication"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Banz published in 1981; **every month of this sample is post-publication** and the premium never showed. The famous "positive then negative around 2010" is a window choice — on the long series the split's difference carries a bootstrap p of 0.45, and the trailing-decade mean flips sign repeatedly (−5.0% to +6.9%/yr). |

> **In one sentence:** the size premium — finance's original factor and the reason small-cap funds exist — never showed up on 39 years of tradable proxies, all of them post-Banz: the spread is statistically zero, small-caps trail large-caps risk-adjusted on identical samples, and the sign of any "decade of size" is decided by where the decade ends.

## What we tested

The **size effect** (Banz 1981; [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it at Sharpe `0.747`): small-cap stocks out-earn large-caps. It's the original factor, the SMB in every textbook model. We test it on the **longest free proxies** — Russell 2000 vs S&P 500 back to **1987** (price indices, which is *conservative*: a total-return spread would be more negative), plus the total-return ETF pairs IWM/SPY and IJR/IVV — measuring the SMB spread, its Lo (2002) t-stat, the two legs' standalone Sharpes **on each pair's common months** (never across mismatched samples), the trailing-10-year sign-flip exhibit, and a post-hoc 2010 window split scored honestly with a circular-block-bootstrap p-value on the difference. The offline control is a synthetic world whose size premium can ramp from positive to negative — and a null.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "small beats big" launched an industry on data nobody can find a premium in |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the SMB spread with its Lo t-stat, aligned leg Sharpes, the window-split bootstrap, the sign-flip exhibit, the quality-not-size caveat |

The fingerprinted real-data run (^RUT/^GSPC 1987–2026 + ETF pairs, as-of 2026-06-01, fp `21b43ddb5825`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [vanishing_act/data.py](vanishing_act/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
