# Study 45 — Vanishing-Act 🎩

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do small-caps beat large-caps? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No. Russell 2000 − S&P 500 over **39 years** earns **+0.1%/yr, Sharpe 0.01 (Lo t = 0.05)** — zero — and small-caps *trail* large on a risk-adjusted basis (Sharpe 0.47 vs 0.61). No pair clears |t| = 2. |
| **Tradability** — is there a size premium to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Not today. The tradable ETF spreads (IWM−SPY, IJR−IVV) are statistically insignificant; what little survives (S&P 600) looks like a **quality screen**, not size. |
| **"Survived publication"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Every pair flips sign: SMB **positive before 2010, negative since**. The cleanest post-publication decay on the bench. |

> **In one sentence:** the size premium — finance's original factor and the reason small-cap funds exist — has *vanished*: zero over 39 years, small-caps actually trailing large-caps risk-adjusted, and a clean sign reversal since 2010 — it worked until Banz put it in print in 1981, and the modern market has arbitraged it away.

## What we tested

The **size effect** (Banz 1981; [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it at Sharpe `0.747`): small-cap stocks out-earn large-caps. It's the original factor, the SMB in every textbook model. We test it on the **longest free proxies** — Russell 2000 vs S&P 500 back to **1987**, plus the total-return ETF pairs IWM/SPY and IJR/IVV — measuring the SMB spread, its Lo (2002) t-stat, the two legs' standalone Sharpes, and a pre/post-2010 decay split. The offline control is a synthetic world whose size premium can ramp from positive to negative (the Banz story) — and a null.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "small beats big" launched an industry and then quietly stopped being true |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the SMB spread with its Lo t-stat, small-vs-large Sharpe, the sign reversal since 2010, the quality-not-size caveat |

The fingerprinted real-data run (^RUT/^GSPC 1987–2026 + ETF pairs, fp `da49197bc37d`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [vanishing_act/data.py](vanishing_act/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
