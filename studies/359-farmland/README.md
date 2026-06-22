# Study 359 — Farmland 🌾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — a real low-beta inflation hedge? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The appraisal index *looks* like a calm diversifier (corr to S&P **−0.13**, +8.4%/yr real) with a **positive** inflation loading — but that loading is **insignificant** (β = +0.66, **t = 0.83**, 32 yrs), and the only tradable proxies are equity-like (β = **0.7–0.9**, ~30% vol). Real-economy plausible, statistically unproven. |
| **Tradability** — can you actually hold it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | LAND & FPI are buyable but small, **leveraged**, illiquid REITs: net Sharpe **~0.2** vs SPY ~1.0, ~30% annual vol. You can own farmland — just not the smooth 6.6%-vol thing the index advertises. |
| **Smooth & uncorrelated?** | ![Busted](https://img.shields.io/badge/Smooth_%26_uncorrelated%3F-Busted-c0392b?style=flat-square) | An appraisal artifact. ρ₁ = **0.63**; un-smoothing **doubles** the vol (6.6% → 13.5%). A synthetic control shows appraisal shrinks a true 0.32 market beta to ~**0.05** — the "uncorrelated calm" is measurement, not the asset. |

> **In one sentence:** farmland is a genuine real asset with a high real return and a mild (statistically weak) inflation tilt — but the "smooth, uncorrelated, best inflation hedge" reputation is an **appraisal-smoothing illusion**, and the only thing you can actually buy (LAND/FPI) is a leveraged, ~30%-vol, market-correlated REIT.

## What we tested

The folklore: the **NCREIF Farmland Index** posts steady, low-volatility, near-zero-correlation total returns, and billionaires (Bill Gates is the largest private US farmland owner) treat farmland as the best inflation hedge. We split that into what's measurable. (1) **The tradable proxies** — the two listed farmland REITs **LAND** (Gladstone Land) and **FPI** (Farmland Partners) — pulled from yfinance and regressed on **SPY**: they carry ~30% vol and a significant 0.7–0.9 market beta, nothing like a calm diversifier. (2) **The inflation-hedge claim** — a hardcoded, **cited public NCREIF farmland annual series** (1992–2023) regressed on **CPI**: a positive but *insignificant* loading (t = 0.83) alongside a high +8.4%/yr real return. (3) **The smoothness itself** — the index's 0.63 lag-1 autocorrelation is the fingerprint of **appraisal smoothing**; un-smoothing (Geltner) doubles the volatility, and a deterministic synthetic control reproduces the closed-form variance/beta shrink exactly. (Same "real signal, mirage once you account for the measurement artifact" shape as [Study 120](../../120-excess-cape-yield/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the smooth chart lies, what farmland really did, and what you actually own if you buy "farmland" — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | CAPM beta with HAC *t*, the inflation regression vs CPI, Geltner un-smoothing, and a closed-form synthetic appraisal-smoothing control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`farmland/`](farmland/). **Not investment advice** — research & education. The NCREIF series is a cited public proxy for the paywalled quarterly index. See [LICENSE](../../LICENSE).*
