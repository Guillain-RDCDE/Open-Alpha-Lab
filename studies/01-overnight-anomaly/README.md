# Study 01 — The Overnight Anomaly 🌙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | SPY overnight mean carries a Newey-West *t* ≈ 5; confirmed across ~441 S&P 500 stocks (~69% have overnight > intraday Sharpe). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Magnitude inflated by a calendar-time illusion; what's left is mostly gap-risk **beta** below trading costs; it **decays** (5y Sharpe ~2→~0.5) and **doesn't scale** (capacity ~\$10M). |
| **Manipulation?** — does the pattern prove fraud? | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | Bayesian posterior ≈ 2–3%; the discriminating evidence (foreign-ETF & China inversions, negative P&L at scale) favours microstructure. |

> **In one sentence:** the overnight effect is *real and broad*, but its size is a clock illusion, what remains is mostly beta below trading costs, it doesn't scale, and the "market-manipulation" reading isn't supported.

## What we tested

Across world markets, almost all the long-run gain has accrued **overnight** (close → next open); the trading day is flat. **Bruce Knuteson** reads this as the fingerprint of large-scale market manipulation — one enormous book quietly nudging prices up in illiquid pre-open windows, worldwide, for decades ([arXiv:1912.01708](https://arxiv.org/abs/1912.01708) and follow-ups). This study is a *response*: we take the fact seriously, then audit the inference end-to-end. (`python papers/download_papers.py` fetches the source papers — [why we don't redistribute](papers/README.md).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story + the stakes in plain language: why markets move at night, and the tricks that inflate the headline |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full machinery: HAC/Lo inference, the clock illusion, firm-level breadth, alpha-vs-beta, decay, capacity, the steelman manipulator, a Reality Check and a Bayesian posterior |

Both render inline on GitHub (pre-executed). Also here: a point-by-point rebuttal of Knuteson's claims in **[RESPONSE.md](RESPONSE.md)**, and the working paper *"Overnight, or Overhyped?"* — **[paper/overnight_alpha.pdf](paper/)** (LaTeX source + reproducible figure script).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
