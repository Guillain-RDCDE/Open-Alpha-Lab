# Study 757 — Cass-Freight 🚚

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the freight cycle predict forward stocks? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | After a 2-month publication+execution lag, "freight expanding" leaves only a **small, insignificant** drift on SPY (strongest **t = 1.09** at 6m, placebo **p = 0.13** — **fails t ≥ 2**) and a **negative** excess on **IYT**, the freight-sensitive sector it should predict best (12m **−0.94** *t*). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | An "own-when-freight-expands" overlay earns a **lower Sharpe than buy-and-hold** on both tapes (**0.59 vs 0.61** SPY; **0.42 vs 0.58** IYT); the long/short version collapses (**0.09** / **−0.05**) because freight contractions bracket the exact market **bottoms** you wanted to own. |
| **Leads the cycle?** | ![Leads_the_cycle%3F: Busted](https://img.shields.io/badge/Leads_the_cycle%3F-Busted-8b949e?style=flat-square) | The lead-lag cross-correlation peaks at **k = −3 months** on both SPY and IYT — freight moves most in step with the equity return of a **quarter earlier**. Stocks lead freight, not the reverse. |

> **In one sentence:** "watch the freight, it leads the real economy" is a real-economy *gauge* dressed as a *forecast* — on a cited Cass-shipments proxy aligned to real SPY and IYT with a strict post-publication lag, the freight cycle **lags** the market by about three months, leaves no significant forward edge (and a *negative* one on transports), and every timing overlay loses to simply buying and holding.

## What we tested

The freight-macro world's favourite line is that the **[Cass Freight Index](https://www.cassinfo.com/freight-audit-payment/cass-transportation-indexes/cass-freight-index)** — Cass Information Systems' monthly gauge of the shipments its freight-payment clients move — is a **real-economy leading indicator**: trucks and trains move the physical economy, so when freight rolls over an industrial and equity slowdown is already baking, not yet priced (the Dow-Theory "watch the transports" instinct, quantified). Cass's monthly index is not freely API-available, so we build a small **cited, approximate monthly proxy** of the shipments series (year-over-year expansion vs the 2001, 2008–09, 2015–16, 2019, 2020 and 2022–24 contractions) and align it to **real** SPY and IYT month-end closes, with a **2-month lag** (Cass prints a month in arrears + one month to trade). We then ask the only two questions that matter: does freight actually *lead* stocks, and does an overlay built on it beat buy-and-hold?

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the Cass index is, why "watch the freight" feels right, and why a gauge of goods already shipped can't front-run a market that already turned — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full machinery: conditional vs unconditional forward returns on SPY & IYT, a Welch *t* + placebo null, the lead-lag cross-correlation, a net-of-cost overlay-vs-buy-and-hold Sharpe race, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`cass_freight/`](cass_freight/). The freight series is an explicit **approximate proxy** (cited annual anchors interpolated), not Cass's licensed tape; SPY and IYT are real and price-only. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
