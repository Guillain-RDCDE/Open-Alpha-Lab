# Study 366 — Merger-Arbitrage 🤝

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the spread beat fair compensation for the break risk? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The realized arb return is **positive (+1.79%/deal)** with an **80%** win-rate — the folklore is partly real — but the mean **fails t ≥ 2** (one-sample *t* = **0.67**, bootstrap *p*(≤0) = **0.24**, 90% CI **[−2.6%, +5.9%]**), the returns are **left-skewed (−0.91)**, and the headline 7–9% spread is **gross of a break tail** (four deals at ~**−19%**) that erases most of it. A positive-but-insignificant point estimate on a small, survivorship-shaped book ⇒ WEAK, not REAL. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Costs are negligible (a 25-bps round-trip shaves ~0.5pp off a multi-month hold), but the payoff is **short a deal-break put**: one blocked deal (−20% to −40%) erases a year of spread, and breaks **cluster** (antitrust waves, financing freezes) so the tail isn't diversified by holding more deals. A coupon whose mean is inside its own noise and whose risk is a fat left tail is **not a NAV-scale free lunch**. |
| **"Free lunch"?** | ![Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square) | The "steady market-neutral spread" is **insurance premium in a costume** — you're paid for *writing* a deal-break put, and on an honest tape the premium ≈ the expected loss. The 80%-win / rare-deep-loser shape is the classic **short-volatility, picking-up-pennies** payoff. |

> **In one sentence:** the merger-arb spread looks like a free 7–9% coupon, but once you count the four deals in our 20-deal book that *broke* (−19% each), the average deal returns just **+1.79%** at *t* = 0.67 — a left-skewed payoff statistically indistinguishable from a fair bet, because the spread is really the premium you collect for **writing insurance against the deal falling through**.

## What we tested

After an all-cash takeover is announced, the target trades a few percent **below** the offer; the pitch is to buy it, hold to close, and pocket that spread as a steady, market-neutral coupon. We harvest realized arb returns from a hardcoded book of **20 real announced all-cash US M&A deals** (2020–2024) — a deliberate mix of clean closes (Microsoft/Activision, Pfizer/Seagen, Cisco/Splunk, …) and high-profile **breaks** (JetBlue/Spirit, TD/First Horizon, Avangrid/PNM, MaxLinear/Silicon Motion). Entry is the close one day after the announcement (no look-ahead); a break snaps the target back to its un-bid standalone level. We then ask whether the average deal pays *more* than fair compensation for the break tail, with a one-sample *t*, a bootstrap of the book mean, win-rate vs the loser magnitude, and a deterministic synthetic deal book whose `edge` knob proves the engine lights up only on a *real* edge. (Acquired targets delist and vanish from yfinance, so entry/exit are documented public closes, confirmed against live data for the two surviving names — named on the Signal axis.)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a 7% spread isn't free money, what happens when a deal breaks, and why "wins 80% of the time" is the wrong question — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-deal realized arb returns, the break tail and negative skew, a one-sample *t* + bootstrap null, costs, and a synthetic fair-bet / planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`merger_arbitrage/`](merger_arbitrage/). The deal book is real (offer + dates + outcome from public announcements); entry/exit marks are **documented closes** (acquired targets delist), confirmed against live data where the target still trades. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
