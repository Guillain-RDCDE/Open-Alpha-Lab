# Study 584 — Exchange-Netflows 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**When coins pour onto exchanges, holders are getting ready to sell — does BTC drop when netflows turn positive?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the "inflows = bearish" relation real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | **There is no real exchange-netflow tape to test.** Exchange-labelled on-chain flow is a paid address-clustering product (Glassnode / CryptoQuant / Nansen) with no usable free tier, so a `REAL` stamp (robust *t* ≥ 2 on a REAL tape) is out of reach by construction. On a fair-null synthetic world the engine finds **nothing**: slope of forward BTC return on z(net-inflow) **+4.3 bps/σ** (*t* **+0.47**), sort spread **+10.3 bps** (Welch *t* **+0.38**, placebo *p* **0.72**). Synthetic-only → capped at NONE. |
| **Tradability** — does the netflow trade pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The series you'd trade on is itself **unreachable** without a paid API — and even the *simulated* sign-timed long/short book loses: gross **−1.1 bps/period**, net **−6.4 bps** after 10 bps/turn + 300 bps/yr short borrow, mean net Sharpe **−0.32** across 25 seeds. |
| **"Would the engine catch it if it were real?"** | ![Yes](https://img.shields.io/badge/Yes-2ea44f?style=flat-square) | Plant the folklore (`bear_beta < 0`) and the harness banks it: seed-robust mean slope-*t* goes **−0.01** (null) → **−2.78** at `bear_beta −0.0025` → **−6.67** strong; a single planted seed gives sort spread **+80 bps** (*t* +2.95, placebo *p* 0.005). The gap is **data, not code**. |

> **In one sentence:** the "coins-to-exchanges = bearish BTC" story is intuitive and everywhere on on-chain desks — but the exchange-netflow series it needs is a **paywalled address-clustering product**, so a no-key retail stack has nothing real to test; on a fair simulation of the null the engine (correctly) finds nothing, while a seed-robust positive control proves it *would* bank the folklore if the data existed — `NONE` × `MIRAGE`, gated by the paywall, machinery certified honest.

## What we tested

The on-chain folklore (CryptoQuant, Glassnode, Nansen): a rising exchange **net-inflow** (coins
moving TO exchanges, deposits − withdrawals) is **bearish** for BTC's forward return — deposited
coins are coins about to be *sold* — while a net-**outflow** to cold storage is bullish
accumulation. Because a real exchange-netflow tape is a **paid, address-clustering product** (the
labelling *is* the moat), this study is **synthetic-only** and capped at WEAK/NONE by house rule
(`REAL` needs a robust *t* ≥ 2 on a *real* tape). We build a deterministic, seeded world — a BTC
return series and a persistent net-inflow series with a single knob `bear_beta` that couples
inflows to *lower* forward returns — and run the full engine: a cross-sectional slope whose *sign*
is the folklore, a high-inflow-vs-low-inflow Welch-*t* sort, a label-shuffle placebo, a signal-timed
long/short book with costs + short borrow, a lag/threshold robustness sweep, and a **seed-robust
(25-seed) synthetic positive control**. The data-availability limitation is named on the SIGNAL
axis. *Distinct from the crypto studies on the tape a free stack **can** reach —
[133 Crypto-Seasonality](../133-crypto-seasonality/), [210 Crypto-Trend](../210-crypto-trend/),
[325 Crypto-Fear-Greed](../325-crypto-fear-greed/); this one is the claim whose data is paywalled.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "netflows" are, why "coins to exchanges = about to sell" sounds right, why you can't actually get the data for free, and what a fair simulation of the null shows |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the flow→forward-return slope with its *t*, the high-vs-low sort + Welch *t*, the label-shuffle placebo, the lag/threshold sweep, costs + borrow, and the seed-robust synthetic positive control |

The reproducible headline run (deterministic synthetic world, seed 584, series fp `8c7301772d7b`,
as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery lives in
[`exchange_netflows/`](exchange_netflows/).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`exchange_netflows/`](exchange_netflows/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
