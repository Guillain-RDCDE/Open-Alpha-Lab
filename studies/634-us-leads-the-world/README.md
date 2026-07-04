# Study 634 — US-Leads-the-World 🌏

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does today's US close predict tomorrow's overseas sessions? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Overwhelmingly. Next-session slopes on the SPY day-t return: Tokyo **+0.561 (HAC t = +19.6)**, Sydney +0.387 (+11.2), London +0.230 (+10.5), Frankfurt +0.222 (+9.2); GLOBAL-4 basket **+0.350 (t = +14.85, R² = 19.5%)** over 29.5 years. 50-seed shuffle placebo collapses to \|t\| ≈ 0.75. Flat across \|US-move\| quintiles; still t = +4.49 in 2018–2026. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | **50–71% of the slope is delivered in the overnight gap** — a print that exists before any order can fill. The only executable version (follow the US sign at the next foreign open) nets **−0.5 bps/day** (Tokyo) and **−5.0** (Frankfurt) at just 5 bps one-way; the one "surviving" line (^AXJO) rests on a staggered-auction index open that is not a tradable price; and the US wrappers (EWJ/EWG) trade during US hours, where the spillover is contemporaneous. Predictability without a vehicle. |
| **Survives the post-2010 algo era?** | ![Confirmed](https://img.shields.io/badge/Survives_algo_era%3F-Confirmed-8b949e?style=flat-square) | GLOBAL-4 post-2010 slope **+0.321 (HAC t = +7.1)**, pre/post difference insignificant (t_diff = +1.06); Tokyo's slope *rose* (+0.544 → +0.584). Only London genuinely decayed (halved, t_diff = +3.42). Information flow priced at untradable prints leaves nothing for the algos to eat. |

> **In one sentence:** "America sneezes" is one of the strongest signals on the whole bench — a 1% US day moves tomorrow's Tokyo session +0.56% (HAC t ≈ +20) and the effect hasn't decayed in three decades — yet it is a textbook **Mirage**: two-thirds of it is delivered at the opening print you cannot trade, the executable open-to-close residue dies at 5 bps one-way, and the US-listed wrappers reprice contemporaneously, so the time zone that creates the predictability is exactly what every accessible vehicle takes away.

## What we tested

We pair each **SPY day-t close-to-close** return (public at 16:00 ET) with the **first session strictly after date t** of four overseas indices whose trading days start after New York's ends — ^N225 (Tokyo), ^GDAXI (Frankfurt), ^FTSE (London), ^AXJO (Sydney) — 1997→2026, 7,415+ pairs per market. The Signal axis runs per-market and GLOBAL-4-basket predictive OLS with **Newey-West HAC t's**, a 50-seed shuffle placebo, decade-by-decade decay and \|US-move\|-size conditioning. The mechanism test decomposes each next session into the **overnight gap** (previous close → open — prints before the first tradable tick) vs the **open→close** leg, on **live-open days only** (Yahoo's index opens are stale for 88% of ^FTSE and 41% of ^AXJO days — measured and named). Tradability prices the only executable trade — follow the US sign at the next foreign open — net of one-way costs, against the **phantom** close-to-close backtest (which needs a time machine: Tokyo's close prints before the US close exists) and the **wrapper trap** (EWJ/EWG trade during US hours). The third axis splits pre/post-2010 with a t on the difference. A deterministic synthetic world with a planted spillover-beta knob proves the pipeline recovers exactly what is planted and nothing when beta = 0. Distinct from [379-etf-lead-lag](../379-etf-lead-lag) — that was *same-timezone* intra-US lead-lag (verified None); this is the documented *cross-timezone* effect (Hamao-Masulis-Ng 1990; Rapach-Strauss-Zhou 2013). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why Tokyo "copies" Wall Street every single morning, why that's been true for 30 years without anyone arbitraging it away, and why you still can't make money on it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-market HAC slopes + the gap/open→close decomposition, stale-open forensics, era decay + size conditioning, the feasible-vs-phantom trade accounting, the pre/post-2010 difference test, and the planted-beta synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`us_leads_the_world/`](us_leads_the_world/). SPY is total-return adjusted; the four cash indices are price-only levels (labeled). Exactly one execution lag — the predictor is public before the target session opens, by construction of the time zones. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
