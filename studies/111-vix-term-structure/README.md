# Study 111 — VIX-Term-Structure

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Q5-Q1 forward-return spread is **−0.98 bps** at h = 1d (wrong sign, HAC *t* = **−0.08**); negative at all horizons tested; the slope does not predict SPY returns. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The timing overlay trails buy-and-hold by **~2.6%/yr** at zero cost; ~14 switches/yr mean every extra bp of cost only widens the gap. |
| **Beats buy-and-hold?** | ![No](https://img.shields.io/badge/Beats_buy--and--hold%3F-No-8b949e?style=flat-square) | The timer sits in cash during backwardation episodes that average **+10.6 bps/day** — nearly 3x the contango mean — missing the best post-crisis bounces. |

> **In one sentence:** the VIX/VIX3M slope is real market information about volatility expectations, but it does not point in the right direction for timing equities — backwardation episodes (the claimed 'danger' signal) are actually followed by *higher* average SPY returns than calm contango days.

## What we tested

The VIX term structure is one of the most-watched vol signals in markets: when ^VIX trades *below* ^VIX3M (contango, the normal state) the curve slopes upward and consensus reads it as 'risk-on'; when ^VIX spikes *above* ^VIX3M (backwardation) the curve inverts and the community reads it as stress and a sell signal for equities. We take the ratio log(^VIX3M / ^VIX) — positive in contango, negative in backwardation — as the slope signal and test three claims: (1) steeper contango (Q5) predicts better SPY forward returns than deep backwardation (Q1); (2) a binary timer that goes long in contango and flat in backwardation beats unconditional buy-and-hold; (3) regime statistics confirm that contango periods are materially better for equities. We run the full ^VIX3M history (2008–2026, 4,640 daily observations), use rolling out-of-sample quintile ranks, and infer with Newey-West HAC t-stats throughout.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the VIX curve means, why contango looks like a green light, and why looking at it that way backwards costs you the best post-crisis bounces |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | rolling quintile forward-return table, HAC t-stats, timing overlay cost sweep, regime regime breakdown, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`vix_term_structure/`](vix_term_structure/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
