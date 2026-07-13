# Study 767 — VIX9D-Term 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Q5-Q1 forward-return spread is **−2.79 bps** at h = 1d (wrong sign, HAC *t* = **−0.35**); negative at every horizon, quintile pattern U-shaped not monotone. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The timing overlay trails buy-and-hold by **~4.1%/yr** at zero cost; **~37 switches/yr** mean by 5 bps/switch the underperformance is itself significant (*t* = −2.09). |
| **Beats buy-and-hold?** | ![No](https://img.shields.io/badge/Beats_buy--and--hold%3F-No-8b949e?style=flat-square) | The timer sits flat during backwardation, the higher-*raw*-return regime (**+6.04 vs +4.97 bps/day**) — it flees the more volatile days that are also the higher-returning ones. |

> **In one sentence:** the ^VIX9D/^VIX short-end slope is real information about *volatility* — contango days are genuinely calmer (Sharpe 0.97 vs 0.60) — but it does not point the right way for timing equity *direction*: backwardation (the claimed 'danger' signal) is followed by *higher* average SPY returns, and a timer that flees it underperforms buy-and-hold before a cent of cost.

## What we tested

The 9-day VIX (^VIX9D) is the twitchiest point on the volatility surface, and its
relationship to the 30-day ^VIX is the most-watched short-end term-structure signal in
vol-trading circles: when ^VIX9D trades *below* ^VIX the front end slopes up (contango,
the normal calm state), and when ^VIX9D spikes *above* ^VIX the very short end inverts
into backwardation — read by the community as acute near-term fear and a risk-off timer
for equities. We take the ratio log(^VIX / ^VIX9D) — positive in contango, negative in
backwardation — and test three claims: (1) steeper contango (Q5) predicts better SPY
forward returns than deep backwardation (Q1); (2) a binary timer that goes long in
contango and flat in backwardation beats unconditional buy-and-hold; (3) regime
statistics confirm contango is materially better for equities. We run the full ^VIX9D
history (2011–2026, 3,901 daily observations), use rolling out-of-sample quintile
ranks, and infer with Newey-West HAC t-stats throughout. This is the short-end cousin of
[Study 111 — VIX-Term-Structure](../111-vix-term-structure/), which tested the ^VIX/^VIX3M
slope.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the 9-day/30-day VIX curve means, why an inverted front end looks like a red light, and why fleeing it costs you the best post-shock bounces |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | rolling quintile forward-return table, HAC t-stats, timing overlay cost sweep, regime decomposition (Sharpe vs raw-return split), synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`vix9d_term/`](vix9d_term/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
