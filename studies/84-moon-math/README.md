# Study 84 — Moon-Math

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | In-sample R² = 0.69 — but log(time) achieves **R² = 0.90** with no scarcity narrative. First-diff R² = **0.09%**: S2F has no forecasting content for price *changes*. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | OOS R² = **-2.36**; MAPE = **200%**. In 2022 the model predicted ~$62–66k while BTC traded at $15,787. |
| **Spurious regression?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Two co-trending series (S2F and price both rise monotonically) always produce a high R² — a plain clock fits *better*. |

> **In one sentence:** the Stock-to-Flow model's famous R² is a textbook spurious regression of two upward-trending series — log(time) fits BTC price better than log(S2F), the model has zero first-difference forecasting content, and it predicted $62k+ through the entire 2022 collapse to $15,787.

## What we tested

PlanB's Stock-to-Flow model (2019): `log(BTC price) = -1.84 + 3.3 * log(S2F)` where
S2F = cumulative mined supply divided by annual new issuance. The claim is that Bitcoin's
*scarcity* — encoded in the protocol's halving schedule — is the primary driver of price,
with a power-law exponent around 3.3. The in-sample R² (~0.69 to 0.94 depending on vintage)
was widely cited as evidence. We steelman it honestly: we derive S2F from the protocol rules
(it is fully deterministic, no estimation needed), replicate the levels regression, then
apply three disconfirmatory tests — the log(time) alternative, first-difference regression,
and post-2021 out-of-sample evaluation — to distinguish a structural causal relationship
from a spurious trend-on-trend fit.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the S2F story, the "looks amazing" R², why two trends always correlate, the 2022 crash in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Granger-Newbold spurious regression theory, Engle-Granger cointegration test, first-diff OLS with HAC SE, OOS R² and MAPE decomposition, synthetic positive/null controls |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`moon_math/`](moon_math/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
