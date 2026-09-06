# Study 994 — The Rounding Tax 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — how much tracking error does whole-share rounding create in a small account? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | A **$3,000** account holding the 6-fund target could not place it: the achieved weights missed the plan by **5.4%** in total absolute terms on average, worst position off by 17.2%, and **0.6%** of the account sat in uninvested residue. Against a fractional-share portfolio running the identical plan, the tracking error was **0.28%/yr** over 15 years. Two numbers put that in proportion. One share of each fund costs $1,518 — the figure usually quoted — but the account that actually *hits* the target weights to within one percentage point is **$38,857**, an order of magnitude more, because owning one of everything gives you a portfolio whose weights are set by share prices rather than by your plan. |
| **Tradability** — is the drag big enough to change what a small investor should hold? | ![Partial](https://img.shields.io/badge/Partial-dab617?style=flat-square) | But the shortfall is mostly **noise, not drag**, and the difference decides the advice. Over 15 years the whole-share portfolio compounded at +9.10% against the fractional ideal's +9.20% — a gap of **-0.10%/yr**, of which cash drag explains -0.03% and trading costs -0.01%, leaving -0.06% that is simply which way the dice fell. Allocation error is mean-zero — you are as likely to be overweight as under — so it produces tracking error without an expected cost; only the cash residue is one-directional. The fix that works is not fractional shares but **fewer funds**: cutting the target to its three largest positions dropped the allocation error from 5.4% to **3.4%**. |

> **In one sentence:** A $3,000 account misses its target allocation by 5.4% and tracks the fractional ideal to 0.28%/yr — but most of that is mean-zero noise, and holding three funds instead of 6 fixes more of it than fractional shares do.

## What we tested

Every model portfolio is written in percentages, and percentages quietly assume
you can buy any fraction of a share. You cannot — not in most retirement plans, most non-US
brokers, or any account holding a fund that trades in whole units. So the portfolio you actually
own is the **rounded** one, and this study measures what the rounding costs.

Three things get lumped together as "the cost of being small" and they behave completely
differently. **Allocation error** — your weights are not your plan's weights — is real and
scales as 1/account size, but it is **mean-zero**: being 2% overweight equities is as likely as
2% underweight, so over twenty years it produces tracking error and almost no expected cost.
**Cash drag** from the uninvested residue is **one-directional** and compounds. **Trading costs**
at rebalancing are one-directional too. `decompose_shortfall` splits them, and the split changes
the advice: quoting the whole gap as a cost overstates it, usually by a factor of two.

The allocator uses **largest-remainder** rather than naive flooring, deliberately — flooring
each position independently leaves a systematic cash residue and would overstate the cost of
being small, which is the direction this study must not err in. Two headline numbers come out of
it: the *one share of everything* figure everyone quotes, and the far larger account that can
actually hit the target weights, because owning one of everything gives you a portfolio whose
weights are set by share prices rather than by your plan. Finally the three escapes are priced —
fractional shares, cheaper-per-share funds, and simply holding fewer funds — and the unglamorous
one wins.
**Dedup:** distinct from **117-rebalancing-bands** and **969-rebalancing-bonus** (rebalancing
policy with continuous weights), **203-bond-etf-vs-index** (tracking error from fund structure),
**512-small-account-fees** (commissions and expense ratios, not share indivisibility) and
**662-cash-vs-bonds** (an allocation question that assumes the allocation is placeable).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a model portfolio actually looks like in a small account, and how much of the shortfall is a real cost rather than a coin flip |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | largest-remainder allocation checked by hand, error-versus-capital curves, a whole-share portfolio run against its fractional twin, the drag/noise decomposition, no-trade bands, and a share-price-only synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`roundingtax/`](roundingtax/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
