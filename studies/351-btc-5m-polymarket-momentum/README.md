# Study 351 — BTC 5-minute Polymarket momentum 🎰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a fresh move predict the close? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Once BTC has moved $70 with 2 minutes left, the window closes that way **94.8%** of the time (n = 2,759, binomial *z* = 47) — monotone in the move, sharper (98.2%) at 1-minute-left. The momentum is genuinely there. |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | EV per share = w − p, and the live Polymarket ask of the favoured side sits at **$0.95–0.99 ≈ w**. Gross edge ~0, negative after the ≤3¢ spread and $30 top-of-book the bot's own config flags. The cheap $0.80 quote doesn't exist when the move is real. |
| **Turn $300 into $14,000?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A martingale survivor. At fair pricing and the pitch's 50%-of-stack sizing, P($300→$14k) = **1.1%**, P(ruin) = **40%**; pay 5¢ over fair and ruin is ~99%. |

> **In one sentence:** the viral "$300 → $14,000 Polymarket bot" rests on a momentum signal that is genuinely real — and entirely priced in: the favoured side is quoted at its own ~95% win-rate, so you pay full freight, and the jackpot is just the lucky 1% of a 50%-of-stack martingale that ruins the other 40%.

## What we tested

A widely-shared post claims a free open-source bot turns $300 into $14,000 by betting Polymarket's **BTC Up/Down 5-minute** markets: with ~2 minutes left, if BTC has already moved $70–100, buy the favoured side (quoted $0.80–0.99) because "the result is basically decided." We fold **45 days** of real 1-minute Bitcoin prices into **12,959** Polymarket-style 5-minute windows (resolve Up iff close ≥ open) and split the claim in two: the continuation win-rate **w** (the signal, measured on the tape) versus the price **p** you actually pay (the favoured ask, captured live from the Polymarket CLOB — read-only, no key, no money). Because a binary share's EV is exactly **w − p**, the whole thing reduces to one comparison — and a Monte-Carlo of the bankroll story settles the rest. (Same high-win-rate / negative-expectancy shape as [Study 301](../../301-triple-rsi/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the momentum is real, why the market already owns it, and how $300 "became" $14,000 — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | continuation vs a binomial null, the EV = w − p identity, the live CLOB-ask capture, a closed-form faithful-engine control, and the martingale Monte-Carlo |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`btc5m_polymarket/`](btc5m_polymarket/). **Not investment advice** — research & education; the advertised bot's hidden key-reading engine is a security hazard, not just a bad trade. See [LICENSE](../../LICENSE).*
