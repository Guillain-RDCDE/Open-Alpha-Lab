# Study 753 — Reverse-Repo-Drain 💵

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a draining ON RRP predict higher stocks? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | At the natural k=1/k=3 horizons the draining regime's next-month return is **below** the filling regime's (spread **−1.0 / −0.2pp**, Welch *t* = **−0.80 / −0.17**), the trailing-change-vs-return correlation is **−0.01**, and the only positive signs (k≥6) peak at Welch **t = +1.20** — the **\|t\| ≥ 2** bar is never met. One fill-then-drain episode, not evidence. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A "hold-when-draining" rule earns net Sharpe **0.49** (long/flat) / **−0.08** (long/short) vs **0.78** for buy-and-hold, ending **1.24×** vs **1.61×** — being out whenever the RRP fills costs more than the drain months return. |
| **Liquidity tell?** | ![Liquidity_tell%3F: Busted](https://img.shields.io/badge/Liquidity_tell%3F-Busted-8b949e?style=flat-square) | The drain is a plumbing artefact of QT + the post-2023 T-bill flood, coincident with **one** bull market — and the 2023-24 "drain = rally" is exactly cancelled by 2021 (a *fill* alongside a bull). |

> **In one sentence:** "a draining reverse-repo facility marks risk-on, so be long stocks" looks tempting because the RRP drained from its $2.55T Dec-2022 peak straight through the 2023-24 rally — but that is one macro episode that also included a *filling* RRP during the 2021 bull, so on a proxy of the FRED balance the drain regime doesn't predict higher forward returns at any horizon (best Welch *t* = 1.2, correlation ≈ 0), a timing rule built on it loses to buy-and-hold, and the co-movement is QT/T-bill plumbing mistaken for a signal.

## What we tested

Liquidity-plumbing commentary (the Zoltan Pozsar *Global Money Notes* lineage and its FinTwit descendants) charts the Fed's **Overnight Reverse Repo (ON RRP) facility** against the S&P 500 as a "hidden liquidity tell": when the RRP **drains**, idle money-market cash is supposedly flowing into risk assets, so a draining RRP marks a **risk-on** regime. The ON RRP balance isn't on yfinance, so we ship a small, **clearly-labelled hardcoded monthly proxy** of the public FRED series [`RRPONTSYD`](https://fred.stlouisfed.org/series/RRPONTSYD) (the 2021 fill → ~$2.55T Dec-2022 peak → 2023-25 drain), align it to month-end SPY, and split next-month returns by whether the RRP was draining — with a one-month execution lag, a block-bootstrap placebo on the long regimes, and a costs-net timing race.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the RRP is, why "drain = cash into stocks" feels right, and why one fill-then-drain cycle can't tell you which way it points |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the drain regime split, drain-vs-fill Welch *t* + a block-bootstrap null on the few long regimes, a drain-timing-vs-buy-and-hold Sharpe race net of costs, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`reverse_repo_drain/`](reverse_repo_drain/). The ON RRP series here is an explicit **hardcoded proxy** for FRED `RRPONTSYD`, not a live pull. SPY is total-return (yfinance auto-adjust). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
