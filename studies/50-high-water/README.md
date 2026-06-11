# Study 50 — High-Water 🏔️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do stocks near their 52-week high out-earn? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No premium found. The nearness long-short earns **−8.4%/yr (Sharpe −0.40, Lo t −2.2)** on large caps, 1996–2026 — but the *negative sign* is partly manufactured by our survivor panel (the short leg holds fallen names we *know* rebounded into today's index), so the bias-robust reading is "no positive premium", not "a reliable loser". |
| **Tradability** — is there a premium to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Nothing to harvest: gross is non-positive at best, and one-way turnover (~3.2× NAV/mo) only digs (net Sharpe −0.58 at 10 bp). |
| **"A distinct anomaly"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The robust half of the verdict: the 52-week-high hedge is **0.82 correlated** with the standard 12-2 momentum hedge — the same factor, relabelled with an anchoring story. Correlation is insensitive to the panel's bias. |

> **In one sentence:** the 52-week-high effect is momentum wearing a behavioural hat — 0.82 correlated with standard 12-2 momentum, *not* the distinct anomaly it's sold as — and on this large-cap survivor panel it showed no premium at all (a negative long-short whose sign the panel itself partly manufactures).

## What we tested

The **52-week-high effect** (George & Hwang 2004; [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it at Sharpe `0.153`): stocks trading near their 52-week high earn higher subsequent returns, via an anchoring mechanism the authors argue is *distinct* from momentum. We test both halves of the claim on **398 *current* S&P 500 names with 20+ years of history** — a survivor panel (the local `fetch_panel` requires an explicit `allow_survivorship_bias=True`), which matters doubly here because the short leg is fallen names guaranteed to have survived. Rank by nearness (price ÷ trailing-12-month high), long the near and short the far, measure the hedge and its Lo t-stat — and crucially, **correlate it with a standard 12-2 momentum hedge** (trailing year, skipping the last month) to see whether it's a separate factor at all; the correlation, unlike the level, is robust to the panel's bias. The offline control is a synthetic trending panel where nearness and momentum are both predictive and correlated (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "buy stocks near their highs" is just momentum, and why no premium showed up on large caps |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the nearness hedge with its Lo t-stat, the 0.82 correlation to 12-2 momentum, the survivor-sign caveat, the cost sweep |

The fingerprinted real-data run (398 names, 1996–2026, fp `c768f59e31fe`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic trending panel in [high_water/data.py](high_water/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
