# Study 549 — Spotify-Mood 🎧

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does streaming valence lead the market? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | **No real valence tape exists** — Spotify closed the audio-features API to new apps (Nov 2024) and never exposed a clean historical chart-valence panel, so the mood series is **synthetic** and can never clear the REAL bar. On the honest test (plausible-but-null mood proxy × real ^GSPC, 196 months) the lag-1 slope is the **wrong sign** and insignificant: HAC *t* **−1.25**, placebo *p* **0.231**, hit-rate **46.7%** vs a **66.2%** base rate. |
| **Tradability** — does the mood-timing rule pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The long-only "long when last month's valence is above median" rule earns **+4.7%/yr** while the market did **+13.2%/yr** over the same live months (it sits out the rallies it can't foresee); the long-short variant is **negative** (−3.3%/yr gross, −4.0% net after 5 bps/leg + 100 bps borrow). Nothing to harvest. |
| **"Music sentiment on the tape?"** | ![Untestable](https://img.shields.io/badge/Untestable-8b949e?style=flat-square) | The academic result (Edmans et al. 2022) is real *on a licensed proprietary valence dataset*. On a free retail stack the claim is **untestable** — there is no reconstructible real valence tape — so we prove the *engine* is faithful on synthetic worlds instead. |

> **In one sentence:** "the mood of the songs people stream leads the stock market" is a lovely alt-data story that a retail desk simply **cannot test** — Spotify shut the only public valence source, so the mood tape must be synthetic; run the honest test with a plausible synthetic mood proxy against the real S&P and you get a wrong-signed, insignificant lag-1 HAC *t* (−1.25), a placebo *p* of 0.23, a below-coin hit-rate, and a market-timing rule that trails buy-and-hold by ~8 pts/yr.

## What we tested

The **music-sentiment** claim (Edmans, Fernandez-Perez, Garel & Indriawan 2022, *Music Sentiment
and Stock Returns Around the World*): the aggregate musical **valence** (Spotify's happy↔sad audio
feature) of the top-streamed songs is a mood proxy that co-moves with — and might *lead* — the
market. Because the only public valence source (Spotify's audio-features API) was **closed to new
apps in November 2024** and never gave a clean historical panel, the **mood series is synthetic**
(a seeded AR(1) valence proxy) — a hard limitation stated on the Signal axis. We join it to **real**
monthly ^GSPC returns (2010-02 → 2026-05, 196 months) and run the honest predictive test: a
Newey-West **HAC regression** of next-month return on lagged valence, a **lag-1..5 sweep** (the
Granger multiple-comparisons trap with a Bonferroni bar), a **circular-shift placebo** null, a
directional **hit-rate vs the honest base rate**, a gross/net **mood-timing rule** (one-month
execution lag, costs × NAV, shorts pay borrow), and a **seed-robust synthetic positive control**
(≥ 25 seeds) that plants a genuine predictive edge and proves the engine catches it (past *t* +2 by
`predictive_beta` ≈ 0.007) while reading ~0 at the null. *Distinct from the social-media mood proxy
in [256 Twitter-Mood](../256-twitter-mood/) and the match-result proxy in
[300 Sports-Sentiment](../300-sports-sentiment/) — this is the **music-valence** proxy, forced
synthetic by the closed Spotify API.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "musical valence" is, why happier charts *might* mean a happier market, why we can't get the real data, and why a plausible mood series predicts nothing here |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC predictive regression, the lag-sweep / Bonferroni trap, the circular-shift placebo, the hit-rate vs base rate, the gross/net mood-timing rule, and the seed-robust synthetic positive control |

The fingerprinted run (real ^GSPC market fp `431d740a90aa`, synthetic mood, joined panel fp
`1e087c2e441e`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery
proof runs on the deterministic synthetic world in [`spotify_mood/data.py`](spotify_mood/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`spotify_mood/`](spotify_mood/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
