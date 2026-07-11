# Study 694 — Matching Low 🟥🟥

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a repeated close reverse the decline? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Entered next open, one lag, measured **fairly** against the basket's own unconditional forward return (Welch *t*, the decisive number) — the forward return **never certifies a reversal** at 1/5/10/20 days (**−0.03 / −0.16 / −0.48 / −0.40**), all inside noise. A random-draw placebo puts every horizon at **p ≥ 0.5**. **0/30 tickers** survive a Bonferroni correction individually. Neither a near-exact close match nor a genuine prior-downtrend filter rescues it (best reading +1.28, still short of the bar). The **vs-zero** *t* climbs to **+10.55** by 20 days — that's the basket's own up-drift, not the pattern; the gap between the two statistics *is* the finding. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Gross/net turn nominally positive from 5 days out (**+0.15% to +0.95%** net at 5 bps), but they track the unconditional base almost bar-for-bar — no **excess** to charge costs against, just the basket's ordinary beta wearing a pattern's name. At 1 day, net of even 5 bps, the trade is already negative. |
| **"Marks support?"** | ![Busted](https://img.shields.io/badge/Marks_support%3F-Busted-8b949e?style=flat-square) | Two down candles closing at the same price look like a mini double bottom, but the forward return carries **no information** beyond the market's own drift — and testing only the "purer" versions (a near-exact close tie; a genuine downtrend behind the pattern) does not sharpen it. |

> **In one sentence:** the matching low — two down candles closing at (about) the same
> price, "a double-bottom in miniature" — fires **~4,600 times** across 30 names + SPY
> over 21.5 years, and once measured fairly against what the same basket earns on an
> ordinary day (not against zero, which is just the market's own up-drift dressed up as a
> pattern), the forward return **never certifies a reversal** at any of 4 horizons
> (Welch *t* from −0.03 to −0.48), a random-draw placebo says **half of random picks beat
> it**, **0/30 tickers** survive a Bonferroni correction, and neither a stricter close-tie
> nor a genuine prior decline saves it: a post-hoc chart label, not a support signal.

## What we tested

We encode the tightest mechanical version a proponent would accept on a fixed **30-name
liquid US large-cap + SPY** basket (yfinance daily OHLCV, 2005→2026, 162,180 bars). A
**matching low** is two consecutive **down** candles (`close < open`) whose closes land
within a **0.15%** relative tolerance of each other (the default, loose match a realistic
scanner would accept — exact ties almost never occur on continuous prices; a **0.03%**
strict tolerance is tested as a myth-check). We wait for the confirming close, enter the
**next open** (one execution lag), and measure the forward **1 / 5 / 10 / 20-day**
return. The pattern is **long-only** — it makes no short-side claim — so the Signal
axis's decisive number is a **Welch *t*** against the basket's own **unconditional**
forward-return pool (what an always-long trader earns on the same names/window), not a
comparison to zero (which the synthetic control shows is contaminated by the tape's own
drift), plus a hit-rate-vs-base-rate and a **5,000-draw random-draw placebo**. A
**Bonferroni correction across the 30-ticker basket** checks whether any single name
quietly carries the effect; Tradability charges a 5/10-bps round-trip cost. A second
myth-check restricts events to those following a **genuine prior downtrend** (neither
filter helps). A deterministic synthetic control with a *planted* post-pattern reversal
confirms the engine would catch a real one (it lights up at *t* = 8.3–13.0) and that zero
edge cannot fake significance under the fair comparison (1/20 null seeds fire, in line
with the ~5% nominal false-positive rate of a two-sigma test), so the negative real-tape
result is a genuine "nothing there". Survivorship — the basket excludes firms whose
decline never reversed and later delisted — is named on the Signal axis. **Dedup:**
[409-tweezer-tops-bottoms](../409-tweezer-tops-bottoms/) tests matching **wick lows**
with no color requirement (a looser, wick-level claim); [460-counterattack-lines](../460-counterattack-lines/)
requires the **second candle to be the opposite color** (the reversal already visible
inside the pattern, not just implied for afterward); [696-double-bottom](../696-double-bottom/)
is the **macro**, multi-week swing-chart version of the same "tested the same price
twice" idea, not a two-*candle* micro pattern. None of them run this study's specific
two-consecutive-down-candles, matching-**close**, long-only, Bonferroni-corrected bar.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a matching low is, why comparing to "zero" overstates the case, why the pattern doesn't beat an ordinary day, and why no stricter recipe saves it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the precise detector, the vs-zero-vs-vs-base direction trap shown live, the drift-neutral Welch-*t* design, the random-draw placebo, the Bonferroni correction across the basket, costs, the tolerance & prior-downtrend myth checks, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`matching_low/`](matching_low/). Detector is two down candles with closes
matching within a stated tolerance (loose default + strict variant for the myth-check).
Basket is **survivors** — named on the Signal axis. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
