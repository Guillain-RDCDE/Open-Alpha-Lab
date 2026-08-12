# Study 847 — Rotten-Tomatoes -> Studio 🍅

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a film's critic tier (fresh/rotten) move the distributing studio? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The **opening-weekend** window — where a pre-released critic verdict should first hit — shows **no gap** (fresh − rotten = −11.9 bps, Welch *t* = **−0.12**). A right-signed, significant gap appears only in the **following-week** `[+2..+6]` window (**+815.7 bps, Welch *t* = +2.58**, permutation *p* = **0.001**, robust to sub-era & leave-one-studio-out) — but it fails every credibility check: no direct-window reaction, a single transient window that reverses after, a pooled magnitude no bigger than random weeks (*p* = 0.079), and a ~6% weekly conglomerate move that is economically implausible and driven by studio-idiosyncratic news (Paramount's 2024 takeover crashes). A real in-sample association, **not attributable to reviews**. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | A long-fresh / short-rotten studio book survives naive costs in-sample (net **+386 bps/leg**, *t* = +2.50, borrow-feasible on large caps) — but it is a 40-trade backtest of a window-selected, non-causal gap: an artifact of the Weak signal, not a deployable edge. |
| **Does a flop tank the studio?** | ![Overstated](https://img.shields.io/badge/Flop%20tanks%20studio%3F-Overstated-8b949e?style=flat-square) | A rotten score *lines up* with a soft following week here, but there is zero opening-weekend reaction, the move is a single transient window, its size is implausible for a conglomerate, and it is inseparable from the studios' ambient fortunes. The causal "the flop tanked the stock" story is not supported. |

> **In one sentence:** across **40 major 2022-2025 wide releases** tagged by their distributing studio (DIS/WBD/PARA/CMCSA/NFLX/SONY) and a coarse Rotten-Tomatoes tier, the studio's **opening-weekend** abnormal return does **not** separate fresh from rotten (Welch *t* = −0.12); a right-signed **following-week** gap does clear the bar (Welch *t* = +2.58, permutation *p* = 0.001) but is a transient, single-window, economically-implausible association inflated by unrelated studio news (Paramount's takeover crash) — a **Weak**, non-causal footprint of "declining studios make both worse films and worse stocks," not proof that a flop tanks the stock.

## What we tested

The folklore, in both directions: *a rotten-scored flop tanks the studio* and *a
fresh-scored hit pops it*. We steelman it as a tier-conditioned corporate event study
(Brown & Warner 1985) on a hand-curated table of **40 major wide releases, 2022→2025**,
each tagged with its **distributing-studio ticker** (Disney DIS, Warner Bros. Discovery
WBD, Paramount PARA, Comcast/Universal CMCSA, Netflix NFLX, Sony SONY) and a coarse public
critic **tier** (fresh ≥ 75 / rotten < 50; mixed 50-74 excluded). We measure the studio's
market-adjusted (studio − SPY, demeaned) abnormal return in two pre-registered windows —
**opening-weekend `[0..+1]`** (the direct test) and **following-week `[+2..+6]`** — and
test the fresh-minus-rotten gap with a Welch *t*, a 20k-draw tier-label permutation
placebo, a random-date placebo, Wilson hit rates and a costed long-short timer. A
deterministic synthetic tape with a *planted* tier drift is the positive control. **As-of
2026-06-30.** **Dedup:** distinct from [771-box-office-bomb](../771-box-office-bomb/)
(single-ticker DIS flop/write-down event, not a cross-studio critic-tier split),
[550-box-office-momentum](../550-box-office-momentum/) (box-office *revenue* momentum, not
reviews), [296-oscars-effect](../296-oscars-effect/) (the *awards* channel) and
[552-app-store-rankings](../552-app-store-rankings/) (product ratings for tech names) —
none test whether a film's Rotten-Tomatoes tier moves its *distributing studio*.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a flop *should* move the studio if the story is right, why the opening-weekend window shows nothing, and why the one window that does light up is not what it looks like |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the two-window event study, the Welch fresh-minus-rotten gap, the tier-label vs random-date placebos (and why they disagree), the window-by-window transience, the Paramount-crash confound, the costed timer, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`rotten_tomatoes/`](rotten_tomatoes/). The film/tier calendar is hand-curated
from Rotten Tomatoes (Tomatometer), Box Office Mojo (opening date + distributor) and trade
press; SPY and the six studio tickers are fetched via yfinance. WBD/PARA ticker-existence
and Netflix's streaming-premiere (not opening-weekend) event type are named honestly. **Not
investment advice** — research & education. See [LICENSE](../../LICENSE).*
