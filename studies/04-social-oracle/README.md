# Study 04 — Social-Oracle 🔮 — does following a viral stock guru actually pay?

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md). For the desk and the house style,
> see the [methodology](../../METHODOLOGY.md). This page follows the desk's standard
> seven beats. Sibling studies: [02 — Falling-Knife](../02-falling-knife/) and
> [03 — Fear-Gauge](../03-fear-gauge/) — this is their cousin in **attention
> space**: the first study whose trigger lives in the information flow, not in the
> price or vol series.*

## Verdict — read this first

*Measured on a **real, public, reproducible** feed: **1,468** viral mention-surge
events across **182** priced names, 2021–2025, from daily r/WallStreetBets mention
counts (CC-BY-4.0 `youyanggu/yolostocks-data` — see [`docs/results_wsb.md`](docs/results_wsb.md)
and [`examples/build_wsb_feed.py`](examples/build_wsb_feed.py)). A single-guru feed
(Serenity / @aleabitoreddit) lives behind X's auth wall and can't be redistributed,
so we test the **same phenomenon on the crowd** — a purer, fully reproducible
instance. Abnormal returns are vs **SPY**; as-of 2026-06-01, price fingerprint
`1a11c294eeba`.*

| Axis | Stamp | Why (one line) |
|---|---|---|
| **Signal** — is the effect statistically real? | `NONE` | A mention has **no abnormal edge over a random day**: excess +0.08% / +0.05% / **−0.66%** at 1d / 1wk / 1mo (p≈0.23 / 0.40 / 0.94 — i.e. *negative* by a month), and the clustering bootstrap straddles zero at every horizon. Not one lucky name (jackknife is flat). |
| **Tradability** — does it survive costs, capacity, scale? | `MIRAGE` | Gross is **pure beta**: +0.72%/trade but only **+5 bps abnormal**; the **median trade is −1.3%**, net hits zero at a 25 bps spread and goes negative beyond, and the equal-weight sleeve runs **−44% with an −84% drawdown** (the 2022 pile-in). |
| **Pump-and-fade?** — does the pop reverse? | `CONFIRMED` | The month-ahead abnormal return is significantly *negative* vs a random day, the share of up-names falls to **45.7%** (vs 51.4% random), and a mention does **worse than a name that was simply already hot** (−1.06% at 1mo). The follower buys the bleed. |

> **In one sentence:** on 1,468 real WallStreetBets viral surges, buying what the
> crowd screams carries **no abnormal edge** — a tiny, insignificant one-day flicker
> that fades to a *negative* month, gross "gains" that are just market beta the costs
> erase, a median trade of −1.3%, and 42 of the most-viral names that literally
> delisted. It's a pump you're late to, dressed as a signal.

> **Not investment advice, and not about a person.** This tests a *phenomenon*
> (social trading), using a public influencer feed only as the worked example.
> Research & education. See [../../LICENSE](../../LICENSE).

---

## 1 · The Claim

A retail-investing folk hero goes viral — here, *白毛股神* **Serenity**
([@aleabitoreddit](https://twitter.com)), whose posts a wave of open-source repos now
scrape, distil into `$SYMBOL`s, and score 0–100 as tradeable "signals". The claim,
stated at full strength the way the dashboards state it:

> *"Her timeline front-runs the market. Track what she mentions, buy the cashtag, and
> ride the move — the proof is right there in the price-vs-mention chart."*

It's the oldest market story in a new costume: **a guru whose calls pay.** The
modern twist is that the "alpha" is now packaged as an installable agent skill —
[`haskaomni/serenity-signal-dashboard`](https://github.com), `serenity-skill`,
`muxuuu/serenity-skill`, `0xAgata-prog/serenity-skill` — each promising that wiring
her feed into your stack lets you "analyse the new stock-god's method and boost your
efficiency". We steelman the strongest version: *a public mention is, on average,
followed by a positive abnormal return you could have captured.*

> 🔬 **For the quants** — H₁: E[CAR_{0→h} | mentioned at t=0] > 0 with a
> clustering-robust *t* > 2, where CAR is the **abnormal** return (name minus
> market), at horizons h ∈ {1, 5, 21}. And the sharper H₁′: that excess survives a
> control for the name's *prior* momentum and for realistic micro-cap costs. Null
> H₀: the forward abnormal return is ≈ 0, or is fully explained by momentum the name
> already had.

## 2 · So What?

If a public mention pays, **attention itself is alpha** — a number anyone can read
off a social feed front-runs price, and the efficient-market story has a gaping,
free hole that anyone with an API key can climb through. That's the dream the
"signal ledger" dashboards sell.

If it *doesn't* — or if it pays for a day and then reverses — then thousands of
followers are systematically **buying negatively-skewed attention beta**: a small,
late pop with a fat fade behind it, on names too thin to exit cleanly. The deeper
lesson is the one Studies 02–03 keep circling from new angles: *a pattern can be
visible to the naked eye and still be the **opposite** of an edge once you ask "more
than a random day?", "more than the momentum it already had?", and "net of what I'd
actually pay?".*

> 🔬 **For the quants** — back-of-envelope: a dashboard's "look, it went up after she
> tweeted" must be read against (a) the name's own pre-event drift (attention follows
> performance, so the run-up is in the chart *before* t=0), (b) the ~50% base rate of
> any short-horizon abnormal return being positive, and (c) a round-trip cost that,
> on a 50–200 bps-spread micro-cap, can exceed the entire claimed monthly bump. The
> interesting quantity is the **excess**, its **sign over momentum**, and its
> **decay**.

## 3 · How We'd Know

The names a viral feed surfaces are volatile by selection — *some* are always
ripping — so a green path proves nothing. As in Studies 02–03 the question isn't
*"did it rise?"* but *"did it rise more than it should have?"* — and here that means
two controls stacked, plus the fade:

- **Excess over a random-day null**, by permutation over every `(name, day)` in the
  same universe — never absolute return, and always on **abnormal** (name-minus-market)
  returns so a small-cap-wide rally doesn't masquerade as skill.
- **Excess over a momentum control** — the cross-control unique to this study:
  hot-streak events (a name already in its top-decile trailing run). *If a mention
  doesn't beat a name that was simply already hot, the oracle is a momentum sensor.*
- **The fade curve** — mean abnormal CAR session by session. A pop that peaks early
  and reverses is the tell that the follower (who enters *after* t=0) buys the bleed.
- **Clustering bootstrap** — mentions arrive in hype *waves*; a meme week is one bet,
  not thirty. A calendar-block bootstrap gives the honest CI.
- **Name jackknife** — drop the most-mentioned name; if the excess collapses, you
  found a stock, not a skill.
- **Costs & capacity** — micro-cap spreads charged twice, and a square-root impact
  model for the dollar size at which your own order erases the edge.

And the honesty rail this study leans on hardest: **coverage.** A scraped feed is
soaked in survivorship — you hear about the calls that worked. Every mention dropped
(no price, delisted symbol, too close to the sample edge) is **counted and reported**,
never silently skipped.

> 🔬 **For the quants** — the shared desk protocol, powered by
> [`quantlab/`](../../quantlab/) and this study's [`social_oracle/`](social_oracle/):
> (1) build abnormal-return CAR paths by exact identity; (2) permutation null +
> momentum-control permutation + calendar-block bootstrap CI; (3) magnitude critique
> — pre-event drift, fade, name concentration; (4) attention-beta vs alpha via the
> momentum control; (5) micro-cap cost sweep + square-root-impact capacity;
> (6) verdict. Engine: `data`, `mentions`, `eventstudy`, `benchmark`, `backtest`,
> `robustness`.

## 4 · The Teardown

> *We ran it on 1,468 real WallStreetBets viral surges (2021–2025, 182 priced
> names, abnormal vs SPY). Reproduce: [`examples/verify_wsb.py`](examples/verify_wsb.py);
> full tables in [`docs/results_wsb.md`](docs/results_wsb.md).*

- **There's no pop worth the name.** The mention-day abnormal return is **+0.14% at
  a day, +0.32% at a week** — and *insignificant* (random-day p≈0.23 and 0.40; every
  event-study *t* < 1). A whiff of attention, nothing you could lean on.
- **And it fades to *negative*.** By a month the abnormal excess over a random day is
  **−0.66%** (p≈0.94 — the mention sits in the bottom 6% of random baskets), and the
  share of names that are even *up* falls to **45.7%**, versus **51.4%** for a random
  name-day. The crowd's pick is *less* likely to be green a month later than a coin
  flip over the same universe.
- **It loses to plain momentum.** A mention beats a name that was *already hot* by a
  hair at one day (+0.25%, p≈0.10) — then does **−1.06% worse** by a month (p≈0.97).
  Whatever the mention adds at t+1 is momentum the name already carried, and it
  reverses harder than the momentum alone.
- **Clustering kills what's left.** The calendar-block bootstrap (hype arrives in
  waves) puts the 1-week excess at **+0.06%, 95% CI [−0.65%, +0.82%]**, p(excess≤0)≈
  **0.45** — dead zero — and the 1-month excess at **−0.67%, CI [−1.94%, +0.62%]**.
  Nothing clears the bar at any horizon.
- **It's not one lucky name.** The jackknife is flat: drop INTC, NFLX, AMD, PLTR or
  AAPL and the conditional mean barely moves (0.0024–0.0038 vs 0.0032). The *nothing*
  is broad, not a single 10-bagger hiding the result.
- **Survivorship makes it worse, not better.** **42 of the most-viral names had no
  tradeable price** — they delisted or were renamed (SIVB, FRC, TWTR, NKLA, RIDE,
  ATVI…). The sample we *can* price is biased toward the survivors, and it *still*
  shows no edge.

> 🔬 **For the quants** — permutation null (2,000 random baskets over the pooled
> abnormal CARs), label-permutation for the momentum gap, circular calendar-block
> bootstrap (`benchmark.py`, `robustness.py`); horizons fixed at +1d/+1w/+1m,
> announced before running. β=1 market adjustment vs SPY; daily returns winsorized at
> ±100% to kill reverse-split / bad-print artefacts in filthy micro-cap data (a
> stated decision — `data.build_panel(clip_daily=1.0)`). Reproduce:
> [`examples/verify_wsb.py`](examples/verify_wsb.py).

<details>
<summary>🔬 The maths, in full</summary>

Abnormal return aₜ = r_cc,ₜ − r_mkt,ₜ (market-adjusted, β fixed at 1 — the standard
short-window event-study choice, robust on thin names where estimating β is noise).
The event path is the additive cumulative abnormal return CAR(t→t+k) = Σ a over the
window, centred so CAR at t=0 is 0; the pre-leg (k<0) is the run-up, the post-leg
(k>0) is the fade. The random-day null permutes the event mask over every valid
`(name, day)` CAR in the universe; the momentum control permutes the label over the
union of mention and hot-streak events, so its null is "mentions and pre-existing
streaks draw forward returns from the same distribution". The block bootstrap
resamples contiguous calendar blocks of `(CAR, is_event)` to preserve hype-wave
clustering. Capacity solves impact_bps(N) = c·10⁴·√(N/ADV$) = edge_bps for N.

</details>

## 5 · The Verdict

> *The two stamps, and the numbers that earned them.*

- **Signal — `NONE`.** A crowd mention does not beat a random day at any horizon.
  The excess is **+0.08% / +0.05% / −0.66%** at 1d/1w/1mo, with random-day p≈**0.23 /
  0.40 / 0.94** (the month is significantly *negative*), the clustering bootstrap
  straddles zero throughout (p(excess≤0)≈0.45 at a week), and the jackknife is flat.
  My going-in prior was `WEAK` — the data was less kind than that. The only thing the
  mention reliably predicts is a *below-average* month.
- **Tradability — `MIRAGE`.** The backtest's gross +0.72%/trade is **pure beta** —
  the abnormal piece is **+5 bps**. The **median trade is −1.3%**, the mean net hits
  zero at a 25 bps spread and goes negative beyond it, and the equal-weight follower's
  sleeve compounds to **−44% with an −84% drawdown** (everything mentioned crashed
  together in 2022). You took meme-stock risk to earn the market's return, then gave
  it to the spread.

> 🔬 **For the quants** — decisive numbers in one place: random-day p≈0.23/0.40/0.94
> (1d/1w/1mo); momentum-gap p≈0.10→0.97; block-bootstrap 1wk excess +0.06% [−0.65%,
> +0.82%] and 1mo −0.67% [−1.94%, +0.62%]; fade-curve `pct_positive` 0.469→0.457;
> backtest mean-abnormal +0.0005, median net −0.0134, sleeve Sharpe ≈0, max drawdown
> −0.84. All from [`docs/results_wsb.md`](docs/results_wsb.md), as-of 2026-06-01,
> price fingerprint `1a11c294eeba`.

## 6 · Could You Trade It?

> *The honest money question — the beat that separates this desk from a dashboard.*

The dashboards show a clean line going up after each mention. The trade that line
implies does not exist for *you*. **Timing:** you read the post after it's public, so
the entry you can actually get is the next open — past the (already insignificant)
pop, into the fade. **No edge to protect:** the abnormal return is **+5 bps**; the
mean net trade is already zero at a 25 bps spread, and the median trade is **−1.3%**
before you pay a cent. **The lived path:** held equal-weight, the sleeve runs −44%
with an −84% drawdown, because the surges cluster in exactly the regimes (2022) where
everything mentioned falls at once. That's the `MIRAGE`.

One honest twist the real feed surfaces: **capacity is *not* the binding constraint
here.** The crowd's most-viral names are mega-caps (NVDA, TSLA, AMD…), so median
dollar-volume is ~\$1B and square-root impact only bites past ~\$5M per trade. That's
the opposite of a thin single-guru micro-cap feed, where capacity *is* the killer.
For the WSB crowd the trade doesn't die of illiquidity — it dies because **there's no
abnormal return to begin with**, and what little gross you book is market beta the
spread takes back.

> 🔬 **For the quants** — `backtest.run` enters at `open[t+1]`, holds 10 sessions,
> charges `CostModel(half_spread_bps=25, slippage_bps=10)`; `cost_sweep` walks the
> half-spread 5→100 bps (mean net +0.40% → −1.50%); `capacity` reports ADV$-scaled
> square-root impact (median ADV ≈ \$1.0B → ~\$5.2M/trade at a 71 bps gross). The
> realistic line is the sleeve's, net — not the gross line the dashboard draws.

## 7 · Going Further

> **The door this leaves ajar.** We measured it as a *negative* — the crowd's pick
> underperforms a random day a month out. So the side that *might* pay is the
> **opposite** of the folklore: being **early or short the fade**, not the late
> follower buying the hype. That's the same inversion as Study 03 (you *sell* the
> fear, you don't buy it). Whether the short leg survives borrow and the brutal cost
> of shorting meme names is the next study, not this one.

- **The single-guru feed.** We ran the *crowd* (reproducible, CC-BY); a clean
  Serenity/@aleabitoreddit feed — timestamped at post time, including the calls that
  flopped — is the obvious next dataset, and the one place capacity might actually
  bind (thin names). Anyone can run it: `python examples/verify_wsb.py` is the
  template; point `data.load_feed` at any `(timestamp, ticker)` CSV.
- **Short the fade.** Test the inverted trade directly, net of borrow.
- **Sentiment & conviction.** The feed carries a mention count; does a *louder* surge
  behave differently from a quiet one?
- **First-mention vs pile-on.** Is a name's first-ever surge different from its tenth
  (the debounce throws pile-ons away — do they carry information)?
- **What a contributor could PR:** a licence-clean single-influencer feed, a
  borrow-cost model for the short leg, or a β-estimated abnormal return to replace the
  β=1 market adjustment.

---

## How this study is laid out

| Path | What's inside |
|---|---|
| [`notebooks/01_for_the_curious.ipynb`](notebooks/) | the story + the stakes, plain language |
| [`notebooks/02_for_the_quants.ipynb`](notebooks/) | the full method: abnormal-return event study, the two nulls, the fade, costs |
| [`docs/results_wsb.md`](docs/results_wsb.md) | **the real run** — every headline table, fingerprinted and as-of'd |
| [`_data/wsb_mentions.csv`](_data/) | the real feed: 1,705 WSB viral-surge events (CC-BY, see [`_data/PROVENANCE.md`](_data/PROVENANCE.md)) |
| [`docs/references.md`](docs/) | sources + literature map (attention returns, the social-trading repos) |
| [`social_oracle/`](social_oracle/) | the study package: `data` · `mentions` · `eventstudy` · `benchmark` · `backtest` · `robustness` |
| [`examples/`](examples/) | [`build_wsb_feed.py`](examples/build_wsb_feed.py) (build the feed) · [`verify_wsb.py`](examples/verify_wsb.py) (the real run) · [`run_synthetic_demo.py`](examples/run_synthetic_demo.py) (offline) · [`verify_real.py`](examples/verify_real.py) (any feed) |

Every number is produced by [`social_oracle/`](social_oracle/), in the house style of
the shared [`../../quantlab/`](../../quantlab/) engine; `pytest` covers it in CI.
