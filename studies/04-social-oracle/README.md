# Study 04 — Social-Oracle 🔮 — does following a viral stock guru actually pay?

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md). For the desk and the house style,
> see the [methodology](../../METHODOLOGY.md). This page follows the desk's standard
> seven beats. Sibling studies: [02 — Falling-Knife](../02-falling-knife/) and
> [03 — Fear-Gauge](../03-fear-gauge/) — this is their cousin in **attention
> space**: the first study whose trigger lives in the information flow, not in the
> price or vol series.*

## Verdict — read this first

*This study ships a **complete, tested method** and an **offline synthetic** proof
that the machinery detects the effect it hunts — but, unlike Studies 01–03, it does
**not** bundle a live dataset: a mention feed is third-party, scraped, and
licence-encumbered. The stamps below are therefore the desk's **stated prior** (from
the literature + the synthetic), and the table is marked `⏳ pending a live feed`.
Drop a real CSV into [`examples/verify_real.py`](examples/verify_real.py) and the
same code fills them in. We publish the method and the prior honestly rather than a
fabricated number.*

| Axis | Stamp (prior) | Why (one line) |
|---|---|---|
| **Signal** — is the effect statistically real? | `WEAK` ⏳ | A mention plausibly carries a small, short-horizon *attention* bump — but it rides on a **run-up that already happened** (the tweet chases the move) and is hard to separate from plain momentum. |
| **Tradability** — does it survive costs, capacity, scale? | `MIRAGE` ⏳ | You read the call *after* the pop, the names are **\$1–3 micro-caps** with 50–200 bps spreads, capacity is a few thousand dollars before your own order is the move, and the abnormal path **fades** over the following weeks. |
| **Pump-and-fade?** — does the pop reverse? | `EXPECTED` ⏳ | The event-study signature the literature predicts: cumulative abnormal return rises *into* the mention, peaks early, then bleeds back — the follower buys the start of the bleed. |

> **In one sentence (the prior):** a viral guru's cashtag plausibly moves a name for a
> day or two, but that's **attention beta that reverses** — the follower, late by
> construction and trading thin micro-caps, is buying a pump already consumed and
> calling it alpha; the method here is built to measure exactly that, on whatever
> feed you bring.

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

> *Run it on a real feed and this section fills with live numbers. Until then, what
> the **offline synthetic** shows — a universe with a deliberately mild pump-and-fade
> baked in, so the test of significance still has to work for its money — confirms
> the method recovers the signature it's built to find.* (Reproduce:
> [`examples/run_synthetic_demo.py`](examples/run_synthetic_demo.py).)

- **The run-up is already in the chart.** The mean abnormal path climbs *into* t=0
  (the tweet chases a move that mostly happened) — exactly the pre-event leg a naive
  "look, it went up after" reading misses.
- **The pop fades.** Forward abnormal CAR peaks within a few sessions and then bleeds
  negative through the month — the follower, entering at the next open, is on the
  wrong side of the reversal.
- **It loses to a random day, and to momentum.** On the synthetic the mention basket
  undershoots both the random-`(name, day)` null *and* the hot-streak control at a
  week and a month — the construction the real test is built to detect.
- **Clustering widens the bar.** The calendar-block bootstrap on the synthetic keeps
  the excess where the iid test put it; on a real feed, where four hype waves can be
  one theme, it's the test that stops a meme week counting as thirty observations.
- **Costs finish it.** The micro-cap cost sweep turns the mean trade more negative at
  every step from a 5 bps half-spread to 100; capacity, via square-root impact, lands
  in the low thousands of dollars per name.

> 🔬 **For the quants** — permutation null (random baskets over the pooled abnormal
> CARs), label-permutation for the momentum gap, circular calendar-block bootstrap
> (`benchmark.py`, `robustness.py`); horizons fixed at +1d/+1w/+1m, announced before
> running. The synthetic's effect sizes are arbitrary (we chose them); what's load-
> bearing is that the **sign and the survival** behave as the method requires.
> Reproduce: [`notebooks/02_for_the_quants.ipynb`](notebooks/).

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

> *The stamps, and what will earn them. Marked `⏳` until a live feed is run.*

- **Signal — `WEAK` (prior).** The literature on attention-driven returns
  (Barber–Odean's attention-buying, the "dumb money" reversal, analyst/influencer
  event studies) says a public mention *does* move a name briefly — so we expect a
  real but small short-horizon bump. What makes it `WEAK` rather than `REAL`: it sits
  on a pre-event run-up, and the momentum control is built to strip most of it out.
  The live `p_greater`, the momentum-gap p-value and the block-bootstrap CI go here.
- **Tradability — `MIRAGE` (prior).** The follower enters late by construction, the
  abnormal path fades, the spreads are micro-cap-wide, and capacity is trivial. The
  live cost sweep and capacity figure — the dollar size where impact eats the bump —
  earn the stamp.

> 🔬 **For the quants** — the decisive cells, to be filled from a live run: random-day
> `p_greater` at 1d/1w/1m; momentum-control `gap` and `p_mention_gt_alt`;
> block-bootstrap `p_excess_le_0`; the fade-curve peak-vs-month; the name-jackknife
> swing; mean net trade across the spread sweep; capacity in USD. The synthetic run
> shows the shape; the feed sets the numbers.

## 6 · Could You Trade It?

> *The honest money question — the beat that separates this desk from a dashboard.*

The "signal ledger" screenshots show a clean line going up after each mention. The
trade that line implies does not exist for *you*, for three compounding reasons.
**Timing:** you read the post after it's public, so the entry you can actually get is
the next open — past the pop, into the fade. **Spread:** these are \$1–3 names; cross
a 50–200 bps spread on the way in and again on the way out and you've paid, round-
trip, more than a month's worth of the abnormal bump before the position moves.
**Capacity:** the square-root impact model puts the size at which your own order
erases the edge in the low single-digit thousands of dollars per name — fine for a
screenshot, useless for capital. Net of all three, the cost sweep marches the mean
trade further into the red at every spread step. That's the `MIRAGE`.

> 🔬 **For the quants** — `backtest.run` enters at `open[t+1]`, holds a fixed window,
> and charges `CostModel(half_spread_bps=25, slippage_bps=10)` — already optimistic
> for the universe; `cost_sweep` walks the half-spread 5→100 bps; `capacity` reports
> ADV$-scaled square-root impact. The realistic curve is the sleeve's, net — not the
> gross line the dashboard draws.

## 7 · Going Further

> **The door this leaves ajar.** The interesting inversion: the side of this trade
> that *might* pay is the **opposite** of the folklore. If a mention reliably triggers
> a pop-and-fade, the edge — if any survives costs — belongs to whoever is **early or
> short the fade**, not to the late follower buying the hype. That's the same shape
> as Study 03's punchline (you *sell* the fear, you don't buy it). Whether anything is
> left after micro-cap costs is exactly what this method is built to answer — bring a
> feed and find out.

- **Bring a real feed.** The whole study is one `python examples/verify_real.py
  mentions.csv` away from live numbers. Survivorship-clean feeds (timestamped at
  post time, including the calls that flopped) are the gold standard.
- **Sentiment & conviction.** The feed carries a score; does a *high-conviction*
  mention behave differently from an offhand one?
- **First-mention vs pile-on.** Is the very first time a name appears different from
  the tenth (the debounce throws the pile-on away — does it carry information)?
- **Short the fade.** Test the inverted trade directly, net of borrow and the brutal
  costs of shorting micro-caps.
- **What a contributor could PR:** a real (licence-clean) mention dataset, a
  borrow-cost model for the short leg, or a beta-estimated abnormal return to replace
  the β=1 market adjustment.

---

## How this study is laid out

| Path | What's inside |
|---|---|
| [`notebooks/01_for_the_curious.ipynb`](notebooks/) | the story + the stakes, plain language |
| [`notebooks/02_for_the_quants.ipynb`](notebooks/) | the full method: abnormal-return event study, the two nulls, the fade, costs |
| [`docs/references.md`](docs/) | sources + literature map (attention returns, the social-trading repos) |
| [`social_oracle/`](social_oracle/) | the study package: `data` · `mentions` · `eventstudy` · `benchmark` · `backtest` · `robustness` |
| [`examples/`](examples/) | [`run_synthetic_demo.py`](examples/run_synthetic_demo.py) (offline) · [`verify_real.py`](examples/verify_real.py) (bring your own feed) |

Every number is produced by [`social_oracle/`](social_oracle/), in the house style of
the shared [`../../quantlab/`](../../quantlab/) engine; `pytest` covers it in CI.
