# References & literature map — Study 847 (Rotten-Tomatoes -> Studio)

## The claim under test

- **The folklore.** Every wide release brings the same market-day chatter: a
  rotten-scored flop is a nine-figure humiliation that should *tank the studio*, and a
  fresh-scored hit should *pop it*. The steelman is an information story — a film's
  critical reception (proxied by the Rotten Tomatoes **Tomatometer**, the single most-cited
  public critic aggregate) is a public, dated signal about a studio's product quality and
  near-term box office, so the distributing studio's stock should move with it around the
  release.
- **The event-study anchor.** This is a standard corporate event study (Brown & Warner
  1985, *Journal of Financial Economics*, "Using daily stock returns"): define a clean,
  dated event, measure the security's **abnormal return** (here market-adjusted, studio −
  SPY) in windows around it, and test whether the cross-event mean differs from zero — and,
  conditioned on the tier, whether fresh and rotten releases differ.
- **The academic prior points to a null.** The entertainment-finance literature on how
  film outcomes map to studio equity is thin and weak precisely because modern studios are
  diversified conglomerates: Disney's parks/ESPN/streaming, Comcast's broadband/NBCU,
  Sony's PlayStation/sensors/music, Warner Bros. Discovery's cable/streaming debt load,
  Paramount's networks, Netflix's subscriber engine — for each, any single film is a tiny
  revenue sliver, and box office is partly anticipated (tracking, pre-sales) well before
  the weekend prints. Efficient-markets logic says a known, largely-anticipated,
  small-magnitude event should barely register at the parent level.

## What we measure, and the honesty rails

- **Tier, not score.** The Tomatometer is coarsened to two unambiguous buckets — **fresh
  (≥ 75)** and **rotten (< 50)** — with mixed 50-74 titles excluded so the contrast is
  clean. The analysis reads only the bucket; the stored score is reference-only. This
  deliberately avoids over-fitting a noisy exact number and matches how the folklore is
  actually stated ("it got panned" vs "critics loved it").
- **Two pre-registered windows.** The **opening-weekend `[0..+1]`** window is the direct
  test (reviews lift embargo *before* a wide release, so the tier is public at the open;
  the weekend gross is public by the Monday after). The **following-week `[+2..+6]`**
  window is the "slow-digest" test. Pre-registering both means the following-week result is
  one of two planned looks, not a mined window — and we report the direct window first.
- **The right statistic for independent events.** Releases are independent, non-overlapping
  calendar dates, so the primary is a **one-sample *t*** per tier and a **Welch *t*** on the
  fresh-minus-rotten gap (the quantity the claim predicts positive). A Newey-West *t* is
  reported as a cross-check; hit rates carry **Wilson (1927)** intervals.
- **Two nulls, because a lone *t* is not enough.** A **tier-label permutation** placebo
  (20 seeds × 1,000 draws) shuffles fresh/rotten labels across events, preserving each
  event's CAR — it isolates whether the *split* is unusual. A **random-date** placebo
  (20 seeds × 200 draws) redraws pseudo-events on each studio's own tape — it isolates
  whether the *pooled magnitude* is unusual. The two disagree here (split significant,
  magnitude not), which is the fingerprint of a **confound correlated with tier** rather
  than a release-driven move — see `docs/results.md`.
- **Coverage & event-type named, not hidden.** WBD trades only from 2022-04-11 and PARA is
  the post-2022-02 Paramount Global re-brand, so every WBD/PARA title is dated after its
  ticker existed. **Netflix has no theatrical opening weekend** — its event is the
  streaming-premiere date, a genuinely different event type pooled with the theatrical
  majors and flagged as such. The two largest rotten "drops" (Paramount's *IF* and *Bob
  Marley: One Love*) are the 2024 Skydance-takeover crashes, not film effects — named
  explicitly in the teardown.

## Why the timer is graded separately

- The long-fresh / short-rotten studio overlay is tested purely as a **falsification
  exercise**: if the tier signal were a real, tradable edge, this book should show a
  positive, cost-surviving return that is *robust*, not window-selected. It survives naive
  costs in-sample (net +386 bps/leg, *t* = +2.50) but rides the same single, transient,
  non-causal window — a **Fragile** backtest artifact, not a deployable strategy. Costs are
  one-way × NAV per leg (5 bps) plus 50 bps/yr borrow on the short (rotten) legs; the entry
  convention (release date snapped to the first tradable session) is the single documented
  execution lag.

## Data sources

- **SPY + DIS / WBD / PARA / CMCSA / NFLX / SONY** daily total-return closes
  (`auto_adjust=True`) — yfinance (no key), cached under `_cache/` (`rt_spy.csv`,
  `rt_dis.csv`, …), 2021-06-01 → 2026-06-30.
- **40 hardcoded major wide releases, 2022 → 2025**, in
  [`rotten_tomatoes/data.py`](../rotten_tomatoes/data.py). No free, machine-readable panel
  of (release + Tomatometer + distributor ticker) exists, so this is a hand-built table
  cross-referenced against **Rotten Tomatoes** (Tomatometer, rottentomatoes.com), **Box
  Office Mojo** (opening date + distributor, boxofficemojo.com) and contemporaneous trade
  press (Variety / The Hollywood Reporter / Deadline). Public record.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [771-box-office-bomb](../../771-box-office-bomb/) — "sell **Disney** after one of *its*
  movies flops": a single-ticker (DIS) event study keyed on the **flop / write-down**
  event, not on a cross-studio **critic-tier** split. 847 conditions on the Rotten
  Tomatoes tier across **six** distributors and asks whether *fresh vs rotten* separates.
- [550-box-office-momentum](../../550-box-office-momentum/) — a **box-office-revenue**
  momentum/anticipation signal, not a **critic-reception** signal; revenue, not reviews.
- [296-oscars-effect](../../296-oscars-effect/) — the **awards** channel (Oscar
  nominations/wins) as a prestige/attention event, a different reception signal at a
  different point in a film's life (awards season, not opening weekend) and a different
  mechanism (prestige, not opening-weekend quality news).
- [552-app-store-rankings](../../552-app-store-rankings/) — a **product-ranking /
  consumer-rating** alt-data signal for tech names, the same "public rating → parent
  stock" question in an entirely different industry (apps, not films) and rating system.

None of the siblings test whether **a film's Rotten-Tomatoes tier moves its distributing
studio's stock around the release** — that is this study's own axis.
