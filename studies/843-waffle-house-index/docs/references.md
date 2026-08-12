# References & literature map — Study 843 (Waffle House Index)

## The claim under test

- **The folklore anchor — the "Waffle House Index".** The chain's near-total refusal
  to close made it a de-facto disaster gauge; FEMA administrator W. Craig Fugate
  popularised the informal **Waffle House Index** (green = full menu / mild, yellow =
  limited menu / serious, red = closed / catastrophic) as a fast, on-the-ground read of
  a storm's severity when official data lag. Widely reported (Wall Street Journal 2011,
  "Waffle House Index"; FEMA blog posts; countless hurricane-season explainers). It is a
  *severity* signal, not a market signal — this study asks the market question it
  implies.
- **The tradable turnaround, steelmanned.** If a storm is severe enough to shut a
  Waffle House, it is severe enough to matter to the two most obviously exposed listed
  sectors: **property & casualty insurers** face a claims/payout shock (their stocks
  should *dip*), and **home-improvement / rebuild** names face a reconstruction-demand
  tailwind (their stocks should *rally*). The sharpest single prediction is directional:
  a **long-rebuilders / short-insurers** book should earn a positive abnormal spread in
  the weeks around a major landfall.
- **Why the prior is that this is already priced.** Hurricanes are **forecast days
  ahead** (unlike a bank failure or a plane crash, which are surprises), so an efficient
  market can price the expected loss into insurers and the expected demand into
  rebuilders *before* landfall; insured-loss estimates are also public within days. The
  honest question for this desk is therefore not "is a hurricane bad for insurers" (of
  course a specific quarter's earnings take a hit) but "is there a *tradable,
  post-landfall abnormal move* left on the modern tape, in the obvious large-cap names,
  once anticipation has done its work" — and whether ~16 events carry enough power to
  detect it.

## What we measure, and the honesty rails

- **Market-adjusted abnormal return** (Brown & Warner 1985's market-adjusted model): a
  name's daily simple return minus SPY's same-day return (an implicit beta of 1). Over a
  short event window this cheap proxy is standard and robust; it strips the common
  market move so a post-storm CAR is a *sector-specific* signal, not "stocks rose that
  fortnight". Baskets are equal-weight of the market-adjusted legs.
- **Per-event CAR over [+0..+20] sessions**, one-sample *t* across the 16 independent,
  far-apart landfall dates (the planned primary; events weeks-to-years apart, so no HAC
  correction is needed the way a daily-panel regression would — a Newey-West *t* is
  reported anyway and auto-falls-back to a plain *t* below n = 8).
- **The directional test** — a *paired* per-event (rebuilders − insurers) CAR, the
  folklore's cleanest one-number prediction; pairing removes any residual common shock.
- **Event window [−10..+20]** with each offset's own *t*, read honestly as a **31-offset
  multiple-comparison** exercise — roughly 1–2 offsets crossing |*t*| ≥ 2 by chance is
  *expected* and is called out as such (the pre-storm −5-session insurer bar).
- **Hit rate carries a Wilson (1927) interval**; the placebo is a 20-seed × 1,000-draw
  random-calendar null (the same falsification design as `707-plane-crash-effect` and
  `316-bank-failure`); an event-resampled bootstrap gives the CI.
- **Anticipation & power named, not hidden.** The window spans the pre-storm run-up
  because forecasts precede landfall; and n = 16 is explicitly flagged as low power,
  with a catastrophic-only (tier-3, n = 7) and a two-era robustness cut so the reader can
  see the one nominally significant subgroup does not generalise.

## Why the timer is graded separately, and the execution convention

- The tradable overlay — long rebuilders / short insurers, entered at the landfall-session
  close, held a few sessions — is tested purely as a **statistical falsification**: if the
  dip/rally claim were real and tradable, this dollar-neutral book should show a positive,
  cost-surviving edge. It does not (every horizon loses). Costs are one-way × NAV on all
  four legs (long + short, enter + exit) plus 50 bps/yr borrow on the short insurer leg.
- **Execution lag.** Landfall is calendar-known and largely forecast, so the entry uses
  the first tradable NYSE session on/after the landfall date with no extra shift — the
  study's single documented convention; see `docs/results.md`.

## Data sources

- **SPY, ALL, TRV, PGR, HD, LOW** daily total-return closes (`auto_adjust=True`) —
  yfinance (no key), cached under `_cache/` (`whi_spy.csv`, `whi_all.csv`, `whi_trv.csv`,
  `whi_pgr.csv`, `whi_hd.csv`, `whi_low.csv`), 2004-06-01 → 2026-06-30.
- **16 hardcoded major US hurricane landfalls, 2005 → 2024**, in
  [`waffle_index/data.py`](../waffle_index/data.py). Landfall dates are National
  Hurricane Center / NOAA public record (each storm's Tropical Cyclone Report); the loss
  tiers are coarse public insured-loss buckets. No free machine-readable "US disaster
  index" keyed to market days exists, so this is a hand-built table of the storms any
  reasonable person would call the market's front page that week.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [283-hurricane-season](../../283-hurricane-season/) — the **seasonal** hurricane
  calendar (is there a tradable pattern to the *June–November season itself*), a
  calendar-anomaly question. This study is an **event study of specific major
  landfalls**, not a seasonal-window bet.
- [316-bank-failure](../../316-bank-failure/) — the same **event-study machinery** on a
  hardcoded table of *bank-failure* dates (a financial-contagion shock, and a *surprise*
  rather than a forecast event).
- [313-geopolitical-shock](../../313-geopolitical-shock/) — wars, invasions and terror
  attacks (a geopolitical-sentiment shock calendar); different trigger, different
  exposed sectors.
- [707-plane-crash-effect](../../707-plane-crash-effect/) — the closest cousin: an event
  study of a **disaster calendar** (aviation) testing a market-wide sentiment dip and a
  sector (airline) extra-drop. This study swaps the trigger (natural disaster) and the
  exposed sectors (insurers dip / rebuilders rally) and uses a market-adjusted rather
  than constant-mean abnormal return.

None of the siblings test **what a major US hurricane does to listed P&C insurers and
home-improvement rebuild names specifically** — that is this study's own axis.
