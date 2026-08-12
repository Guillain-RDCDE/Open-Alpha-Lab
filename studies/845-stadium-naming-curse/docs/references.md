# References & literature map — Study 845 (Stadium Naming-Rights Curse)

## The claim under test

- **The folklore.** A persistent piece of markets lore holds that a company splashing
  out on expensive stadium naming rights is flashing a **top signal**: managerial
  hubris and peak earnings spent on a vanity trophy rather than the core business, after
  which the sponsor underperforms — sometimes spectacularly. The legend is anchored on a
  handful of vivid blow-ups: **Enron Field** (Houston Astros, named 1999; Enron bankrupt
  December 2001), the **MCI Center** (Washington; WorldCom's accounting fraud, bankrupt
  2002), and the modern crypto pair — the **FTX Arena** (Miami Heat, 2021; FTX collapsed
  November 2022) and **Crypto.com Arena** (2021; deep 2022 crypto-winter cuts). The
  honest question this study asks is whether the "curse" is **systematic** across dated,
  publicly-traded deals or a **cherry-picked** few survivors of hindsight.

- **The steelman mechanism.** There is a serious version of the hypothesis:
  corporate-finance research on **free-cash-flow agency costs** (Jensen 1986) and on the
  long-run underperformance of **over-investing, empire-building firms** predicts that
  conspicuous, low-ROI trophy spending correlates with subsequent underperformance. A
  20-year naming-rights contract worth tens or hundreds of millions is a highly visible
  such commitment, often struck when a sponsor's cash and confidence are near a peak
  (the "peak-earnings signaling" reading). Marketing studies that measure the sponsor's
  *announcement-day* stock reaction (Clark, Cornwell & Pruitt on naming-rights
  announcements) find small, mixed effects; this study instead measures the **long-run
  (1–2 year) forward** abnormal return, which is where the hubris story lives.

- **What this study does.** We hand-curate **34** major naming-rights deals with
  announcement dates and sponsor tickers, flag the **5** untradable cautionary tales
  (private or delisted-into-bankruptcy sponsors — Enron, WorldCom, FTX, Crypto.com,
  SoFi) honestly and exclude them from the return test, and measure each *listed*
  sponsor's **buy-and-hold abnormal return vs SPY** over the 1- and 2-year windows after
  its deal, cross-sectionally.

## What we measure, and the honesty rails

- **Buy-and-hold abnormal return (BHAR).** (sponsor total return) − (SPY total return)
  over the post-deal window (Barber & Lyon 1997, *Journal of Financial Economics*, on
  the properties of long-horizon BHAR) — the standard long-horizon event-study statistic,
  cleaner than summing daily abnormal returns over one-to-two years. The sponsor is
  entered at the close of the first NYSE session **on/after** the announcement date
  (searchsorted snap — the single documented execution lag; the deal is public by that
  close, so zero look-ahead).
- **Cross-event inference.** A one-sample *t* across deals (the primary; deals are
  distinct names on mostly non-overlapping dates), a **Newey-West** HAC *t* as a
  calendar-clustering-robust cross-check (several deals share an era's market weather),
  and a **Wilson (1927)** interval on the hit rate.
- **Sub-era robustness.** The desk's Real bar requires an effect that holds **before and
  after 2010**, not one driven by a single era; we report the split explicitly. This
  study's 1-year effect fails it (pre-2010 *t* = −0.84), which is *why* it is stamped
  Weak, not Real.
- **Random-entry placebo.** Keep the same tickers but read each deal's BHAR from a
  **random pseudo-announcement date** on that ticker's own tape, thousands of times —
  preserving each name's own return distribution and the sample size while breaking the
  deal→outcome link. A real curse must put the observed cross-event mean in the **left**
  tail (the same falsification design as the sibling event studies 160 and 707).
- **Survivorship, named on the Signal axis.** The tradable sample **excludes the
  sponsors that went to zero** (Enron, WorldCom) — there is no free full-path tape for a
  delisted-into-bankruptcy stock. The measured curse is therefore an *understatement* of
  the folklore's worst cases; that the *survivors* still underperform is stated as the
  honest, against-the-grain finding.

## Why the overlay is graded separately, and its cost model

- The tradable read — **short the sponsor, long SPY** for the window — is graded as a
  separate axis. It nets a positive +10.1% at 1 year but is stamped **Fragile**: it
  inherits the 1-year signal's non-robustness (dead by 2 years), it is tail-driven, and
  it would require **shorting exactly the least-borrowable names** (Caesars, crypto-
  adjacent, high-short-interest sponsors) in exactly the crises that produce its payoff.
- Costs: 2 legs × one-way cost × NAV on entry+exit (4 × `cost_bps`), plus borrow on the
  short leg pro-rated to the window (default 100 bps/yr — conservative for the easy
  names, wildly optimistic for the hard ones that matter most).

## Data sources

- **SPY** and each listed **sponsor ticker** — daily total-return closes
  (`auto_adjust=True`) via yfinance (no key), cached under `_cache/` as
  `snc_<ticker>.csv`, 1997 → 2026-06-30 (each from its own listing start). ADRs used
  where the parent is foreign (Toyota **TM**, Honda **HMC**, Barclays **BCS**, Bank of
  Nova Scotia **BNS**, Mercedes-Benz Group **MBGYY**). Fiserv is cached under its
  pre-2023 ticker **FISV** (renamed FI in 2023; Yahoo keeps the continuous history under
  FISV). Comerica (**CMA**) has no Yahoo tape and is a named no-coverage drop.
- **34 hardcoded naming-rights deals** in
  [`stadium_curse/data.py`](../stadium_curse/data.py). No free, machine-readable
  naming-rights index exists, so this is a hand-built table cross-referenced against each
  venue's public naming-rights record (each venue's naming history, contemporary deal
  announcements, and the sponsors' own filings). Announcement dates are the
  widely-reported deal-announcement dates (public record).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [160-skyscraper-curse](../../160-skyscraper-curse/) — the closest cousin: the "world's
  tallest **building** breaks ground at the top of the cycle" curse. Same
  hubris/peak-signaling family, but 160 tests a **macro/market** timing signal from a
  *building*; this study tests a **cross-sectional, firm-level** forward return from a
  *stadium naming deal*. Different unit (a country's index vs a single sponsor's stock),
  different trigger (a construction milestone vs a naming contract).
- [746-hq-relocation](../../746-hq-relocation/) — the "shiny new **headquarters** marks
  the top" corporate-vanity signal. Same trophy-real-estate intuition, but an HQ move is
  a *capex/operational* decision about where a firm works; a naming-rights deal is a
  *marketing/branding* spend on a building the firm does not own or occupy. Different
  corporate action, different sponsor set.
- [722-logo-rebrand](../../722-logo-rebrand/) — the "expensive **rebrand** signals a
  company papering over trouble" corporate-vanity signal. Same "conspicuous marketing
  spend as a top signal" family, but a rebrand changes the firm's *own identity*; a
  naming-rights deal buys someone else's stadium. Different action, non-overlapping event
  set.

None of the siblings test **what buying a stadium's or arena's name does to the
sponsor's stock** — that is this study's own axis.
