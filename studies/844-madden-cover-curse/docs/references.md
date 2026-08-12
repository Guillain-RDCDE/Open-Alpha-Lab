# References & literature map — Study 844 (Madden-Cover-Curse)

## The claim under test

- **The folklore.** The "**Madden curse**" is the gaming superstition that the athlete
  chosen for the cover of that year's *Madden NFL* suffers an injury or a slump the
  following season — a run of coincidences (Garrison Hearst, Michael Vick, Marshall Faulk,
  Donovan McNabb, Shaun Alexander…) blown up into lore, with the same jinx folklore riding
  along on the *NBA 2K* cover. We are not a sports-medicine desk, so we test the **finance
  transplant**: the tradable reflex "**buy the hype into the new Madden / NBA 2K**" — does
  the *publisher* (EA for *Madden*, TTWO for *NBA 2K*) earn an abnormal return around the
  cover reveal and the annual launch?
- **Why the launch is the honest anchor.** A cover *reveal* date is fuzzy (teasers, leaks,
  staggered edition reveals), but the **launch date is an unambiguous public fact**,
  scheduled and marketed months in advance. We anchor on the launch and carry the cover
  athlete for colour. Dates are hardcoded from each title's Wikipedia release box and the
  publishers' press releases ([`data.py`](../madden_curse/data.py)).
- **The efficient-markets prior.** A recurring, pre-announced, once-a-year product ship is
  exactly the kind of catalyst a semi-strong-efficient market should already price — see
  Fama (1970, *Efficient Capital Markets*, JF). And a single annual sports title is a small
  slice of a multi-franchise publisher's revenue (EA also ships EA Sports FC and Apex
  Legends; Take-Two also ships GTA and the wider Rockstar/2K catalogue), so the desk's prior
  is firmly **None**.

## What the literature actually says about event drift

- **Post-earnings-announcement drift (PEAD)** — Ball & Brown (1968, *JAR*); Bernard & Thomas
  (1989, 1990, *JAR / JAE*). The canonical "prices drift *after* a scheduled information
  event." A game launch is a product ship, not an earnings print, but the folklore borrows
  PEAD's intuition; our test asks whether any drift is present around the *launch*
  specifically. We find none that clears significance.
- **"Buy the rumour, sell the news" / anticipation effects** — the idea that a known catalyst
  is bid up beforehand and sold once realised is old market lore with a thin formal record;
  the closest academic cousins are the pre-announcement drift and scheduled-announcement
  premium literatures (e.g. Savor & Wilson, 2016, *JFQA*, on scheduled macro announcements).
  None of it says a *product launch* pays.
- **Attention & investor-catalyst effects** — Barber & Odean (2008, *RFS*) on attention-driven
  buying; Da, Engelberg & Gao (2011, *JF*) on search-based attention. These motivate *why*
  gamers and retail might crowd a high-profile launch, and therefore why a post-ship
  reversal is plausible — but attention is not, by itself, a tradable edge, and our tape
  shows the crowd's move (if any) is a small, insignificant dip.
- **Product-launch / new-release event studies** — Chaney, Devinney & Winer (1991, *Journal
  of Business*) find product-announcement stock reactions are small and highly conditional;
  a routine, expected annual sequel is precisely the case where the reaction should be
  smallest. Consistent with our clean-zero result.

## Dedup — how this differs from its siblings

This is a distinct event/instrument from the desk's other pop-culture and launch studies:

- [720-super-bowl-advertiser](../../720-super-bowl-advertiser/) — the *advertisers'* stocks
  around the Super Bowl broadcast, a media-buy event; here it is the *game publishers'*
  stocks around a *product ship*.
- [774-nintendo-direct](../../774-nintendo-direct/) — **NTDOY** around a *Nintendo Direct
  broadcast* (an owned-media showcase, not a launch); here it is **EA/TTWO** around an
  actual *game launch date*.
- [550-box-office-momentum](../../550-box-office-momentum/) — momentum in *film box-office*
  receipts, a demand series, not a publisher event study around a fixed date.
- [846-game-launch-drift](../../846-game-launch-drift/) — a broader/adjacent game-launch drift
  cut; this study is the **Madden/NBA-2K-specific "cover curse"** framing on EA + TTWO only.

## Data & method

- **Real tape:** `EA` (Electronic Arts) and `TTWO` (Take-Two Interactive) daily adjusted
  (total-return) closes vs `SPY` via [yfinance](https://github.com/ranaroussi/yfinance), one
  combined panel. Each launch is anchored to *its own* publisher; we measure the *abnormal*
  return `publisher − SPY`, not the raw move.
- **Statistics:** one-sample *t* of the abnormal return across independent, non-overlapping
  launches (the correct unit — not a daily panel); a Newey-West (HAC) *t* on the date-ordered
  series; Wilson hit-rate interval; a 20-seed × 200-draw random-window placebo per cut;
  per-publisher and sub-era splits; a leave-one-out jackknife; a costed net leg.
- **Synthetic positive control:** a seeded paired (publisher, benchmark) world with a
  *planted* launch-week drift — the detector must recover a planted bump with unit slope and
  stay quiet on the null. See [`strategy.py`](../madden_curse/strategy.py).

*Fama, E. (1970). Efficient Capital Markets. **Journal of Finance**. · Ball, R. & Brown, P.
(1968). **JAR**. · Bernard, V. & Thomas, J. (1989, 1990). **JAR / JAE**. · Barber, B. &
Odean, T. (2008). **RFS**. · Da, Z., Engelberg, J. & Gao, P. (2011). **JF**. · Savor, P. &
Wilson, M. (2016). **JFQA**. · Chaney, P., Devinney, T. & Winer, R. (1991). **Journal of
Business**. · Launch dates & cover athletes: Wikipedia per-title release boxes; EA and
Take-Two / 2K Sports press releases.*
