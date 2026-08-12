# References & literature map — Study 846 (Blockbuster Game-Launch Drift)

## The claim under test

- **The folklore.** Gaming/finance chatter treats a marquee AAA launch — *GTA V*, *Red Dead
  Redemption 2*, *Zelda: Tears of the Kingdom*, a *Call of Duty* or *Assassin's Creed* — as a
  tradable catalyst: **"buy the hype into the launch"** and either ride the momentum or
  **"sell the news."** We test the **finance transplant**: does the *publisher's* stock
  (TTWO / EA / NTDOY / UBSFY) earn an abnormal return around the ship date and over the ~20-
  session drift that follows?
- **Why the launch date is the honest anchor.** Reveal/announcement dates are fuzzy (teasers,
  leaks, delays), but the **launch date is an unambiguous public fact**, scheduled and
  marketed months in advance. We anchor on the standard-edition US street date, hardcoded
  from each title's Wikipedia release box and the publishers' press releases
  ([`data.py`](../game_launch/data.py)).
- **Publisher mapping.** Each launch is tied to the listed equity it most plausibly moves:
  **TTWO** (Rockstar/2K — GTA, Red Dead, Borderlands, Civilization), **EA** (Battlefield,
  Star Wars, Apex, Anthem, Mass Effect), **NTDOY** (Nintendo US ADR — Zelda, Mario, Pokémon,
  Smash), **UBSFY** (Ubisoft US ADR — Assassin's Creed, Far Cry, Watch Dogs). **ATVI**
  (Activision Blizzard — CoD, Diablo IV, Overwatch) is in the calendar for the record but has
  **no tape**: Microsoft closed its acquisition on **2023-10-13** and Yahoo serves no `ATVI`
  history, so those five launches are honestly excluded. *Cyberpunk 2077* (CD Projekt) and
  *Elden Ring* (Bandai Namco), named in the folklore, are out of scope — neither publisher is
  in this US-listed ticker set.
- **The efficient-markets prior.** A recurring, pre-announced, heavily-marketed product ship
  is exactly the kind of catalyst a semi-strong-efficient market should already price — see
  Fama (1970, *Efficient Capital Markets*, JF). And a single title is a small slice of a
  multi-franchise publisher's revenue, so the desk's prior is firmly **None/Weak**. With only
  ~33 resolvable events, the test is also **low-power**.

## What the literature actually says about event drift

- **Post-earnings-announcement drift (PEAD)** — Ball & Brown (1968, *JAR*); Bernard & Thomas
  (1989, 1990, *JAR / JAE*). The canonical "prices drift *after* a scheduled information
  event." A game launch is a product ship, not an earnings print, but the "ride the drift"
  reflex borrows PEAD's intuition; our test asks whether any drift is present around the
  *launch* specifically. We find none that clears significance.
- **"Buy the rumour, sell the news" / anticipation effects** — the idea that a known catalyst
  is bid up beforehand and sold once realised is old market lore with a thin formal record;
  the closest academic cousins are the pre-announcement drift and scheduled-announcement
  premium literatures (e.g. Savor & Wilson, 2016, *JFQA*, on scheduled macro announcements).
  None of it says a *product launch* pays.
- **Attention & investor-catalyst effects** — Barber & Odean (2008, *RFS*) on attention-driven
  buying; Da, Engelberg & Gao (2011, *JF*) on search-based attention. These motivate *why*
  gamers and retail might crowd a high-profile launch, and therefore why a post-ship move is
  plausible — but attention is not, by itself, a tradable edge, and our tape shows no
  reliable move in either direction.
- **Product-launch / new-release event studies** — Chaney, Devinney & Winer (1991, *Journal
  of Business*) find product-announcement stock reactions are small and highly conditional; a
  routine, expected AAA sequel is precisely the case where the reaction should be smallest.
  Consistent with our clean-zero result.

## Dedup — how this differs from its siblings

This is a distinct event/instrument from the desk's other pop-culture and launch studies:

- [844-madden-cover-curse](../../844-madden-cover-curse/) — the **Madden/NBA-2K "cover curse"**
  framing on **EA + TTWO only**, anchored on the annual *sports*-title ship; this study is the
  broader **blockbuster** launch cut across five publishers (TTWO/EA/NTDOY/UBSFY/ATVI) and
  marquee single-title releases, with a headline **20-day** drift window (vs 844's 2-week).
- [774-nintendo-direct](../../774-nintendo-direct/) — **NTDOY** around a *Nintendo Direct
  broadcast* (an owned-media showcase, not a launch); here it is the publishers around an
  actual *game launch date*.
- [771-box-office-bomb](../../771-box-office-bomb/) — a *film flop* event on the studio's
  stock, a demand-miss shock, not a scheduled game ship.
- [550-box-office-momentum](../../550-box-office-momentum/) — momentum in *film box-office*
  receipts, a demand series, not a publisher event study around a fixed date.

## Data & method

- **Real tape:** `TTWO`, `EA`, `NTDOY` (Nintendo ADR) and `UBSFY` (Ubisoft ADR) daily
  adjusted (total-return) closes vs `SPY` via [yfinance](https://github.com/ranaroussi/yfinance),
  one combined panel. Each launch is anchored to *its own* publisher; we measure the *abnormal*
  return `publisher − SPY`, not the raw move. `ATVI` is delisted (no post-2023-10-13 history).
- **Statistics:** one-sample *t* of the abnormal return across independent, non-overlapping
  launches (the correct unit — not a daily panel); a Newey-West (HAC) *t* on the date-ordered
  series; Wilson hit-rate interval; a 20-seed × 200-draw random-window placebo per cut;
  per-publisher and sub-era splits; a leave-one-out jackknife; a costed net leg.
- **Synthetic positive control:** a seeded paired (publisher, benchmark) world with a *planted*
  launch drift — the detector must recover a planted bump with unit slope and stay quiet on the
  null. See [`strategy.py`](../game_launch/strategy.py).

*Fama, E. (1970). Efficient Capital Markets. **Journal of Finance**. · Ball, R. & Brown, P.
(1968). **JAR**. · Bernard, V. & Thomas, J. (1989, 1990). **JAR / JAE**. · Barber, B. &
Odean, T. (2008). **RFS**. · Da, Z., Engelberg, J. & Gao, P. (2011). **JF**. · Savor, P. &
Wilson, M. (2016). **JFQA**. · Chaney, P., Devinney, T. & Winer, R. (1991). **Journal of
Business**. · Launch dates: Wikipedia per-title release boxes; TTWO / EA / Nintendo / Ubisoft
press releases. · ATVI delisting: Microsoft–Activision Blizzard deal close, 2023-10-13.*
