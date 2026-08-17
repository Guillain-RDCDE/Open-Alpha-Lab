# Lot 913-962 — "the plumbing lot" (50 teardowns)

Fifty ideas the desk has **never** tested, deliberately tilted away from the folklore/TA/factor
space (which studies 1-912 have mined close to exhaustion) and toward the part of the market a
price-taker can actually *act on*: **wrapper mechanics, front-end cash management, corporate-action
plumbing, execution choices, leverage arithmetic, fixed-income structure, and the global/crypto
wrappers**. Roughly 60% of the lot is a structural race with a real chance of an `Investable`
stamp; the other 40% are implementation traps and yield illusions where the honest answer is
expected to be red.

House pattern for every study (reference build: `studies/912-gold-trend-managed/`):

* a **real tradable tape** (yfinance, `auto_adjust=True` total-return closes) or a
  **hardcoded, publicly-verifiable event calendar** clearly labelled as such;
* **one execution lag**, documented; costs one-way × NAV; shorts pay borrow;
* **excess-of-cash vs excess-of-cash** Sharpe races (BIL/SHY proxy), gross **and** net;
* HAC (Newey-West) *t* on the return/Sharpe difference + a block-bootstrap CI;
* an **era cut** and a **cost sweep**;
* a **seeded synthetic control** the detector must recover from a planted effect and stay quiet
  on the null;
* `docs/results.md` fingerprinted + as-of `2026-06-30`, `docs/references.md` literature map,
  two executed notebooks, offline tests wired into CI.

Shapes: **A** = hardcoded event-study; **B** = cross-sectional sort; **C** = labelled-proxy signal
→ forward-return timing; **D** = structural race (wrapper/rule vs its honest replication).

| # | slug | name | tape(s) | shape | claim (one line) | exp. |
|--|--|--|--|--|--|--|
| 913 | tracking-difference-persistence | Tracking-Difference Persistence | SPY, VOO, IVV, SPLG, QQQ, QQQM | D | Does last year's best S&P tracker stay the best one next year? | Weak/Investable |
| 914 | sec-lending-offset | Securities-Lending Offset | IWM, IJR, EEM, IEMG + indices | D | Do lending-heavy funds hand back the borrow revenue, or keep it? | Weak/Fragile |
| 915 | k1-vs-1099-structure | K-1 vs 1099 | DBC/PDBC, USO/BNO/USCI | D | Does the tax-friendly commodity clone track its K-1 twin, or pay for the wrapper? | Weak/Fragile |
| 916 | withholding-drag-international | Withholding Drag | VEA, IEFA, EFA, VXUS | D | How much of an international fund's dividend is eaten before it reaches you? | Real/Fragile |
| 917 | nav-staleness-timezone | Stale NAV | EWJ, EWG, FXI, EWA vs SPY | C | US closes strong — does tomorrow's Tokyo ETF already owe you the move? | Weak/Mirage |
| 918 | etn-creation-halt | Creation Halt | USO, VXX, GBTC-era ETNs | A | When a fund stops creating shares the premium explodes — is the fade tradable? | Weak/Fragile |
| 919 | index-methodology-change | Methodology Shock | QQQ 2023 special rebalance, Russell banding | A | Do index rule changes move the affected basket in a way you can front-run? | None/Mirage |
| 920 | total-cost-of-ownership | Total Cost of Ownership | SPY vs SPLG, QQQ vs QQQM, IVV | D | Cheapest fee or tightest spread — at what holding period does each win? | Real/Investable |
| 921 | bill-ladder-vs-etf | Bill Ladder vs ETF | BIL, SGOV, ^IRX | D | Does a home-made 3-month T-bill ladder beat the cash ETF that charges for it? | Real/Investable |
| 922 | frn-vs-fixed-front-end | Floating-Rate Front End | USFR, TFLO vs BIL, SHY | D | Through hikes and cuts, which end of the cash curve actually pays more? | Real/Investable |
| 923 | mmf-yield-lag | The Cash Lag | BIL, SGOV, USFR, ^IRX | C | Cash vehicles reprice late — can you switch ahead of the lag? | Weak/Fragile |
| 924 | cut-cycle-duration-extension | First Cut | IEF, TLT, BIL + hardcoded cut dates | A | Is the first Fed cut the moment to extend duration? | Weak/Fragile |
| 925 | short-rate-momentum-switch | Front-End Trend | SHY, IEF, TLT, BIL | C | Trend-follow the short rate to pick your duration — signal or noise? | Weak/Fragile |
| 926 | t-plus-one-settlement | T+1 | SPY, IWM, EEM | A | Did the May-2024 move to T+1 settlement change the overnight tape? | None/Mirage |
| 927 | dutch-auction-buyback | Dutch Auction | hardcoded tender list, SPY | A | Does a self-tender buyback mark the bottom for the issuer? | Weak/Fragile |
| 928 | odd-lot-tender | Odd-Lot Priority | hardcoded tender list, SPY | A | The odd-lot holder gets filled first — is that a real retail-only edge? | Weak/Fragile |
| 929 | rights-offering-discount | Rights Offering | hardcoded rights issues, SPY | A | Is the deep discount on a rights issue a gift or a warning? | None/Mirage |
| 930 | when-issued-spinoff | When-Issued Window | hardcoded spin-offs, parent + child | A | Between when-issued and regular-way, who is mispricing the child? | Weak/Fragile |
| 931 | cef-ipo-decay | The CEF IPO Hole | hardcoded CEF IPOs | A | You pay the underwriter on day one — how long until the discount catches you? | Real/Mirage |
| 932 | spac-trust-yield | Trust Yield | 2022 SPAC vintage, BIL | D | A pre-deal SPAC below trust: T-bills plus a free option, or a trap? | Real/Fragile |
| 933 | preferred-vs-baby-bond | Same Issuer, Two Ladders | issuer preferred vs baby bonds | D | For the same balance sheet, which rung of the capital structure pays best? | Weak/Fragile |
| 934 | lump-sum-vs-dca | Lump Sum vs DCA | SPY, IEF, BIL | D | Drip it in or send it all — which one actually wins, and how often? | Real/Investable |
| 935 | value-averaging | Value Averaging | SPY, BIL | D | Edleson's rule beats DCA on paper — does it on the tape, with the cash it needs? | Weak/Fragile |
| 936 | rebalance-bands | Tolerance Bands | SPY, IEF, GLD | D | Rebalance on a 5/25 band or on the calendar — is either worth the turnover? | Weak/Investable |
| 937 | tranched-rebalancing | Tranches | SPY/IEF rule, 21 start dates | D | Split the rule into overlapping tranches — does it really kill timing luck? | Real/Investable |
| 938 | open-vs-close-execution | Open or Close | SPY, IWM, EEM, EFA | D | The same monthly rule, executed at the open vs the close — which leaks less? | Weak/Fragile |
| 939 | drip-vs-sweep | DRIP or Sweep | SPY, VYM, SCHD | D | Reinvest the dividend the day it lands, or sweep it quarterly? | None/Mirage |
| 940 | turnover-budget | The Turnover Budget | momentum sleeve on sector ETFs | D | How often can a momentum sleeve trade before the costs eat the whole edge? | Real/Fragile |
| 941 | double-short-leveraged-pair | Short Both Legs | TQQQ + SQQQ, QQQ | D | Short the 3x long *and* the 3x short — is the decay harvest free money? | Weak/Mirage |
| 942 | inverse-etf-structural-loss | The Inverse Tax | SH, PSQ, SDS vs shorting SPY | D | Is an inverse ETF a worse short than simply being short? | Real/Investable |
| 943 | leverage-reset-frequency | Reset Frequency | SSO, UPRO vs monthly-reset replication | D | Daily reset is the villain — would a monthly reset really be better? | Weak/Fragile |
| 944 | optimal-leverage-realized | How Much Leverage | SPY at 1x-3x, BIL | D | The maths says ~2x; what did the real tape actually reward? | Real/Fragile |
| 945 | leverage-financing-cost | The Hidden Financing | SSO, UPRO vs 2x/3x replication + BIL | D | What rate are you really paying inside a leveraged fund? | Real/Investable |
| 946 | distribution-rate-illusion | Distribution ≠ Return | ~15 high-payout ETFs | B | Does a fatter advertised distribution predict a fatter total return? | Real/Mirage |
| 947 | buffer-ladder-vs-single | The Buffer Ladder | BUFR vs single-vintage buffers | D | Does laddering defined-outcome funds beat picking one vintage? | Weak/Fragile |
| 948 | income-fund-capture-ratio | Capture Ratio | income ETFs vs SPY/QQQ | B | Up-capture minus down-capture: the only honest scorecard for income funds? | Real/Fragile |
| 949 | tips-roll-carry | Riding the TIPS Curve | VTIP, TIP, LTPZ, BIL | D | Is there a roll-down carry in real yields, or only duration risk? | Weak/Fragile |
| 950 | strips-convexity-pickup | Zero-Coupon Convexity | EDV, ZROZ vs TLT + BIL | D | Does a zero-coupon barbell pay you for convexity, or just for duration? | Weak/Fragile |
| 951 | crossover-credit-bbb-bb | The Crossover Rung | LQD, ANGL, HYG, USHY | D | Is the BBB/BB boundary the best-paid rung on the credit ladder? | Real/Fragile |
| 952 | muni-after-tax-equivalent | After-Tax Equivalent | MUB, VTEB vs LQD, AGG | D | At which tax bracket does the muni actually win, on the real tape? | Real/Investable |
| 953 | convertible-replication | Replicating the Convert | CWB vs SPY + LQD mix | D | Is a convertible fund anything more than equity plus credit in a costume? | Weak/Mirage |
| 954 | hy-as-levered-equity | High Yield in Disguise | HYG, JNK vs SPY + IEF | D | Is high-yield credit just levered equity you pay a bond fee for? | Real/Fragile |
| 955 | adr-overnight-catchup | ADR Catch-Up | ADRs vs home closes + FX | C | The home market already moved — does the ADR still owe you the gap? | Weak/Mirage |
| 956 | adr-custody-fee-drag | The Custody Fee | ADR vs local line total return | D | The invisible ADR pass-through fee: how much does it cost per year? | Real/Fragile |
| 957 | holdco-nav-discount | Holdco Discount | listed holdcos vs their stakes | D | Buying a conglomerate below the sum of its parts — does the gap ever close? | Weak/Fragile |
| 958 | spot-btc-etf-basis | Spot ETF Basis | IBIT, FBTC vs BITO, BTC-USD | D | After the spot ETFs landed, is the cash-and-carry basis still there? | Weak/Fragile |
| 959 | crypto-etf-fee-war | Crypto Fee War | IBIT, FBTC, ARKB, BITB, GBTC | D | In the newest wrapper race, does the cheapest fund actually track best? | Real/Investable |
| 960 | eth-staking-yield-forgone | The Unstaked ETF | ETHE/ETHA vs ETH-USD + staking | D | US spot ETH funds do not stake — what does that cost the holder? | Real/Fragile |
| 961 | gold-wrapper-cheapest | Which Gold | GLD, IAU, GLDM, SGOL, BAR | D | Five wrappers, one metal — does the cheapest one really win? | Real/Investable |
| 962 | sector-etf-vs-holdings | Do It Yourself | XLK/XLE vs their top holdings | D | Can you replicate a sector ETF with a handful of names and skip the fee? | Weak/Mirage |

**Dedup notes** (checked against all of 1-912): 913/920/959/961 are *wrapper-selection* races — the
desk has never tested tracking difference; 378 is an intraday premium/discount **proxy** and 379 is
a lead-lag, neither is this. 934-940 are *implementation* choices (97 is fixed 60/40, 102 is the
rebalancing bonus, 836 documents timing luck without testing the tranche fix). 941-945 are the
*mechanics* of leverage (593/594 are leveraged **allocations**). 946-948 sit next to 337 (per-fund
income race) and 910 (CEF managed distributions) but ask the cross-sectional question neither did.
949-954 are fixed-income **structure** races, distinct from 380 (generic roll-down), 381
(breakevens), 884 (barbell), 885/887/907 (sleeve races) and 339 (long-only converts). 955-957 are
listing-structure gaps, distinct from 620 (A-H) and 621 (share classes). 958-960 are post-2024 spot
ETF plumbing, distinct from 618 (GBTC premium) and 619 (BITO roll).
