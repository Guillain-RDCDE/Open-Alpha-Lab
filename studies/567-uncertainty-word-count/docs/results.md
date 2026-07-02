# Results — Study 567 (Uncertainty-Word-Count): the LM uncertainty dictionary as a vol/return signal

*Generated from [`uncertainty_word_count/`](../uncertainty_word_count/). **This study is
synthetic-only by design** — there is no free, no-key retail feed of parsed EDGAR filings scored
against a licensed Loughran-McDonald **Uncertainty** lexicon and joined to *forward* realised
volatility (that lives in paid NLP vendors and hand-parsed academic samples). So the reproducible
headline runs on the deterministic, seeded firm-event panel `synthetic_panel(seed=567)` (planted
anomaly `uncert_beta = 0.020`, return drag `ret_beta = −0.020`): 40 firms × 12 periods = **480
firm-events**, panel fingerprint `fa07e8ea11c3`, knob fingerprint `dc53b20dee4a`. As-of
**2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `FRAGILE`

The claim (Loughran & McDonald 2011; Campbell et al. 2014; Kravet & Muslu 2013): the **density of
uncertainty / hedging words** in a firm's disclosure — its share of tokens in the LM Uncertainty
word list ("uncertain", "maybe", "could", "risk", "approximately", "depend") — predicts **higher
future realised volatility** and (the softer half) **lower future returns**.

**We cannot certify this on a real tape** — there is no free feed, so the Signal axis is **capped at
`WEAK`** (a `REAL` stamp needs a robust *t* ≥ 2 on a **real** tape; literature support alone reads
`WEAK`). What the synthetic engine *does* show, cleanly, is that the effect is **overwhelmingly a
confound** until you control for it: the naive uncertainty→forward-vol slope is *t* **13.4**, but the
firms that write hedgy text are exactly the firms that are *already jumpy*, and once you control for
**trailing realised vol** the slope collapses to *t* **3.3** — two-thirds of the raw signal was
vol-persistence, not text. The planted textual residual survives (placebo *p* = **0.0005** on the
seeded world), the return drag is *t* **−6.92**, and the long-plain/short-hedgy book earns **+1.79%**
per event (*t* 2.46) gross, **+5.85%/yr** net — but on a real tape none of this is verified, and the
sub-window slope-*t* wanders (−0.46 → +2.89 → +1.20 → +2.73). `WEAK × FRAGILE`.

## Data stamp

- **Synthetic firm-event panel** (`synthetic_panel(seed=567)`): 40 firms × 12 filing periods = 480
  events, columns `ticker, period, uncert, trail_vol, fwd_vol, fwd_ret`, fingerprint `fa07e8ea11c3`
- **Planted world**: `uncert_beta = 0.020` (forward-vol loading on *residual* uncertainty),
  `ret_beta = −0.020` (return drag), knob fingerprint `dc53b20dee4a`
- **Real tape**: none (documented stub — `fetch_panel()` returns an empty frame; no free feed exists)

## The headline — the confound is (almost) the whole naive signal

| Forward-vol regression | Uncertainty slope | *t* | Reads as |
|---|---|---|---|
| **Naive** (fwd_vol on uncert) | **13.00** | **+13.41** | huge — but contaminated |
| **Controlled** (fwd_vol on [uncert, trail_vol]) | **6.13** | **+3.30** | the isolated textual edge |

The naive uncertainty→forward-vol slope is enormous (*t* 13.4). But hedgy firms are jumpy firms, and
vol is autocorrelated, so most of that is **vol persistence, not text**. Adding trailing realised vol
as a control (trail-vol slope +0.254, the persistence channel) **halves the coefficient and cuts the
*t* to 3.3**. That surviving 3.3 is the planted textual residual — real *in this synthetic world*.

**Label-shuffle placebo** (2000 perms, shuffle uncertainty against [fwd_vol, trail_vol], refit the
controlled slope): *p* = **0.0005** — the residual sits deep in the tail, not manufactured by the
control.

## The secondary axis — the return drag

| | value |
|---|---|
| Forward-return slope on uncertainty (controlled) | **−9.58** per density unit |
| Slope *t* | **−6.92** (negative = the "lower returns" claim) |

Hedgier text goes with *lower* forward returns in the planted world — the softer half of the claim,
here by construction.

## The tradable long-short (long plain / short hedgy, return axis)

| | value |
|---|---|
| Plain (low-uncertainty) tercile mean forward return | **+2.92%** |
| Hedgy (high-uncertainty) tercile mean forward return | **+1.13%** |
| Spread (plain − hedgy), per event | **+1.79%** (two-sample *t* **2.46**) |
| Gross, annualised (~4 filings/yr) | **+7.15%/yr** |
| Net (5 bps/leg round-trip × 4 turns + 50 bps/yr borrow) | **+5.85%/yr** |

The tercile sort (k = 144 per tail, pooled across the 480 events) reproduces the drag as a tradable
spread. Net of costs and a hedgy-leg borrow it still clears — *in the synthetic world*. On a real
tape this is unverified.

## Robustness — the per-window slope-*t* wanders

| Sub-panel | Controlled uncert slope | Slope *t* | Reads as |
|---|---|---|---|
| P01–P03 | −1.72 | **−0.46** | absent / wrong sign |
| P04–P06 | 11.10 | **+2.89** | present |
| P07–P09 | 4.39 | **+1.20** | weak |
| P10–P12 | 9.95 | **+2.73** | present |

Even in a world where the effect is *planted*, a 120-event sub-panel is noisy: the slope-*t* ranges
from −0.46 to +2.89. The pooled 480-event *t* of 3.3 is solid, but any single quarter's worth of
filings is not — a caution that maps directly to how fragile a real, once-a-quarter textual signal
would be.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `uncert_beta` | Mean controlled slope-*t* (25 seeds) | |
|---|---|---|
| 0.000 (null) | **−0.26** | flat — no false signal |
| 0.010 | +1.94 | emerging |
| 0.020 (headline) | **+4.13** | clears the bar |
| 0.030 | +6.28 | strong |
| 0.045 | +9.41 | very strong |

At the null the controlled slope-*t* is ≈ 0 (−0.26); planting a genuine textual link (`uncert_beta >
0`) drives it monotonically past +2. The detector works — the null does not print a false positive,
and the effect scales with the plant. (Control only; never cited for a real-tape stamp — there is no
real tape.)

## Why this is `WEAK`, not `REAL`

1. **No free real feed.** Parsing EDGAR full-text against a licensed LM Uncertainty lexicon and
   joining to event-time forward vol is a paid-vendor / hand-parsed pipeline. The desk's stack cannot
   reach it, so `fetch_panel()` is a documented stub returning an empty frame — and a `REAL` Signal
   requires a robust *t* ≥ 2 on a **real** tape. Capped at `WEAK`, exactly like the desk's
   lego-returns / whisky-cask / earnings-call-tone studies.
2. **The confound is the danger.** The single most important lesson here is that the *naive* signal
   (*t* 13.4) is mostly vol persistence — a real study that skips the trailing-vol control would
   publish a spurious blockbuster. Only the controlled slope (*t* 3.3) is the text.
3. **Sub-window fragility.** Even with the effect planted, per-quarter slope-*t* wanders below the
   bar — a once-a-quarter textual signal has little data per period.

## The honest takeaway

The uncertainty-word-count claim is well documented in the literature (LM 2011 and its descendants),
and the engine here proves the machinery would catch it *if* it were real — the null reads flat and a
planted effect scales cleanly past *t* 2. But on this desk's own tape it **cannot be certified**:
there is no free feed of scored filings joined to forward vol, so the Signal is `WEAK` by
construction, and the tradability is `FRAGILE` (the effect is dominated by a vol-persistence confound
in the raw data and its per-window *t* wanders). The value of this study is the **confound lesson**:
uncertainty words predict future vol mostly because uncertain firms are *already* volatile — control
for trailing vol before you believe the text.
