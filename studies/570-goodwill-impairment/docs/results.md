# Results — Study 570 (Goodwill-Impairment): does bloated goodwill predict the write-down and the drop?

*Generated from [`goodwill_impairment/`](../goodwill_impairment/) on this study's **synthetic-only**
deterministic panel (no real tape exists for a no-key retail stack — see the data caveat below).
Headline world: `synthetic_panel(n_firms=400, ret_alpha=-0.14, imp_beta=2.2, seed=570)`, panel
fingerprint `6bd491ebb228`, world-truth fingerprint `8710cfb12768`. As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE`

The **overpaid-acquisition** claim (accounting): a firm carrying a *bloated goodwill balance* —
goodwill large relative to total assets — overpaid for its acquisitions, and that overpayment
eventually surfaces as a **goodwill impairment** (write-down). Because the market is slow to price
the coming write-down, high-goodwill firms should (a) impair at higher rates and (b) earn **lower**
forward returns — a long-low-goodwill / short-high-goodwill spread with the wrong-way sign a
risk-premium story would predict.

**This study is synthetic-only.** The point-in-time panel the real test needs — goodwill / total
assets at year *t*, tagged impairment events, and forward returns — is **not reachable** from a
no-key retail stack (yfinance exposes neither a clean point-in-time goodwill history nor tagged
write-down events, and there is no free survivorship-free source for it). So we cannot put a *t*
on a real tape, and per the desk's inference bar a synthetic-only study **can never earn `REAL`**
(that requires a robust *t* ≥ 2 on a REAL tape). It is capped at `WEAK`. `Tradability` is `MIRAGE`:
there is no real result to trade, and the real-world frictions (illiquid, corporate-action-prone
high-goodwill names on the short leg; a hard borrow; thin capacity) sit on top of a signal that has
never touched a real tape.

What the code **does** show, on the synthetic world, is that the **engine is a faithful detector**:
with the effect planted (`ret_alpha = -0.14`) it recovers a clean low-minus-high spread of
**+5.78%** (Welch *t* **+3.86**, placebo *p* **0.0005**), a firm-level slope of **−17.3%** per unit
goodwill/assets (slope-*t* **−4.22**, corr **−0.21**), and it stays **flat at the null** (mean
slope-*t* **−0.05** over 25 seeds). The impairment-event link is real in the panel too — the
high-goodwill quintile impairs at **12.5%** vs the low quintile's **6.2%** (a **6.2pp** gap) — and
that link *persists at the return-null*, exactly as the accounting story says (bloated goodwill
impairs more whether or not the market mis-prices it). So the machinery would catch the puzzle on a
real tape; we just don't have the tape.

## Data stamp

- **Panel**: 400 synthetic firms, one cross-section, columns `goodwill_ta` (goodwill / total
  assets, year *t*), `impaired` (forward 0/1 write-down event), `forward_ret` (forward return);
  `ret_alpha = -0.14`, `imp_beta = 2.2`, `seed = 570`; fingerprint `6bd491ebb228`
- **World truth**: `ret_alpha = -0.14`, `imp_beta = 2.2`; fingerprint `8710cfb12768`
- **Real tape**: **none** — `fetch_panel(fetch=True)` raises by design (data unavailable)

## The goodwill sort (quintiles, 80 firms/bucket) — the planted puzzle, recovered

| Quintile | Forward return | Impairment rate |
|---|---|---|
| **Low goodwill/assets** (least overpaid) | **+9.42%** | **6.2%** |
| **High goodwill/assets** (most overpaid) | **+3.64%** | **12.5%** |
| **Spread (low − high)** | **+5.78%** (Welch *t* **+3.86**) | gap **+6.2pp** (two-prop *z* **+1.36**) |

The claim predicts low-goodwill > high-goodwill (a *positive* spread) and a higher impairment rate
in the high bucket. The engine recovers both on the planted world. The label-shuffle placebo
*p* = **0.0005** says the spread is not noise on this panel.

## The firm-level relation (the sign IS the puzzle)

| | value |
|---|---|
| Slope (forward_ret on goodwill/assets) | **−17.3%** per unit |
| Slope *t* | **−4.22** (a *negative* slope is the puzzle) |
| corr(goodwill/assets, forward return) | **−0.21** |

A negative slope is the overpaid-acquisition drag: more goodwill, less return — the effect the
knob planted.

## Robustness — the sign holds across bucket definitions

| Tail fraction | Spread (low − high) | Welch *t* | Firm slope-*t* |
|---|---|---|---|
| 0.10 (deciles) | **+7.71%** | **+3.88** | −4.22 |
| 0.20 (quintiles, headline) | **+5.78%** | **+3.86** | −4.22 |
| 0.25 (quartiles) | **+5.30%** | **+3.94** | −4.22 |
| 0.30 | **+4.79%** | **+3.72** | −4.22 |

The spread is positive and *t* > 3.5 at every sensible bucket cut — the sign is stable *on the
synthetic world*. (Sign-stability on a real tape is exactly what we cannot check.)

## Costs

| | value |
|---|---|
| Gross spread (low − high, headline) | **+5.78%** |
| Net (5 bps/leg round-trip + 150 bps/yr borrow, 1y hold) | **+4.08%** |

Costs shave ~1.7pp off the synthetic gross. On a real tape the high-goodwill short leg — recent
over-payers, often smaller and corporate-action-prone — would carry a materially harder borrow than
the 150 bps charged here; the frictions are named, not solved, because there is no real tape to
apply them to.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `ret_alpha` | Mean slope-*t* (25 seeds) | Mean long-short *t* | |
|---|---|---|---|
| 0.00 (null) | **−0.05** | **+0.13** | flat — no false signal |
| −0.05 | −1.99 | +1.80 | effect emerging |
| −0.10 | −3.77 | +3.31 | effect visible |
| −0.14 (headline) | −5.02 | +4.35 | clears the bar |
| −0.20 | −6.57 | +5.61 | strong |

At the null both stats sit ≈ 0 — no false signal; planting a genuine overpaid-M&A effect
(`ret_alpha < 0`) drives the slope-*t* below −2 and the long-short *t* above +2 as it grows. The
detector works — which is precisely why the honest verdict is `WEAK` (engine validated, literature
support real) rather than `NONE`, and why it **cannot** be `REAL` (no real tape). (Control only;
never cited as a real-tape result.)

## Why this is `WEAK`, not `REAL` — the data caveat, stated plainly

1. **No point-in-time goodwill panel.** The real test needs goodwill / total assets *as reported at
   year t* for a survivorship-free universe, plus **tagged impairment events**. yfinance exposes a
   shallow, restated snapshot at best and no impairment tags; a real replication needs
   Compustat/point-in-time footnote data. `fetch_panel(fetch=True)` raises rather than fabricate a
   tape.
2. **Synthetic-only ⇒ capped at `WEAK`.** The desk's rule: `REAL` requires a robust *t* ≥ 2 on a
   *real* tape. A planted-and-recovered synthetic effect proves the *machinery*, not the *world*.
3. **The literature leans the right way** (Li–Shroff–Venkataraman–Zhang 2011; Hayn–Hughes 2006;
   Gu–Lev 2011) — high goodwill and overpaid acquisitions do predict impairments and negative
   drift — which is why the signal is `WEAK` (supported) rather than `NONE` (contradicted).

## The honest takeaway

The overpaid-acquisition story is coherent and has real literature behind it, and the engine here
would catch it (planted effect recovered at *t* ≈ +3.9 for the spread, −4.2 for the firm slope;
flat at the null over 25 seeds). But this study **cannot** certify the effect on the tape, because
the point-in-time goodwill / impairment panel it needs is out of reach for a no-key retail stack.
`WEAK` × `MIRAGE`: a real, literature-backed idea and a validated detector, with an honest
data-availability wall between it and a tradable claim.
