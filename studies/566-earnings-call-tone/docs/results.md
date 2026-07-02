# Results — Study 566 (Earnings-Call-Tone): a linguistic cousin of PEAD, on synthetic data

*Generated from [`earnings_call_tone/`](../earnings_call_tone/). This study is **synthetic-only**:
scored earnings-call transcripts joined to event-time abnormal returns are a paid-vendor /
hand-scored-academic product with **no free, no-key feed**, so the reproducible core is a
deterministic, seeded event panel (seed 566) and the real-tape fetch returns an **empty frame** by
design. The headline world plants a genuine linguistic drift (`tone_beta = 0.020`) so the machinery
can be shown faithful. Panel: 40 firms × 12 quarters = **480 firm-quarter calls**, fingerprint
`3ec051a6458a`; world params fingerprint `1c74d461b0e5`. As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `FRAGILE` · "Confounded by the number?" `NAMED`

The claim (Loughran & McDonald 2011; Price et al. 2012; Mayew & Venkatachalam 2012): the **net tone**
of an earnings-call transcript — how upbeat vs guarded management sounds — forecasts the stock's
**post-call drift** (the CAR after the call). It is the linguistic cousin of post-earnings-announcement
drift (PEAD, [Study 363](../../363-pead-drift/)): the *number* drifts, and the claim is that the
*words* drift on top of it.

Two facts decide the stamps. **First: there is no free real tape**, so a `REAL` Signal — which this
desk grants only to a robust *t* ≥ 2 on **real** data — is out of reach; the literature supports the
effect, so the honest stamp is **`WEAK`** ("the literature says real; this desk's tape can't certify
it"). **Second: the entire trap is confounding by the number.** Transcripts sound upbeat *because*
the quarter beat, so the naive tone→CAR slope double-counts PEAD. In our **null** world (`tone_beta =
0`, tone truly tells you nothing beyond the number) the *naive* tone slope-*t* still averages **+6.50**
across 25 seeds — a clean **false positive** — while the *surprise-controlled* slope-*t* averages
**−0.19**, correctly flat. Only the surprise-controlled slope isolates the words. That confound is
`NAMED` on the front card, and it is why an uncontrolled real-world replication of this claim is an
upper bound.

With a genuine linguistic edge planted, the engine banks it: the controlled tone slope-*t* is
**+7.95** on the headline panel and rises monotonically with the planted `tone_beta` (below). The
machinery is faithful — but faithfulness on synthetic data is a *machinery proof, never market
evidence* (METHODOLOGY): it caps the Signal at `WEAK`, it does not lift it.

## Data stamp

- **Panel**: 40 firms × 12 quarters = 480 firm-quarter earnings calls, columns `ticker`, `quarter`,
  `net_tone`, `surprise`, `car`; fingerprint `3ec051a6458a`
- **World**: deterministic generator, seed 566, planted `tone_beta = 0.020`, `surprise_beta = 0.030`;
  params fingerprint `1c74d461b0e5`
- **Real tape**: none available (paid-vendor / hand-scored); `fetch_panel` returns an empty frame

## The headline regression — naive vs surprise-controlled (the whole study)

| Regression of post-call CAR on… | tone slope | tone *t* | reads as |
|---|--:|--:|---|
| **net_tone alone** (naive) | **+0.0251** | **+12.05** | inflated by PEAD |
| **[net_tone, surprise]** (controlled) | **+0.0176** | **+7.95** | the isolated linguistic edge |

The naive slope is ~40% larger than the controlled one: a big chunk of the apparent tone edge is just
the numeric surprise wearing a linguistic costume. The controlled slope (*t* +7.95, placebo *p*
**0.0005**) is the honest headline — and on *this planted world* it is real by construction.

## The confounding proof — a false positive at the true null

| World | naive slope-*t* (25 seeds) | controlled slope-*t* (25 seeds) |
|---|--:|--:|
| **Null** (`tone_beta = 0`, tone adds nothing) | **+6.50** | **−0.19** |

At the null the naive read manufactures a *t* of +6.5 out of thin air — because tone is entangled with
the surprise that genuinely drives CAR. The controlled read sits at ≈ 0, as it must. **This is the
central result**: any tone→drift study that does not control for the numeric surprise is measuring
PEAD in disguise.

## The tradable long-short (naive, tone tails)

| | value |
|---|--:|
| Upbeat-tail mean CAR | **+2.65%** |
| Guarded-tail mean CAR | **−3.11%** |
| Spread (upbeat − guarded) | **+5.76%** per event (two-sample *t* +9.48) |
| Gross annualised (~4 calls/yr) | **+23.0%** |
| Net (4 turns/yr × 5 bps/leg + 50 bps borrow) | **+21.7%** |

The long-short *looks* spectacular — but it is the **naive** read: the tone tails are also the
surprise tails, so this spread is PEAD + tone, not tone alone. Costs are a footnote here; the *confound*
is the story, and a real short of the guarded tail pays borrow on exactly the names the market is
already selling.

## Robustness — the controlled sign holds across sub-panels (planted world)

| Sub-panel (quarters) | controlled tone slope | controlled tone *t* |
|---|--:|--:|
| Q01–Q03 | +0.0208 | **+4.44** |
| Q04–Q06 | +0.0129 | **+3.28** |
| Q07–Q09 | +0.0210 | **+5.01** |
| Q10–Q12 | +0.0185 | **+3.61** |

With a genuine edge planted, the controlled slope is positive and *t* > 3 in all four sub-panels — the
sign is stable *when the effect is truly there*. (On a real tape the sign-stability of a *controlled*
tone slope is exactly the unknown this desk cannot resolve without the data.)

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `tone_beta` | controlled mean slope-*t* (25 seeds) | |
|---|--:|---|
| 0.00 (null) | **−0.19** | flat — no false signal |
| 0.01 | +4.63 | edge emerging |
| 0.02 (headline) | +9.45 | clears the bar |
| 0.04 | +19.10 | strong |
| 0.06 | +28.74 | very strong |

At the null the controlled slope-*t* ≈ 0; planting a real linguistic drift drives it monotonically
past +2. The detector works — so the honest limitation is *data availability*, not a broken engine.
(Control only; never cited for a Signal stamp — a synthetic Sharpe in support of Signal is circular.)

## Why this can't be `REAL`

1. **No free real tape.** Scored transcripts × event-time CAR is a paid-vendor / hand-scored product.
   `REAL` needs a robust *t* ≥ 2 on *real* data; we have none. Capped at `WEAK`, stated on the Signal axis.
2. **Confounding by the number.** The naive tone slope is PEAD in disguise (false-positive *t* +6.5 at
   the null); only a surprise-controlled slope isolates the words, and even a real study must run that
   control before claiming a linguistic edge.
3. **Lexicon fragility (in the wild).** Real net-tone depends on the sentiment lexicon
   (Loughran-McDonald vs generic), on scripted-vs-Q&A weighting, and on management's strategic
   optimism — none of which the synthetic panel stress-tests. A real replication inherits all of it.

## The honest takeaway

Earnings-call tone as a linguistic cousin of PEAD is a *credible* effect — the literature finds it, and
our engine proves the machinery would bank it if it were there. But this desk cannot certify it: there
is no free real tape (`WEAK`, capped), and the naive version is a confound (a fake *t* +6.5 at the null),
so the tradable expression is `FRAGILE` — it rests entirely on cleanly separating the words from the
number, a separation a paid vendor and careful event study can do and a retail stack cannot. `WEAK` ×
`FRAGILE`, confound `NAMED`.
