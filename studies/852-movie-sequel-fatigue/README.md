# Study 852 — Movie-Sequel Fatigue 🎬

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is franchise fatigue real in studio reactions? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The fatigue slope is negative (the folklore's sign) but the *natural* raw specification is insignificant (*t* = **−1.20**, permutation p = **0.24**); it clears significance only with a franchise fixed effect (*t* = **−3.05**) and even then lives **entirely post-2018** (early *t* = −0.95 vs late *t* = −8.85), failing cross-era robustness; the mean reaction (**−0.73%**, NW *t* = −2.06) sits inside the random-date luck cloud (p = **0.11**); H2 persistence points the right way but is insignificant (AR(1) *t* = 0.81, Welch *t* = −1.36). Directionally consistent, not robust. *Survivorship: the famous franchises, chosen with hindsight — magnitudes are an upper bound.* |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The short-the-fatigued-sequel leg nominally clears costs (14 fires, +1.82% net @ 5 bps, *t* = 2.79) but rests on in-sample conditional selection of an H2 persistence effect that is itself insignificant, fires only 14 times in 20 years, and is concentrated in the same post-2018 window — a number that won't survive out of sample. |
| **Does the market flinch when the franchise gets tired?** | ![Overstated](https://img.shields.io/badge/Overstated-8b949e?style=flat-square) | 43 famous franchise entries, a proper event study, two placebos and a two-era cut agree: a whiff of "later sequels react worse," but only in a fixed-effect specification that is entirely a recent-era artifact — not the robust, tradable fatigue the folklore imagines. |

> **In one sentence:** across 43 famous franchise entries (Marvel, Star Wars, Fast & Furious,
> Jurassic World, Pirates, Transformers), "sequel fatigue" points the right way — later
> sequels *do* draw a slightly worse studio reaction, ≈−0.9%/sequel *within* a franchise — but
> the effect is insignificant in its natural form (*t* = −1.20), reaches significance only with
> a fixed-effect specification that lives entirely post-2018, the average reaction is inside the
> random-date noise (p = 0.11), and the "short the tired sequel" trade rests on 14 in-sample
> fires of an insignificant persistence signal — **Weak signal, Fragile tradability.**

## What we tested

The folklore: as a franchise ages, each sequel opens weaker, so the *studio* should react
worse to sequel N than to sequel N-1, and a down sequence should overhang the next entry. We
steelman it on a hand-curated table of **46 franchise entries across 14 sub-franchise lines**
— Marvel (Avengers, Guardians, Thor, Captain America, Ant-Man, Doctor Strange, Black Panther),
the Star Wars sequel trilogy, Pirates of the Caribbean and Frozen (Disney / `DIS`); Fast &
Furious 5-10, Jurassic World, Despicable Me (Comcast-NBCUniversal / `CMCSA`); Transformers
(Paramount / `PARA`) — each with its real wide-release opening date and true sequel number.
We anchor a studio-reaction window on the first session on/after the Monday after the opening
weekend (the box-office is public by then; base = the opening-Friday close, zero look-ahead),
measure the studio's abnormal return `studio − SPY`, and ask two things: **(H1)** does that
reaction CAR *decline with sequel number* (raw and franchise-fixed-effect OLS slope, a two-era
robustness cut, a 5,000-draw label-permutation placebo)? and **(H2)** does a *down sequence*
predict a worse next entry (a within-franchise AR(1) + a down/up Welch split)? A random-date
placebo checks whether the per-event CARs are ordinary tracking noise, a costed short-the-tired-
sequel timer prices the trade, and a 20-seed synthetic control with a *planted* fatigue slope
proves the machinery. The 3 pre-2021 Transformers fall outside `PARA`'s continuous history and
are dropped, not back-filled. **Survivorship** (famous franchises chosen with hindsight) is
named on the Signal axis. **Dedup:** [771-box-office-bomb](../771-box-office-bomb/) tests
selling a studio after a *single* notorious flop (not the sequel-number trend);
[847-rotten-tomatoes-studio](../847-rotten-tomatoes-studio/) tests the *critic-score* reaction,
not the sequel ordinal; [550-box-office-momentum](../550-box-office-momentum/) tests *momentum*
in studio returns, not the within-franchise fatigue tilt. **As-of 2026-06-30.**

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "franchise fatigue" *should* show up in the studio's stock if the story is right, what the tape actually shows across 43 sequels, and why the one significant number is a recent-era artifact |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the raw vs franchise-FE fatigue slope, the two-era cut, the AR(1) persistence, both placebos (label-permutation + random-date), the costed short leg, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`sequel_fatigue/`](sequel_fatigue/). The franchise calendar is hand-curated from
Box Office Mojo / studio releases; DIS/CMCSA/PARA and SPY are total-return closes via yfinance,
`PARA` coverage named honestly (continuous only from 2021). **Not investment advice** —
research & education. See [LICENSE](../../LICENSE).*
