# Study 722 — Logo-Rebrand 🎨

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the rebrand a real signal? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The renewal/floundering claim is the **drift**, and the 6-month drift is **−4.6%** (Welch *t* = **−1.05**, placebo *p* = **0.33**) — a `NONE`. The only flicker is a **+2.4%** week-one pop (*t* = **1.96**, *p* = 0.022) that **fails t ≥ 2 and collapses to t = 1.21** once two news-confounded outliers (GM, BlackBerry) drop. **Survivorship** named here: the worst-outcome rebrands delisted/went private. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Buy-the-rebrand-and-hold is **−2.2% gross** / **−2.4% net** per event — the soft-negative drift eats the little pop, and the pop leg isn't investable (sub-*t*=2, outlier-built). |
| **Renewal or red-flag?** | ![Coin flip](https://img.shields.io/badge/Renewal_or_red--flag%3F-Coin_flip-8b949e?style=flat-square) | The drift can't separate them: **name-changes drift −16%, logo refreshes +8%** — the firm's *prior* health leaking through backwards, not a forward signal. |

> **In one sentence:** a corporate rebrand is neither a reliable renewal catalyst nor a reliable floundering-firm tell — across a transparent table of ~26 real rebrands (2010–2025), the 6-month drift is a statistical zero (**−4.6%**, *t* = −1.05) and the only positive, a week-one **+2.4%** pop, is two confounded headlines deep and never clears *t* = 2, so the buy-the-rebrand trade loses **−2.2%** before costs and the whole thing is a coin flip wearing a new logo.

## What we tested

Two camps read the same event oppositely: **renewal** ("a fresh name/logo marks a turnaround — buy it") and **floundering** ("a rebrand is a distress tell — fade it"). Both are directional bets on the abnormal drift after a rebrand, so one event study referees. We hardcode a **transparent table of ~26 real rebrands 2010–2025** — name changes (Facebook→Meta, Coach→Tapestry), identity refreshes (ConAgra→Conagra Brands), and pure logo redesigns (Pepsi, J&J, Walmart, Mastercard). Around each reveal we measure the **abnormal (excess-of-SPY) return** on a short **announce** leg `[+1…+5d]` and a longer **drift** leg `[+6…+126d]`, with a one-day entry lag, then judge each against a Welch *t*, a placebo null sized to the event count, an outlier-drop fragility curve, and a deterministic synthetic power control. We name the honesty problem loudly: the worst-outcome rebrands (Twitter→X private, Weight Watchers→WW bankrupt, Paramount acquired) **left the tape**, biasing the survivor drift *up* — **against** the floundering thesis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a new logo is neither a comeback nor a cry for help, and how two stock outliers (BlackBerry, GM) fake a "rebrand pop" — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | abnormal-return event windows on a rebrand table, announce/drift legs vs zero, a Welch *t* + placebo null, an outlier-fragility curve, the costed renewal trade, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`logo_rebrand/`](logo_rebrand/). The rebrand table is hardcoded & transparent; the priced tape is **survivor-biased** (the worst-outcome rebrands delisted/went private), named on the Signal axis. Sibling study: [389 — Name-Change-Effect](../389-name-change-effect/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
