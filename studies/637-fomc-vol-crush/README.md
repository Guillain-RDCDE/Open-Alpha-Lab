# Study 637 — FOMC Vol Crush 📉🕑

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the VIX collapse the afternoon of a Fed decision? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Decision-day ΔVIX **−0.415 pts (−1.97%)** vs +0.014 on all other days — Welch *t* = **−3.51** (points) / **−3.94** (log), Newey-West *t* = **−3.47**, hit rate **65.9%** (Wilson [60.0%, 71.4%]), random-calendar placebo **p = 0.00005** over 261 scheduled decisions, 1994→2026. The crush *sticks* (day +1 ≈ 0) and lands on a day whose **realized** SPY range is *louder* than average (*t* = +4.04) — genuine uncertainty resolution. Caveat: the post-2011 slice alone is *t* = −1.21 (smaller/noisier lately; the decay itself uncertified, *t* = +1.15). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | You can't buy the ^VIX index. The retail echo — SVXY held for the decision day — shows **+48.9 bps/event net** at 5 bps *in sample* but only Welch *t* = **+1.58**, collapsing to **+14.7 bps at *t* = +0.24** since SVXY halved its exposure (2018); worst decision day **−10.5%**, and the vehicle class already gapped to zero once (XIV, Volmageddon). The crush lives in an untradable index; futures pre-price it. |
| **"Retail can bank it with a 1-day SVXY hold"?** | ![Mixed](https://img.shields.io/badge/Retail_capture%3F-Mixed-8b949e?style=flat-square) | Direction right, bankability wrong: 61.9% hit and +49 bps net/event in sample, but **uncertifiable** (*t* = 1.58 < 2), ≈ zero post-2018, and one bad afternoon (−10.5%) erases a year of edge. |

> **In one sentence:** the folklore is right about the physics — over 261 scheduled decisions since 1994 the VIX drops **~0.4 points (~2%)** into the Fed-day close (Welch *t* ≈ −3.9, placebo *p* = 0.00005) and the drop *sticks* — but the index isn't tradable, VIX futures pre-price the event, and the retail short-vol echo (SVXY, *t* = 1.58; *t* = 0.24 post-2018, worst day −10.5%) is a **Mirage**.

## What we tested

We hardcode the **261 scheduled FOMC decision dates 1994-02-04 → 2026-06-17** (Fed calendar
archive; scheduled meetings only — emergency actions are surprises, the opposite of the claim)
and split daily ^VIX changes into decision days vs all 7,916 other days: Welch *t* (points and
log), a Newey-West dummy-regression *t*, a Wilson-bounded hit rate and a 20-seed × 1,000-draw
random-calendar placebo. An event window [−5..+3] tests the pre-meeting run-up (faint,
*t* = 0.90) and the post-day persistence (the crush holds); SPY's same-day high-low range
(*t* = +4.04) shows implied collapsing *while realized is loud* — uncertainty resolution, not a
calm afternoon. The third axis asks the honest retail question: SVXY entered at the prior close
(the calendar is public years ahead — zero look-ahead) and exited at the decision close, net of
one-way costs × 2, with SVXY's survivor status (XIV died 2018) named. A 20-seed synthetic null
plus a planted-crush world proves the machinery. **Dedup:** siblings
[517-pre-fomc-drift](../517-pre-fomc-drift/) (equity *returns* before the meeting),
[67-fed-drift](../67-fed-drift/) (its decayed version), [135-fomc-cycle](../135-fomc-cycle/)
(week-parity returns) and [322-fomc-blackout](../322-fomc-blackout/) (the pre-meeting window)
never test what the **VIX does on the decision day** — this study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why option prices carry a "Fed premium", why it must evaporate at 2 pm on decision day, what actually happens to the VIX — and why you still can't get paid for knowing it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch/HAC splits, the event-window anatomy, the realized-vs-implied cross-check, the era contrast, the SVXY capture test with costs and tails, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`fomc_vol_crush/`](fomc_vol_crush/). The event calendar is hardcoded from the Federal
Reserve's historical FOMC calendars; ^VIX/SPY are indices (no survivorship), SVXY's
survivor status is named on the third axis. **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
