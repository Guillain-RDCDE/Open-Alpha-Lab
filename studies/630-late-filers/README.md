# Study 630 — Late Filers (Form NT) 📝

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the NT ("we can't file on time") a sell signal? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | On 1,180 NT events (2004–2026) the post-filing drift is **−105.7 bps**/60 td but the primary calendar-time test reads **HAC *t* = +0.83**, matched self-control Welch *t* = **−0.77** — this tape can't certify it. The panel is severely **survivor-biased** (only **19.2%** of the 89k NT filings map to a still-listed ticker; the delisting disasters that give the claim its teeth are structurally excluded, biasing the drift toward zero). Literature says real; this survivors-only tape alone can't certify it — **Weak**, not Real, not None. |
| **Tradability** — can you short the confession? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Even granting the uncertified drift, the short nets **+14 bps**/event in the friendliest case (10 bps one-way, 3%/yr borrow) and **negative everywhere realistic** — NT filers are small, distressed, hard-to-borrow names, and the fattest payoffs sit in delisted stocks with no borrow at all. |
| **"The SECOND consecutive NT is the real kill signal"?** | ![Busted](https://img.shields.io/badge/Second_NT_the_kill_signal%3F-Busted-8b949e?style=flat-square) | Repeat offenders drift **+58 bps** vs **−482 bps** for first offenses (Welch *t* = +1.85) — the **wrong sign** for the folklore and sub-2 anyway: chronic lateness is priced as routine; whatever information exists is in the *first* confession. |

> **In one sentence:** the accounting literature is right that a late filing is bad news, but on
> the only tape a free desk can build — the 19.2% of NT filers that *survived* to keep a ticker —
> the 60-day post-NT drift (−106 bps, calendar-time HAC *t* = +0.83) never separates from
> small-cap noise, the short is eaten by borrow and spreads, and the "second NT kills" folklore
> is backwards — so **Weak, Mirage, Busted**.

## What we tested

We pulled all **102,505 exact-form NT 10-K / NT 10-Q filings** from the EDGAR quarterly master
index (2002Q1–2026Q2), mapped CIKs to tickers via the SEC registry (the named survivorship
gate), and ran a market-adjusted event study on 1,180 usable events: **short at the close of the
day after the filing** (exactly one execution lag), CAR = Σ daily (stock − SPY) over **60
trading days**, screens stated (entry ≥ $1, no |daily| > 100% glitch windows). Because NT
filings cluster around statutory deadlines, the **primary test is a calendar-time portfolio
with a Newey-West HAC *t***, backed by a matched self-control (same firms 252 td earlier, Welch
*t*) and a **25-seed** random-dates placebo. The third axis Welch-tests repeat offenders vs
first offenses. A deterministic synthetic control with a planted post-NT drift proves the
machinery detects what exists (and stays silent on the null). As-of **2026-06-30**.

Distinct from [565-filing-readability](../565-filing-readability/): that study scores the *text
style* of filings that arrived; this one tests the **filing event** itself — the report that
didn't arrive on time.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Form 12b-25 confession is, the average path after 1,180 of them, why the survivors-only panel biases everything toward zero, and why the short is a mirage — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | calendar-time HAC primary test, matched self-control + 25-seed placebo, NT 10-K/10-Q and window robustness, the repeat-offender third axis, cost × borrow table, and the planted-drift synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`late_filers/`](late_filers/). The signal is the **NT filing event** (EDGAR form-type
index), benchmark SPY, one documented execution lag; **survivorship named on the Signal axis**
(19.2% ticker coverage — the delisted worst offenders are missing, understating the signal).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
