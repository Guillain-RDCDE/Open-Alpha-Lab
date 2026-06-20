# Results — Study 317 (Fed-Balance-Sheet) on the real SPY tape

*The "Don't fight the Fed" test: daily SPY returns sorted by the Fed's announced
balance-sheet direction (QE = expanding, QT = shrinking, FLAT = neither), and a literal
timing rule (long in QE/FLAT, cash-or-short in QT) raced against buy-and-hold. The regime
is the **announced programme direction** — a hand-built table standing in for the
network-blocked FRED `WALCL` series (see [`docs/references.md`](references.md) and the
table in [`fed_balance_sheet/data.py`](../fed_balance_sheet/data.py)). SPY daily,
**split-only / price-only** (not total return). As-of **2026-06-12**; the partial month
of June 2026 is dropped before the stats. Match the fingerprint to confirm you hold the
same tape.*

## Data stamp

| Ticker | Window | Days | Fingerprint |
|---|---|--:|---|
| SPY (split-only) | 1993-01-29 → 2026-05-29 | 8,390 | `d937d577b4c5` |

Regime day-counts: **QE 1,536 · FLAT 5,392 · QT 1,462.**

## The headline — daily SPY return by Fed balance-sheet regime

| Regime | n | mean (bps/day) | vol (bps/day) | ann. Sharpe | HAC *t* |
|---|--:|--:|--:|--:|--:|
| **QE** (expanding) | 1,536 | **+8.35** | 122.7 | 1.08 | **+3.09** |
| FLAT (neither) | 5,391 | +2.40 | 119.6 | 0.32 | +1.75 |
| **QT** (shrinking) | 1,462 | **+5.81** | 101.7 | 0.91 | **+2.35** |

- QE days really did drift up (+8.35 bps/day, *t* = +3.09). So far, so "don't fight the Fed".
- **But QT days *also* drifted up** — +5.81 bps/day at *t* = +2.35. Stocks rose while the Fed
  was shrinking its balance sheet, too.

## The decisive contrast — QE minus QT (circular block bootstrap, 5,000 reps, block=21)

| Contrast | point (bps/day) | 95% CI | bootstrap *p* |
|---|--:|--:|--:|
| **QE − QT** | **+2.54** | **[−4.10, +9.15]** | **0.47** |
| QE − FLAT | +5.95 | [+0.36, +11.35] | 0.038 |

The whole slogan rests on QE beating QT. **It doesn't, to any reasonable confidence:** the
QE−QT gap is +2.54 bps/day with a CI that comfortably straddles zero (*p* = 0.47). The only
contrast that even flickers is QE − FLAT (the Fed eases hardest precisely in/after crashes,
when the rebound is mechanical) — and that is the rebound, not a QE/QT signal.

## Could you trade it? — the rule vs buy-and-hold (excess-of-cash Sharpe)

Position by regime: long in QE and FLAT; **cash** in QT (conservative variant) or **short**
in QT (aggressive variant). One execution lag, applied once; costs one-way × NAV on turnover
(the rule trades ~0.1–0.2 times/yr, so costs are irrelevant here). Both legs are netted of
the cash rate before Sharpe, so this is an excess-vs-excess race.

| Rule | net ann. return | net Sharpe | HAC *t* | vs buy-and-hold |
|---|--:|--:|--:|--:|
| Buy-and-hold SPY | **+10.83%** | **0.552** | — | — |
| Cash in QT (5 bps) | +8.05% | 0.446 | +3.03 | **−2.78%/yr, ΔSharpe −0.11** |
| **Short in QT (5 bps)** | **+5.33%** | **0.279** | +1.86 | **−5.50%/yr, ΔSharpe −0.27** |

The timing rule's own return is statistically positive (it is mostly long SPY), but **it
loses the race outright**: sitting in cash during QT cost ~2.8%/yr, and *shorting* during QT
— literally "fighting the Fed when it tightens" — nearly halved the return. Costs do not
move the verdict (turnover is a fraction of a trade per year); the rule fails on the simplest
test of all: it underperforms doing nothing.

## Synthetic positive control — the engine is a faithful detector

The same machinery, on a deterministic synthetic tape that plants a real QE>QT drift, banks
it; on the null (no regime edge) it correctly finds nothing. A synthetic number can never
back a real stamp — this only proves the harness works.

| Synthetic tape (qe_edge) | QE *t* | QT *t* | QE−QT diff (bps) | 95% CI | signal? |
|---|--:|--:|--:|--:|:--:|
| 0.000 (null) | −0.47 | −0.95 | +3.28 | [−7.71, +14.05] | no (CI brackets 0) |
| 0.0005 | +1.20 | −1.96 | +13.27 | [+2.29, +24.05] | yes |
| 0.0010 (control) | +2.88 | −2.98 | +23.27 | [+12.29, +34.04] | yes |

## Verdict

- **Signal — NONE.** The claim is QE beats QT. The QE−QT daily-return gap is +2.54 bps,
  95% CI [−4.10, +9.15], bootstrap *p* = 0.47 — indistinguishable from zero. Stocks rose in
  *both* regimes (QT *t* = +2.35). The contrast the slogan depends on is not on the tape.
- **Tradability — MIRAGE.** The "fight the Fed in QT" rule underperforms buy-and-hold by
  ~2.8%/yr in cash and ~5.5%/yr short, at lower Sharpe — and that is *before* this is just
  the long-run equity premium you were always paid. There is nothing to harvest.
- **"Don't fight the Fed"? — BUSTED.** Over 33 years, fighting the Fed when it tightened
  would have *cost* you. The slogan confuses "the Fed eases into crashes (and stocks then
  rebound)" with "QE drives stocks and QT sinks them". The second claim is false here.
