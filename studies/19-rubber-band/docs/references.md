# References & literature map — Study 19 (Rubber-Band)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). The relevant entry is **strategy §4.4, "Mean
  reversion (ETFs)"**, which defines the Internal Bar Strength indicator `IBS = (Close − Low) / (High −
  Low)` on the prior day's bar and builds a dollar-neutral book that buys the lowest-IBS ETFs and sells
  the highest. The book describes the construction; it does not backtest it — so the entry is the
  hypothesis we put through the protocol.

## The claim under test — the steelman

- **Internal Bar Strength as a short-term reversal signal.** Popularised for ETFs by Larry Connors &
  Cesar Alvarez (*Short Term Trading Strategies That Work*, and the broader Connors Research body of
  work on RSI(2) / IBS mean reversion). The empirical regularity: a bar that closes near its low (IBS ≈
  0) has a positive expected next-day return, one closing near its high (IBS ≈ 1) a lower one. The
  mechanism is short-horizon mean reversion / liquidity provision — being paid to absorb end-of-day
  selling pressure.

- **Why a daily reversal can be real.** Short-horizon return reversal is one of the oldest documented
  market regularities: Bruce Lehmann, *"Fads, Martingales, and Market Efficiency"*, **Quarterly Journal
  of Economics** 1990; Andrew Lo & Craig MacKinlay, *"When Are Contrarian Profits Due to Stock Market
  Overreaction?"*, **Review of Financial Studies** 1990. Reversal profits are genuine in gross returns —
  the question has always been whether they are anything but compensation for liquidity provision once
  costs are paid.

## The honest counters — why the verdict is `REAL` / `MIRAGE` / `DECAYED`

- **Reversal profits are eaten by the bid-ask spread.** The canonical critique of short-horizon
  contrarian strategies (Lo–MacKinlay 1990; and the bid-ask-bounce literature, Roll 1984): a large part
  of measured reversal is the bid-ask bounce, and what remains is a liquidity premium consumed by the
  spread you pay to trade it daily. `decompose.breakeven_cost` and the per-name `extension` table make
  this concrete — the edge lives inside a handful of bps.

- **Post-publication decay.** McLean & Pontiff, *"Does Academic Research Destroy Stock Return
  Predictability?"*, **Journal of Finance** 2016: published anomalies weaken markedly after they become
  known. IBS / short-term-reversal is a heavily-publicised retail signal, and the study's first-half →
  second-half → last-5-year Sharpe collapse is exactly that decay.

- **Adverse liquidity gradient.** The bounce is strongest in the most volatile, least liquid
  instruments (here, thin country ETFs) — precisely the names with the *widest* spreads — so the edge
  and the cost to harvest it scale together. The per-name break-even table is the evidence.

## The desk's own method — engine and reproducibility

- **HAC / Newey–West inference.** Newey & West, *Econometrica* 1987 — the autocorrelation-robust *t*
  on the daily timing stream (`decompose.mean_tstat_hac`).
- **Reproducibility.** Headline numbers are pinned with
  [`quantlab.repro`](../../../quantlab/repro.py) (an as-of date + a content fingerprint of the basket
  closes), so a re-run that matches the fingerprint holds the same tape.

## Caveats stated in the open (house rule)

- **Split-only OHLC.** IBS is a one-day, intraday-shape signal and the book holds at most overnight, so
  dividends (a slow total-return effect) are immaterial — and split-only keeps the OHLC bar internally
  consistent. Stated, not hidden.
- **Basket composition grows over time.** The equal-weight basket averages only the ETFs that have
  listed by each date, so the early sample leans on a few names (SPY/QQQ); the *recent* sub-samples are
  the tradability-relevant ones, which is where the decay verdict is read.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
