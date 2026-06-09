"""Study 19 — Rubber-Band: does a stock that closes near its low really snap back tomorrow?

The second study mined from Kakushadze & Serur's *151 Trading Strategies* (strategy 4.4, ETF mean
reversion via Internal Bar Strength). The steelman is a beloved short-term reversal signal: IBS =
(Close - Low) / (High - Low), and a low-IBS day (closed near its low) tends to bounce the next session
(Connors & Alvarez). We run it through the desk's protocol and split, as ever, "is the bounce real?"
from "can you bank it?". The reusable pieces, in the desk's usual split:

    * :mod:`data` — daily OHLC bars: a synthetic generator with a *baked-in* IBS->next-day reversal
      (low close today, positive return tomorrow, by construction), plus a cache-only reader for a
      basket of liquid ETFs. The null (kappa=0) is a random walk where IBS is uninformative.
    * :mod:`ibs` — the signal and the engine: :func:`ibs.ibs` and the load-bearing
      :func:`ibs.reversal_strength` -- bucket days by IBS, read the average next-day return, and ask
      whether a low close really earns more than a high one.
    * :mod:`strategy` — the investable books: a single-asset timing overlay (``w = 1 - 2*IBS``, long
      after a low close), its equal-weight basket average, and the §4.4 cross-sectional dollar-neutral
      book -- all turning over ~daily, so the cost model is the protagonist.
    * :mod:`decompose` — the inference: a **Newey-West** t-stat that the bounce is real, the
      **break-even cost** (the slippage that zeroes the net Sharpe), and a bootstrap Sharpe CI. The
      verdict it lands: Signal `REAL`, Tradability `MIRAGE` -- a genuine bounce that lives entirely
      inside the bid-ask spread.
    * :mod:`extension` — the beat-7 worked complement: the **realistic-spread test**, name by name --
      even the ETF with the strongest gross bounce breaks even below the spread it actually trades at.
"""
