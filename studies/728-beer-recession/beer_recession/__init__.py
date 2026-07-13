"""Study 728 — "People drink more beer in recessions" (beer stocks as a defensive tilt).

Are Molson Coors (``TAP``) and Boston Beer (``SAM``) *counter-cyclical* — do they hold up
when the market falls, the way the "recession-proof beer" folklore says? We test the two
tradable expressions of the claim against ``SPY``:

1. **Downside beta.** Split the tape into up- and down-market months and measure each
   name's beta on each side. A genuinely defensive stock has a *lower* beta on the way
   down (bear-beta < bull-beta, and < 1).
2. **Recession-window returns.** During the three NBER recessions in the sample (2001,
   2008 GFC, 2020 COVID) do the beer names *beat* SPY — the direct counter-cyclical test?
3. **Could you trade it?** NBER dates a recession only **ex-post** (announced 6–18 months
   late), so a "rotate into beer at the downturn" rule is a look-ahead; and over the full
   sample both names badly lagged SPY.

See :mod:`beer_recession.data` (yfinance month-end Adj Close + the hardcoded, cited NBER
recession windows + a deterministic asymmetric-beta synthetic control) and
:mod:`beer_recession.strategy` (CAGR/vol/MDD, bull/bear beta, Newey-West CAPM alpha, the
recession-window paired *t*, and the opportunity-cost of holding beer "for defense")."""

from . import data, strategy

__all__ = ["data", "strategy"]
