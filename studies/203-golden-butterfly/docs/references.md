# Literature map — Study 203 (Golden-Butterfly)

## Original source

1. **Tyler (PortfolioCharts.com, 2015).** *"The Golden Butterfly Portfolio."*
   https://portfoliocharts.com/portfolios/golden-butterfly/. The originating
   design: 20% SPY (large-cap blend), 20% IWN (small-cap value), 20% TLT (long
   Treasuries), 20% SHY (short Treasuries / cash), 20% GLD (gold). Explicitly
   derived from Browne's Permanent Portfolio with the equity leg split into
   large-cap and small-cap-value and the cash leg split into short and long
   Treasuries. Tyler's in-sample Sharpe claim is over the 1970-present window
   using index proxies pre-ETF, a substantially more favourable inflation period
   for gold and small-cap-value than 2004-2026.

## Parent design

2. **Browne, H. (1999).** *Fail-Safe Investing: Lifelong Financial Security in
   30 Minutes.* St. Martin's Griffin. The Permanent Portfolio (25/25/25/25
   stocks/gold/bonds/cash) is the direct parent of the Golden Butterfly. The
   regime logic (prosperity/inflation/deflation/recession) is inherited unchanged;
   the GB's innovation is the equity split and the bond maturity spread.
   Contrasted directly in Study 144 of this repo.

## The small-cap-value premium

3. **Fama, E., & French, K. (1992).** "The Cross-Section of Expected Stock
   Returns." *Journal of Finance* 47(2), 427-465. The canonical three-factor
   model: the size (SMB) and value (HML) factors document a persistent premium
   for small-cap value stocks over large-cap growth, the justification for the
   GB's IWN leg. The premium is the central bet the GB makes beyond the PP.

4. **Davis, J., Fama, E., & French, K. (2000).** "Characteristics, Covariances,
   and Average Returns: 1929 to 1997." *Journal of Finance* 55(1), 389-406.
   Long-horizon evidence showing the small-cap-value premium persisting over
   multiple business cycles. The post-publication evidence is weaker: the premium
   was strong 1963-1990, mixed 1990-2006, and recovered partly 2016-2022.

5. **Asness, C., Frazzini, A., Israel, R., & Moskowitz, T. (2015).** "Fact,
   Fiction and Value Investing." *Journal of Portfolio Management* 42(1), 34-52.
   AQR's analysis of value investing's real-world feasibility including turnover,
   implementation shortfall, and the small-cap-value interaction. Relevant to the
   GB's IWN leg's implementability at scale.

6. **Fama, E., & French, K. (2021).** "The Value Premium." *Review of Asset
   Pricing Studies* 11(1), 105-121. Revisits the value premium post-publication.
   Finds the premium attenuated but surviving in international data; in the US
   the 2007-2020 underperformance of value vs growth is the key headwind for
   the GB's SCV leg in the same window we test.

## Empirical comparisons

7. **Portfolio Visualizer (2024).** "Backtest Portfolio Asset Allocation — Golden
   Butterfly." https://www.portfoliovisualizer.com. Interactive comparison across
   lazy portfolios; confirms the GB's strong historical Sharpe in the 1970-2020
   window and notes the 2010s decade as the GB's weakest relative period (growth
   dominated small-cap value, dragging the SCV leg).

8. **Meb Faber Research Blog (2015).** "Comparing the Best Lazy Portfolios."
   https://mebfaber.com. Compares the Golden Butterfly alongside the PP, 60/40,
   All-Weather, and Ivy Portfolio across overlapping historical windows; finds
   the GB is near-optimal on Ulcer Index (a drawdown-adjusted risk metric) in
   the pre-2004 proxy backtests. Caveat: pre-GLD data uses gold futures proxies.

9. **Bernstein, W. (2010).** *The Investor's Manifesto.* Wiley. Chapter on lazy
   portfolios covers the PP's regime logic and the academic case for the value
   premium; provides context for the GB's design philosophy and its limitations
   in a concentrated index-growth environment like 2013-2020.

10. **Antonacci, G. (2014).** *Dual Momentum Investing.* McGraw-Hill. Chapter 4
    benchmarks the PP and related static portfolios against momentum-based
    alternatives; finds the PP's Sharpe advantage over 60/40 is robust but its
    CAGR shortfall is not offset by the value premium in small-cap-value tilts.
    The GB's relative underperformance in momentum-driven bull markets (2013,
    2019) is directly predicted by Antonacci's analysis.

## Risk-parity context

11. **Asness, C., Frazzini, A., & Pedersen, L. (2012).** "Leverage Aversion and
    Risk Parity." *Financial Analysts Journal* 68(1). The institutional cousin
    of both the PP and the GB; risk-parity weights assets by inverse volatility
    rather than equally. The GB's equal-weight approach is simpler but less
    Sharpe-optimal; the relative Sharpe of PP vs GB in this study (0.782 vs
    0.682) is partially explained by the PP's higher effective risk-parity weight
    on the less-volatile legs (TLT, SHY).
