# References -- Study 165 (Chinese-Zodiac)

## Primary literature on Chinese New Year / lunar calendar effects

**Maberly, E. D., & Pierce, R. M. (2003).** "The Halloween Effect and Japanese Equity Prices:
Myth or Exploitable Anomaly?" *Asia-Pacific Financial Markets*, 10(4), 319-334.
Tests seasonal calendar effects on Asian equity markets; finds limited robustness after
multiple-comparisons correction.

**Yuan, K., Zheng, L., & Zhu, Q. (2006).** "Are investors moonstruck? Lunar phases and stock
returns." *Journal of Empirical Finance*, 13(1), 1-23.
Tests lunar calendar effects on 48 markets. Finds small lunar-phase effects that do not
survive transaction costs or out-of-sample testing. Establishes that calendar-label studies
need stringent multiple-comparisons control.

**Chan, M. W. L., Khanthavit, A., & Thomas, H. (1996).** "Seasonality and cultural influences
on four Asian stock markets." *Asia Pacific Journal of Management*, 13(2), 1-24.
Early empirical work testing holiday and lunar calendar effects on Hong Kong, Singapore,
Taiwan, and Thailand equity markets.

**Yen, G., & Shyy, G. (1993).** "Chinese new year effect in Asian stock markets."
*NTU Management Review*, 4(1), 417-436.
One of the earliest systematic papers on the CNY rally hypothesis. Finds positive pre-CNY
returns but does not apply modern multiple-comparisons corrections.

**Wong, W.-K., Agarwal, A., & Du, J. (2005).** "Financial integration for India stock market,
a fractional cointegration approach." *Department of Economics Working Paper*, NUS.
Documents the difficulty of extracting reliable calendar signals from short emerging-market
time series -- the power problem is structural.

## Multiple comparisons in calendar studies

**Sullivan, R., Timmermann, A., & White, H. (2001).** "Dangers of data mining: The case of
calendar effects in stock returns." *Journal of Econometrics*, 105(1), 249-286.
The canonical reference for multiple-comparisons problems in calendar-effect research. Shows
that most documented calendar anomalies disappear when data-mining bias is accounted for
using White's Reality Check. Essential for any zodiac-type multi-category test.

**Bonferroni, C. E. (1936).** "Teoria statistica delle classi e calcolo delle probabilita."
*Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze*, 8,
3-62. Original Bonferroni correction, applied here as p * 12 for a 12-animal simultaneous test.

## Zodiac and Chinese cultural calendar market folklore

**Hirshleifer, D., & Shumway, T. (2003).** "Good day sunshine: Stock returns and the weather."
*Journal of Finance*, 58(3), 1009-1032. Demonstrates that seemingly absurd environmental
variables can show spurious correlations with market returns -- the methodological template
for debunking zodiac claims.

**Jacobsen, B., & Marquering, W. (2008).** "Is it the weather?" *Journal of Banking & Finance*,
32(4), 526-540. Robustness checks on weather and mood effects show that many calendar claims
vanish on out-of-sample data or after Bonferroni correction.

## Data sources and methodology

**CNY dates:** Hong Kong Observatory / Chinese Lunar Calendar official records.
The 1990-2026 Chinese New Year dates and zodiac animal assignments used in this study
are hardcoded from the official Chinese lunisolar calendar computation.

**FXI (iShares China Large-Cap ETF):** BlackRock / Yahoo Finance daily adjusted closes,
2004-present. FXI tracks the 50 largest Chinese companies listed in Hong Kong.

**^GSPC (S&P 500):** Yahoo Finance daily adjusted closes, 1994-present.

## Related desk studies

- [Study 48 -- Groundhog](../../48-groundhog/): Groundhog Day prediction power test (similar
  small-n, fun-claim teardown).
- [Study 136 -- Mark-Twain](../../136-mark-twain/): October effect debunked; shows how
  crash-stripping is the honest robustness test for rare-event calendar claims.
- [Study 55 -- Summer-Lull](../../55-summer-lull/): Halloween indicator -- a calendar
  seasonality with weak but non-zero real evidence.
- [Study 76 -- Rice-Paper](../../76-rice-paper/): Cross-sectional Bonferroni application.
