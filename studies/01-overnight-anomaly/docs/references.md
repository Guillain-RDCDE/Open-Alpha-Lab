# References & literature map

Citation style: **author–date** (Chicago / *Journal of Financial Economics*
convention). In-text citations elsewhere in this repository take the form
*(Lou, Polk, and Skouras 2019)*; machine-readable entries are in
[`references.bib`](../references.bib). PDFs of the openly available sources can
be fetched with [`python papers/download_papers.py`](../papers/README.md).

---

## Where this work sits in the literature

The night-vs-day return gap is an established empirical fact with several
competing explanations. Keeping them straight is the whole point of this
repository; conflating them is how a real anomaly gets oversold.

**1. Documenting the phenomenon.**
Cooper, Cliff, and Gulen (2008) and Berkman et al. (2012) establish that the US
equity premium accrues overnight, with intraday returns flat to negative.
Knuteson (2019, 2020) extends the observation across world markets and decades.

**2. Competing explanations for *why*.**

| Mechanism | Key reference | One-line claim |
|---|---|---|
| Investor clientele / demand | Lou, Polk, and Skouras (2019) | different clienteles trade at the open vs the close, splitting returns by horizon |
| Attention & the open auction | Berkman et al. (2012) | retail attention bids up the open, a hidden cost paid by open-buyers |
| Funding / dealer inventory | Boyarchenko, Larsen, and Whelan (2023) | the drift is a microstructure/funding phenomenon concentrated near the open |
| Retail trading | Haghani, Ragulin, and Dewey (2024) | retail flow explains the drift in indices, meme stocks and Bitcoin |
| Market-specific microstructure | Qiao and Dam (2020) | China's "T+1" rule *inverts* the pattern — a decisive natural experiment |
| Orchestrated manipulation | Knuteson (2019, 2020, 2022, 2023) | a large quant firm expands/contracts its book to harvest the open |

**3. This repository's position.**
The fact is real and **already multiply-explained** by clientele and
microstructure mechanisms; the inverted Chinese case (Qiao and Dam 2020) and the
listing-clock inversion of foreign-market ETFs are hard to reconcile with a
single global manipulator. Most of the measured "overnight return" is gap
**beta**, not alpha, and net of realistic execution costs the residual edge does
not survive. We therefore treat Knuteson's empirical contribution as genuine
while regarding the manipulation *attribution* as unproven.

---

## Methodology

The statistical and microstructure machinery used in
[`02_for_the_quants.ipynb`](../notebooks/02_for_the_quants.ipynb) and
[`quantlab/analytics.py`](../../../quantlab/analytics.py):

Almgren, Robert, Chee Thum, Emmanuel Hauptmann, and Hong Li. 2005. "Direct
Estimation of Equity Market Impact." *Risk* 18 (7). *(square-root impact law —
capacity analysis)*

Lo, Andrew W. 2002. "The Statistics of Sharpe Ratios." *Financial Analysts
Journal* 58 (4): 36–52. https://doi.org/10.2469/faj.v58.n4.2453. *(Sharpe-ratio
standard errors)*

McLean, R. David, and Jeffrey Pontiff. 2016. "Does Academic Research Destroy
Stock Return Predictability?" *The Journal of Finance* 71 (1): 5–32.
https://doi.org/10.1111/jofi.12365. *(post-publication alpha decay)*

Newey, Whitney K., and Kenneth D. West. 1987. "A Simple, Positive Semi-Definite,
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
*Econometrica* 55 (3): 703–708. https://doi.org/10.2307/1913610. *(HAC
standard errors)*

## Subject-matter references

Berkman, Henk, Paul D. Koch, Laura Tuttle, and Ying Jenny Zhang. 2012. "Paying
Attention: Overnight Returns and the Hidden Cost of Buying at the Open."
*Journal of Financial and Quantitative Analysis* 47 (4): 715–741.

Boyarchenko, Nina, Lars C. Larsen, and Paul Whelan. 2023. "The Overnight Drift."
*The Review of Financial Studies* 36 (9): 3502–3547.
https://doi.org/10.1093/rfs/hhad020. Working paper: Federal Reserve Bank of New
York Staff Reports, no. 917 (2020).

Cooper, Michael J., Michael T. Cliff, and Huseyin Gulen. 2008. "Return
Differences between Trading and Non-Trading Hours: Like Night and Day." Working
paper. https://ssrn.com/abstract=1004081.

Haghani, Victor, Vladimir Ragulin, and Richard Dewey. 2024. "Night Moves: Is the
Overnight Drift the Grandmother of All Market Anomalies?" *Journal of Investment
Management* 22 (2). Working paper (2022): https://ssrn.com/abstract=4139328.

Knuteson, Bruce. 2019. "Celebrating Three Decades of Worldwide Stock Market
Manipulation." arXiv:1912.01708. https://arxiv.org/abs/1912.01708.

Knuteson, Bruce. 2020. "Strikingly Suspicious Overnight and Intraday Returns."
arXiv:2010.01727. https://arxiv.org/abs/2010.01727.

Knuteson, Bruce. 2022. "They Still Haven't Told You." arXiv:2201.00223.
https://arxiv.org/abs/2201.00223.

Knuteson, Bruce. 2023. "Nothing to See Here: How to Say It When You Need to."
SSRN Working Paper 4619084. https://ssrn.com/abstract=4619084.

Lou, Dong, Christopher Polk, and Spyros Skouras. 2019. "A Tug of War: Overnight
Versus Intraday Expected Returns." *Journal of Financial Economics* 134 (1):
192–213. https://doi.org/10.1016/j.jfineco.2019.03.011.

Qiao, Kenan, and Lammertjan Dam. 2020. "The Overnight Return Puzzle and the
'T+1' Trading Rule in Chinese Stock Markets." *Journal of Financial Markets* 50:
100534. https://doi.org/10.1016/j.finmar.2020.100534.
