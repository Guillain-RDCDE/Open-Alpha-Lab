# References — Study 09 (Phantom-Kernel)

The claim under test, the model, its extensions, and the empirical literature on order flow
that decides whether the load-bearing assumption holds.

## The claim (the thing we steelman and test)
- **Ruuj (@RuujSs), "How To Implement The Avellaneda-Stoikov Model The Way Serious Market
  Makers Do" (X / Twitter, 2026).** The popular write-up that prompted this study: presents
  the two AS equations as the near-universal foundation of market making and frames *not*
  using them as "leaving serious money on the table." (Notably advertises a "Rust"
  implementation while every code block is Python — a tell worth remembering about
  confidently-presented artefacts.)

## The model and its extensions
- **Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a limit order book."**
  *Quantitative Finance* 8(3), 217–224. The reservation price and optimal-spread equations,
  derived under the exponential arrival kernel `lambda(delta) = A e^{-k delta}`.
- **Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013). "Dealing with the inventory risk:
  a solution to the market making problem."** *Mathematics and Financial Economics* 7(4),
  477–507. Adds hard inventory bounds; turns the HJB into linear ODEs (a true closed form).
- **Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and High-Frequency
  Trading*.** Cambridge University Press. The Cartea–Jaimungal line modelling adverse
  selection / informed trading explicitly — the friction World B switches on.
- **Cartea, Á. & Jaimungal, S. (2015). "Risk metrics and fine tuning of high-frequency trading
  strategies."** *Mathematical Finance.* Inventory and adverse-selection extensions.

## Why the exponential kernel is the wrong shape (the empirical literature)
- **Gopikrishnan, P., Plerou, V., Gabaix, X. & Stanley, H. E. (2000). "Statistical properties
  of share volume traded in financial markets."** *Phys. Rev. E* 62, R4493. Power-law tails
  in traded volume.
- **Gabaix, X., Gopikrishnan, P., Plerou, V. & Stanley, H. E. (2003). "A theory of power-law
  distributions in financial market fluctuations."** *Nature* 423, 267–270. Heavy-tailed
  trade size / order flow — the reason order *reach* is Pareto, not exponential.
- **Bouchaud, J.-P., Farmer, J. D. & Lillo, F. (2009). "How markets slowly digest changes in
  supply and demand."** In *Handbook of Financial Markets.* Order-flow statistics, long
  memory, and the limit-order-book shape; market-order sizes are heavy-tailed.
- **Cont, R. (2001). "Empirical properties of asset returns: stylized facts and statistical
  issues."** *Quantitative Finance* 1, 223–236. Heavy tails and volatility clustering — the
  jumps and stochastic vol of World B.

## Method (shared desk engine)
- **Newey, W. & West, K. (1987).** HAC standard errors. *Econometrica* 55, 703–708.
- **Lo, A. (2002). "The statistics of Sharpe ratios."** *Financial Analysts Journal* 58(4).
- Reproducibility stamp (as-of + content fingerprint): [`quantlab/repro.py`](../../../quantlab/repro.py).
- Bootstrap Sharpe CI: [`quantlab/stats.py`](../../../quantlab/stats.py).
