"""Study 47 — Paper-Moon: the Fed Model, a market-timing rule built on a money illusion.

The Fed Model compares the S&P's earnings yield (E/P) to the 10-year Treasury yield: stocks are
"cheap" when E/P > the bond yield, "expensive" when below. We test the timing rule on 125 years of
Shiller data and find it adds nothing buy-and-hold doesn't already give — and that its distinctive
ingredient, the bond-yield comparison, is inert. All the signal is in E/P; the rest is Asness's (2003)
nominal-vs-real confusion.
"""

from . import data, strategy  # noqa: F401
