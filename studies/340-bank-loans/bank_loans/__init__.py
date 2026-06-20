"""Study 340 — Bank-Loans: do floating-rate senior bank loans (BKLN) actually
protect you when rates rise?

The pitch, steelmanned: *"Senior bank loans float with short-term rates, so unlike a
bond fund they barely fall when rates rise — a high-yield income sleeve with almost no
interest-rate risk."* We pit **BKLN** (Invesco Senior Loan ETF, total return) against a
long-duration Treasury (**TLT**), an intermediate Treasury (**IEF**) and equities (**SPY**)
on two axes that decide the claim:

- **Rate sensitivity (the claim's "real" part)** — across rate regimes (and on the worst
  duration days), does BKLN actually shrug off rate moves the way the brochure says?
- **Where the risk hides (the claim's catch)** — floating-rate loans carry *credit* risk,
  not *duration* risk. Does BKLN behave like a safe bond, or like equity-in-disguise that
  detonates exactly when stocks crash?

Distinct from Study 338 (Preferred-Stocks — a *perpetual junior equity-hybrid* identity
test) and Study 97 (60/40 *allocation* race): this is a *duration-vs-credit* decomposition
of a single instrument marketed on its **low rate sensitivity**.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
