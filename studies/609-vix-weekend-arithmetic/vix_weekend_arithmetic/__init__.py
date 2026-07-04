"""Study 609 — VIX Weekend Arithmetic.

The claim under test: the VIX has a guaranteed weekend seesaw baked into its own
formula — pure calendar-variance arithmetic (30 *calendar* days in the window,
but variance only accrues on *trading* days) forces a day-of-week drift into the
index. We measure the day-of-week pattern on the full ^VIX tape (1990+), race it
against the variance-day-count model's prediction, and ask whether any of it
leaks into a tradable vehicle (VIXY).
"""
