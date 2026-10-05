# Preregistration: long/short calls in PTT [標的] posts and subsequent stock returns

> English translation, added 2026-10-05. The registered document is the Chinese original, [PREREGISTRATION_TARGET.md](PREREGISTRATION_TARGET.md), committed in `7689f22` on 2026-09-26 before any event return was computed; development results were added in `5c9028b` and final results in `6c7ae59`. Where the two differ, the Chinese original governs.

Registered: 2026-09-26. This document, `scripts/07_target_events.py`, `scripts/08_target_dev.py` and `scripts/10_target_prereg.py`
were committed **before** computing the return of **any** [標的] event (including the development period).
Development results are used only to check that the code is correct and to estimate power. Any change to the settings below must be recorded under "Change log", with a reason, before the final test is run.
The final test results, good or bad, are recorded as they are in the last section.

## Background

The daily sentiment analysis (see README) found that the sentiment of PTT comments mainly reacts to prices that have already moved and has no predictive power for next-day returns.
[標的] posts are different: the author states long or short in the board's required format, usually with analysis and an entry/exit plan, so they are closer to stock recommendations.
In the literature, the social media content with predictive power is mostly this kind of analysis post (for example, Seeking Alpha articles), not chatter.

## Data

| Item | Setting |
|---|---|
| Posts | Original [標的] posts (no Re:), merged by post ID from pttcc (2019-2024) and pttweb (2015/04 to 2025/01, including deleted posts) |
| Development period | Posted 2016-01-01 to 2023-12-31 (2015 posts almost never have the "分類" (category) field) |
| Final test period | Posted 2024-01-01 or later (to the last pttweb post, 2025-01-24) |
| Prices | Daily quotes for all stocks from TWSE MI_INDEX and TPEx dailyQuotes (`scripts/fetch_prices.py`), including delisted stocks |
| Adjustment | TWSE-listed: reference prices from TWT49U for ex-rights/ex-dividend and from TWTAUU for capital reductions; TPEx: the previous day's "next-day reference price"; returns still beyond the price limit are set to missing |
| Market | TWSE-listed: the TAIEX total return index (with dividends); TPEx: the return of TPEx common stocks weighted by the previous day's market cap |

## Events

- The "分類" field in the body is long or short; posts whose title states the opposite direction are excluded.
- The title (or, if it has no match, the "標的：" line of the post template) matches **exactly one** Taiwan stock, and it is a **common stock** (four digits, not starting with 0).
  The ticker must be trading at the time of posting; tickers that look like years (19xx, 20xx) count only if the title also has the stock's name.
- Entry: the first close after posting (a post before 13:30 enters at that day's close; a later post, or one on a non-trading day, enters at the next trading day's close).
- Of the posts by the same author (or with the same title when there is no author) on the same stock in the same direction within 5 trading days, only the first is kept.

## Returns

- Market-adjusted cumulative return `car_h`: the sum, over the h trading days after the entry close, of the stock's adjusted return minus its market's index return, in log returns.
- An event with missing values in the window (delisting, an unadjustable jump) is left out for that horizon.
- Pre-post return `pre_20`: the market-adjusted cumulative return over the 20 trading days ending at the close the day before entry.

## Hypotheses and tests

- **H1 (primary)**: mean `car_5`, long posts − short posts > 0.
- **H2 (primary)**: mean `car_20`, long posts − short posts > 0.
- Test: `car_h ~ constant + long` (OLS), standard errors two-way clustered by entry date and stock, one-sided.
  H1 and H2 are Holm-corrected at an overall α = 0.05: the smaller p is compared with 0.025, and if it passes, the larger with 0.05.
- **H3 (descriptive)**: mean `pre_20`, long posts − short posts ≠ 0 (two-sided, α = 0.05). This checks whether authors chase rallies and sell-offs.

Decision: if H1 or H2 passes after Holm correction, the long/short calls in PTT [標的] posts have out-of-sample predictive power; if neither passes, this is not supported.
Also reported (not a hypothesis): the average return from buying only the long posts, excluding those whose entry close is locked limit-up, after round-trip costs.

## Power

After the development period is run, the smallest difference the final test period can detect (one-sided α = 0.025, 80% power) is computed by scaling the development-period standard errors by sample size,
and filled in here before the final test is run.

The main final-test sample is 596 posts, only 57 of them short (in 2024 only 44% of [標的] posts set the category to long or short, against 66% in the development period).
Scaled from the development-period standard errors, the smallest detectable difference in the final test is about 3.2% at 5 days and about 5.8% at 20 days.
The actual development-period differences are only 0.41% and 0.54%, so under the original design the final test has only about 5-6% power.

## Change log

No changes. After the development results came out (below), I considered switching to a hypothesis with more power, but that idea came only after seeing the development period. On 2026-09-26 I decided to run the final test as originally designed and accept the low power.

## Development-period reference values (in-sample)

`python scripts/08_target_dev.py`: 2016-2023, 10,705 posts (8,709 long, 1,996 short), 1,467 stocks.

| | Long | Short | Difference | t | One-sided p |
|---|---|---|---|---|---|
| H1 `car_5` | −0.11% | −0.51% | +0.41% | 1.99 | 0.023 |
| H2 `car_20` | −0.78% | −1.31% | +0.54% | 1.44 | 0.075 |
| H3 `pre_20` (two-sided) | +4.32% | +5.37% | −1.05% | −1.98 | two-sided 0.048 |

- Almost all of the difference comes from the 498 posts (4.7%) whose entry close was already locked limit-up (long) or limit-down (short): over the next 5 days, long +2.41% and short −6.50%.
  This is price continuation caused by the price limit, and a locked close usually cannot be traded. Excluding them, the difference is +0.09% at 5 days (t = 0.47) and +0.28% at 20 days (t = 0.77).
- Buying only the long posts (excluding limit-up closes): market-adjusted −0.24% at 5 days and −0.93% at 20 days (t = −5.87); after round-trip costs, −0.83% and −1.51%.
  Against stocks with similar liquidity and past returns, though, it trails by only −0.11% at 20 days, so the underperformance against the market is mainly a trait of these stocks.
- Authors chase rallies: in the 5 days before posting, long posts +1.94% and short posts +1.25% (difference t = 2.81); both had already risen 4-5% in the 20 days before posting.
  On the posting day itself (the entry day), long − short is +0.89% (t = 9.1).

## Final test results

Run on 2026-09-26: `python scripts/10_target_prereg.py --final` (596 posts: 539 long, 57 short; entry 2024-01-03 .. 2025-02-03).

| | Long | Short | Difference | t | p | Verdict |
|---|---|---|---|---|---|---|
| H1 `car_5` | −0.48% | +0.84% | −1.32% (reversed) | −1.24 | one-sided 0.89 | Failed |
| H2 `car_20` | −0.46% | −1.90% | +1.44% | 0.71 | one-sided 0.24 | Failed |
| H3 `pre_20` | | | −2.46% | −1.10 | two-sided 0.27 | No significant difference |

Not registered: buying only the long posts (excluding limit-up closes) returns market-adjusted −0.60% at 5 days and −0.83% at 20 days, and −1.19% and −1.41% after round-trip costs,
in the same direction as the development period (−0.24%, −0.93%).

**Conclusion**: neither H1 nor H2 passed, and the 5-day difference even reversed. By the registered rule, this does not support "the long/short calls in [標的] posts have out-of-sample predictive power".
It was known in advance that the final test had low power (only 57 short posts), so this result alone cannot rule out a small effect;
but the development period already showed that the small gap comes from continuation after limit-locked closes and is close to 0 once those are excluded.
