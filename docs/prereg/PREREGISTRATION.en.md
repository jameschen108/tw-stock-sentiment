# Preregistration: PTT sentiment for 2603 Evergreen and next-day intraday returns

> English translation, added 2026-10-05. The registered document is the Chinese original, [PREREGISTRATION.md](PREREGISTRATION.md), committed in `0d3f9c0` on 2026-09-25 before the final test; its results section was added in `8fe5d4c` the same day. Where the two differ, the Chinese original governs.

Registered: 2026-09-25. This document and `scripts/06_prereg_test.py` were committed **before** running any analysis on 2024 (the final test period).
None of the settings below will be changed after the 2024 results come out; the results, good or bad, are recorded as they are in the last section.

## Background

In the development period (2019-2023) I tried 2330 and 2603 × lexicon / weak labels / LLM-label classifier × close-to-close / open-to-close, about 10 combinations.
Only "2603 + LLM-label classifier + open-to-close" showed a fairly consistent sign. Because it was picked out of many combinations, it needs to be checked on 2024, which has not been looked at.

## Fixed settings

| Item | Setting |
|---|---|
| Stock | 2603 Evergreen Marine (alias 長榮, excluding 長榮航 / 長榮空 / 長榮鋼 / 長榮桂冠) |
| Sentiment model | `data/models/sentiment_clf_llm_2603.joblib`: 1,902 Claude Opus 5.5 labels (2019-2023), character n-gram TF-IDF + logistic regression |
| Daily sentiment | Row t = posts visible before the 09:00 open on day t+1 (averaged within each account first, then across accounts) |
| Target | Log return from open to close on t+1 |
| Low-sentiment day | The day's sentiment ≤ the 40th percentile of the previous 250 trading days (excluding the day itself, at least 60 days) |
| Trading cost | Commission 0.1425% × 2 + day-trade securities transaction tax 0.15%, about 0.435% per round trip; filled at the open and close prices, no slippage |

## Hypotheses and decision rules

- **H1 (primary)**: in `ret_next ~ sent + log_posts + ret + ret_lag1 + ret_lag2 + mkt_ret + vol_20 + vlm_z` (OLS, Newey-West with 5 lags),
  the coefficient on `sent` is > 0, one-sided p < 0.05.
- **H2**: the mean next-day intraday return after low-sentiment days is lower than after other days, Welch t-test, one-sided p < 0.05.
- **H3 (is it profitable)**: "short at the next day's open after a low-sentiment day, buy back at the close" has a total return > 0 after costs.

Decision: if H1 and H2 both pass, the signal is statistically confirmed, and H3 decides whether it is worth trading after costs.
If H1 or H2 fails, the signal does not hold out of sample, and the development-period result is treated as chance under multiple comparisons.

2024 has only about 240 trading days, so power is limited: failing does not mean the effect is 0, but this registration decides by the criteria above.

## Development-period reference values (in-sample; the threshold was chosen after seeing this data, so they are optimistic)

| | Result |
|---|---|
| H1 | Coefficient +0.0084, t = 2.20, one-sided p = 0.014 (n = 1,188) |
| H2 | Low-sentiment days −59.6 bp vs other days −1.4 bp, one-sided p = 0.003 (368 vs 840 days) |
| H3 | Total return +7.3%, Sharpe 0.21, max drawdown −46%, 368 day trades |

## Final test results (2024)

Run on 2026-09-25: `python scripts/06_prereg_test.py --final` (2024-01-02 .. 2024-12-30, 241 days, 102 low-sentiment days).

| | 2024 result | Verdict |
|---|---|---|
| H1 | Coefficient **−0.0048**, t = −0.49, one-sided p = 0.69 (n = 241) | Failed |
| H2 | Low-sentiment days **−4.8 bp** vs other days −21.7 bp (reversed), one-sided p = 0.73 | Failed |
| H3 | Total return **−36.0%**, Sharpe −1.89, 102 day trades, total cost 44% | Failed |

Supplementary (`05_predict.py --final`, not registered): AUC change from adding sentiment, logit −0.014 [−0.058, +0.013], gbm −0.019 [−0.060, +0.038].

**Conclusion**: the development-period sign that low Evergreen sentiment is followed by an intraday drop did not recur at all in 2024; the direction even reversed.
By the registered rule, the signal does not hold out of sample, and the development-period result is treated as chance under multiple comparisons.
