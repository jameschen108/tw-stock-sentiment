# Preregistration: pre-open PTT discussion volume and same-day trading volume and range

> English translation, added 2026-10-05. The registered document is the Chinese original, [PREREGISTRATION_VOLUME.md](PREREGISTRATION_VOLUME.md), committed in `b386158` on 2026-09-26 before any 2024 volume or range result was computed; its results section was added in `810b934`. Where the two differ, the Chinese original governs.

Registered: 2026-09-26. This document, `scripts/11_volume_prereg.py` and `src/pttsent/volume.py` were committed **before** computing any 2024 trading volume or range result;
`--final` first checks that these three files have no uncommitted changes, and refuses to run otherwise.
The final test is run only once. Any change to the settings below must be recorded under "Change log", with a reason, before the final test is run. The results, good or bad, are recorded as they are in the last section.

## Background

Both the daily sentiment and the [標的] studies found that PTT has no predictive power for later returns that holds up out of sample (see README).
In the literature, attention more often predicts trading volume and volatility. But point 1 of "Next steps" in the README records a simple version:
daily post counts with the day cut at 13:30, used to predict the next day's |return|, added nothing beyond past volatility, and nothing for volume either.

Post counts cut at 13:30 mostly react to that day's intraday price and volume, which the baseline model already has.
Discussion between the close and the next open happens after the last trade, and the baseline model has not seen it.
Exploration in the development period (below) found that discussion in this window adds information about same-day volume, but about 3/4 of TSMC's signal reflects the ADR's overnight move.
So the baseline model has to include the public overnight information available before the open, and the question is whether PTT adds anything on top.

## What 2024 has already been used for

2024 is the final test period of the two earlier preregistrations ([PREREGISTRATION.en.md](PREREGISTRATION.en.md), [PREREGISTRATION_TARGET.en.md](PREREGISTRATION_TARGET.en.md)), and what was looked at there was **returns**:
how sentiment and [標的] posts relate to next-day or later returns. 2024 daily post counts (cut at 13:30) appeared as a control or feature in those return models
(the H1 regression in `06_prereg_test.py`, and `05_predict.py --final`), but the only outputs were the sentiment coefficient and the AUC of the whole model.
The relation between post counts and trading volume or range in 2024 was never computed; all exploration code here used only rows with target dates before 2024-01-01.
So 2024 is still unseen data for this new hypothesis. A second check on PTT data from 2025 onward can be done later and is outside the scope of this registration.

## Fixed settings

| Item | Setting |
|---|---|
| Stocks | 2330 TSMC, 2603 Evergreen Marine, 2317 Hon Hai |
| Trading days | TWSE trading days in `data/prices/panel.parquet`; one row is trading day T, and the forecast is made before T's open (09:00) |
| Primary target | `y_lv` = log(shares traded on T) |
| Secondary target | `y_vol` = log(ln(T's high / T's low)); days with high = low (locked all day) are missing |
| Pre-open discussion `pre_abn` | The number of texts in `data/interim/texts_{ticker}.parquet` (`01_prepare_ptt.py`: posts whose title mentions the ticker or an alias with their whole comment thread, plus comments that mention it themselves) in [T−1 13:30, T 09:00), log1p, minus the mean of the same window over the previous 20 trading days; days whose 20-day window falls partly before 2019-01-01 are missing |
| Baseline: recent price and volume | y on the previous day, its 5-day mean and its 22-day mean (HAR); the other target on the previous day; the absolute value and the negative part of the previous day's return |
| Baseline: calendar | Tuesday to Friday dummies; log calendar days since the previous trading day; TAIEX futures settlement day (third Wednesday of each month, moved to the next trading day on holidays); MSCI rebalance day (last trading day of February, May, August and November) |
| Baseline: TAIEX futures night session | T's night session (T−1 15:00 to T 05:00, the monthly contract with the largest volume): log range, \|return\| against the same contract's previous day-session close, log volume minus the mean of the previous 20 night sessions |
| Baseline: US stocks | Trading days of TSM (TSMC's ADR) and ^SOX (Philadelphia Semiconductor Index) that close after T−1's close and before T's open (accumulated over Taiwan holidays): \|cumulative return\|, the negative part of the cumulative return, Parkinson volatility; for TSM also log volume minus its 20-day mean; when there is no new US trading day, these are 0 and a flag is added |
| Baseline: news | cnyes headlines tagged with the stock or with an alias in the title (2330: 台積電 / 台積 / TSMC; 2603: 長榮, after removing 長榮航 / 長榮空 / 長榮鋼 / 長榮桂冠; 2317: 鴻海), counted in [T−1 13:30, T 09:00), log1p, minus the mean over the previous 20 trading days |
| Baseline: stock events | Previous close at limit-up / limit-down (`limit_up_close`, `limit_down_close` in `panel.parquet`: close ≥ base price × 1.095 or ≤ base price × 0.905); T within a disposition period (TWSE's list of disposition securities, announced after the previous trading day's close); T an ex-rights/ex-dividend day (TWSE TWT49U). For a stock where an event never happened, the column is all 0 and so is its coefficient |
| Model | OLS with an expanding window: in the evaluation period it is re-estimated every 21 trading days, using only rows before the start of each block (at least 250 rows); the baseline and the baseline plus `pre_abn` use the same sample and the same re-estimation points |
| Test | Clark-West adjusted loss difference f = e0² − (e1² − (ŷ0 − ŷ1)²), where 0 is the baseline and 1 adds `pre_abn`; one-sided t-test of mean > 0, Newey-West with 5 lags |
| Final test period | Target dates 2024-01-01 to 2024-12-31 (242 trading days); training data is every row from 2019-02 up to the start of each block |
| US data | `scripts/fetch_us.py` (Yahoo Finance chart API, 2014-01-02 .. 2025-06-30, 2,890 days each, downloaded 2026-09-26), sha256 below the table |

`data/` is not in git, so the US files used here are recorded below:
`TSM.json` sha256 `1bd3c51007326b5fd57d4eecd994ae0cbbacf337ef433042eab695d48f95df22`,
`SOX.json` sha256 `24377f3f5393d83b95e1c8401f6f00020b8ff0d3c59f8ee349e4a613fb08ff63`.

## Hypotheses and decision rules

- **H1 (primary)**: the mean of f for `y_lv`, averaged across the three stocks on common trading days, is > 0, one-sided p < 0.05.
- **H2 (secondary)**: for each stock, the mean f for `y_lv` is > 0; the three p-values are Holm-corrected at an overall α = 0.05.
- **H3 (secondary)**: the mean f for `y_vol` across the three stocks is > 0, one-sided p < 0.05.

Decision:
- If H1 passes: pre-open PTT discussion volume has out-of-sample incremental predictive power for same-day trading volume, beyond the public overnight information available before the open (night session, ADR, SOX, news volume) and the known calendar and stock events.
- If H1 fails: the conclusion above is not supported, and the development-period result is treated as chance from exploration or over-optimism. Power is limited (next section), so failing does not mean the effect is 0.
- H2 and H3 are reported separately and do not change the H1 verdict.

Also reported (not hypotheses): each stock's out-of-sample MSE change, the regression coefficient of `pre_abn` over the evaluation period (Newey-West) and the effect of 1 standard deviation,
and the increment after also adding T's opening gap (known only after the open). The opening gap is the market's sum of all the overnight news:
if H1 passes but the pooled CW test is not significant after adding the gap (one-sided p ≥ 0.05), this is read as PTT discussion volume mainly reflecting the size of the overnight news rather than extra attention.

## Power

Scaling the development-period loss differences to the 242 trading days of 2024 (one-sided α = 0.05):

| | Expected t | Power |
|---|---|---|
| H1 pooled log volume | 2.82 | 88% |
| H2 2330 / 2603 / 2317 log volume (uncorrected) | 1.21 / 2.02 / 1.69 | 33% / 65% / 52% |
| H3 pooled log range | 1.63 | 50% |

This is an upper bound: the development-period effect was settled on after seeing the results of about 20 variations, so the true effect is probably smaller.
If the effect is only half the development-period size, H1's power is about 41%.

## Exploration done in the development period

All of the following used only rows with target dates before 2024-01-01, and all of it was done before this registration was written:

1. Earlier: post counts cut at 13:30 predicting the next day's |return| added nothing for any of the three stocks (README "Next steps", point 1).
2. The 13:30 window and the pre-open window × two baselines (recent price and volume plus weekday, then also the night session) × two targets × three stocks; also whole-board post counts predicting the TAIEX futures day session.
   Range showed an increment only after controlling for the night session, and the volume increment also grew after that control. This looked like a suppression effect and was not adopted.
3. Robustness of the pre-open window: adding cnyes news volume, counting only the evening after the close or only from midnight to 09:00, counting unique accounts instead, counting posts only, excluding Mondays and days after long holidays,
   splitting into 2020-21 and 2022-23, adding settlement and MSCI rebalance days.
4. Adding TSM and SOX (TSMC's volume increment shrank from −3.96% to −0.96%), adding the opening gap, estimating power.

The pre-open window was chosen after seeing that the 13:30 version added nothing, with information timing as the reason; it was not a hypothesis formed before looking at any data.

## Differences between the final code and the exploration version

These were fixed before running the final test, and none of them was chosen based on development results:

- The exploration version treated 2018 post counts (no PTT data) as 0, which inflated `pre_abn` in January 2019 (TSMC mean 0.95); the final version sets those days to missing.
- The exploration version dropped missing values separately for the two models, so their samples and re-estimation points could differ; the final version uses the same sample.
- The calendar is TWSE trading days rather than each stock's own trading days (this affects only the few days in 2022-09 when Evergreen was suspended for its capital reduction).
- The night-session return is measured against the same contract's previous day-session close (the exploration version used the highest-volume contract in each session, which mixes in the spread at rollover).
- Settlement days move to the next trading day on holidays (the exploration version took the month's third Wednesday with trading).

After these fixes (and before adding the controls in the next section), the volume results are close to the exploration version (TSMC −0.87% → −0.71%, Evergreen −1.90% → −1.77%, Hon Hai −1.17% → −1.09%),
but TSMC's range increment went from −0.66% to −0.19%, and putting the January 2019 artifact back does not restore the original number. This result is sensitive to details and unstable.

## Controls added just before registration

After the draft was done and the reference values at that point had been seen, three more controls were added that affect volume but do not show up in the opening gap:
a previous close at limit-up / limit-down (unfilled orders carry over to the next day), disposition periods (call-auction matching at intervals lowers volume), and ex-rights/ex-dividend days. The decision to add them was made before seeing any result with them.
In the development period, Evergreen closed limit-up on 29 days and limit-down on 18, and was under disposition for 20 days (2021-06-18 to 07-20); TSMC and Hon Hai each closed at a limit on only 1-2 days and were never under disposition;
ex-rights/ex-dividend days numbered 19 for TSMC, 4 for Evergreen and 5 for Hon Hai.

With them added, the results barely changed: pooled volume CW t 5.48 → 5.48; TSMC −0.71% → −0.69%, Evergreen −1.77% → −1.90%, Hon Hai −1.09% → −1.08%.
Evergreen's increment even grew, so it is not propped up by limit closes or disposition days. The reference values in the next section include these three controls.

## Development-period reference values (in-sample, optimistic)

`python scripts/11_volume_prereg.py`: evaluation 2020-02-21 .. 2023-12-29 (916 days common to all three; 945 days for TSMC and Hon Hai).

| | CW t | One-sided p | Out-of-sample MSE change | Effect of 1 SD | MSE change with the gap added (CW t) |
|---|---|---|---|---|---|
| **H1** pooled log volume | +5.48 | < 0.0001 | | | (+4.56) |
| H2 2330 log volume | +2.40 | 0.008 | −0.69% | +4.5% | −0.39% (+1.90) |
| H2 2603 log volume | +3.92 | < 0.0001 | −1.90% | +8.4% | −1.47% (+3.44) |
| H2 2317 log volume | +3.33 | 0.0004 | −1.08% | +5.8% | −0.80% (+2.91) |
| **H3** pooled log range | +3.18 | 0.0007 | | | |
| 2330 log range | +1.30 | 0.10 | −0.19% | +5.2% | −0.15% (+1.30) |
| 2603 log range | +3.76 | < 0.0001 | −1.43% | +7.0% | −1.26% (+3.62) |
| 2317 log range | −0.37 | 0.64 | +0.17% | +1.7% | +0.11% (−0.51) |

"Effect of 1 SD": how much higher volume (or range) is, in %, when `pre_abn` is 1 standard deviation higher, converted from the regression coefficient over the evaluation period.

Not yet controlled, and possibly shrinking the signal: freight-rate indices and shipping stocks for Evergreen, overnight news on stocks such as Apple and Nvidia for Hon Hai, and earnings calls and monthly revenue release dates.
This news would show up in prices, so the opening-gap check covers part of it; the data is also hard to get (Yahoo has no freight-rate index, and the monthly revenue data on hand has only the month, not the release date), so these were not added.

## Change log

No changes.

## Final test results

Run on 2026-09-26, after preregistration commit `b386158`: `python scripts/11_volume_prereg.py --final` (target dates 2024-01-02 .. 2024-12-31, 242 days for each stock).

| | CW t | One-sided p | Out-of-sample MSE change | Effect of 1 SD | Verdict |
|---|---|---|---|---|---|
| **H1** pooled log volume | +3.76 | 0.0001 | | | **Passed** |
| H2 2330 log volume | +1.67 | 0.047 | −1.26% | +3.6% | Failed (Holm, needs < 0.025) |
| H2 2603 log volume | +2.63 | 0.004 | −2.96% | +10.5% | Passed (Holm) |
| H2 2317 log volume | +1.65 | 0.050 | −1.10% | +4.4% | Failed (Holm) |
| **H3** pooled log range | +2.92 | 0.002 | | | **Passed** |
| 2330 log range (descriptive) | +1.03 | 0.15 | −0.34% | +1.6% | |
| 2603 log range (descriptive) | +2.54 | 0.006 | −2.61% | +6.6% | |
| 2317 log range (descriptive) | +0.75 | 0.23 | −0.24% | +1.3% | |

For interpretation (not a hypothesis): after also adding the opening gap, the pooled log-volume CW t = +3.19 (one-sided p = 0.0007), still significant;
the MSE changes are 2330 −0.73%, 2603 −2.19%, 2317 −1.15%.

**Conclusion**: H1 passed. Pre-open PTT discussion volume has out-of-sample incremental predictive power for same-day trading volume beyond the night session, the TSMC ADR, SOX, news volume and the known calendar and stock events.
By the registered interpretation rule it stays significant after adding the opening gap, so it reflects more than the size of the overnight news.
The effect is small: out-of-sample MSE falls 1-3% across the three stocks, most clearly for Evergreen. Individually only Evergreen passes after Holm correction; TSMC and Hon Hai point the same way,
with p around 0.05 on their own, but miss the corrected threshold. Range (H3) also passes, mainly because of Evergreen.
Only volume and range are tested here: it predicts whether the day will be busier, not whether the stock will rise or fall.
