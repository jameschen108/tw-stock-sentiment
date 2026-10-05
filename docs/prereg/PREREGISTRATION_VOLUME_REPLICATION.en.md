# Preregistration: pre-open PTT discussion volume and same-day trading volume, 2025-2026 replication

> English translation, added 2026-10-05. The registered document is the Chinese original, [PREREGISTRATION_VOLUME_REPLICATION.md](PREREGISTRATION_VOLUME_REPLICATION.md), committed in `7937159` on 2026-09-27 before any test result for 2025 onward was computed; its results section was added in `de0c0a7`. Where the two differ, the Chinese original governs.

Registered: 2026-09-27. This document and `scripts/12_volume_replication.py` were committed **before** computing any test result for 2025 onward;
`--final` first checks that these two files and the three files of the original registration ([PREREGISTRATION_VOLUME.md](PREREGISTRATION_VOLUME.md),
`scripts/11_volume_prereg.py`, `src/pttsent/volume.py`) have no uncommitted changes, and refuses to run otherwise.
The replication is run only once, and the results, good or bad, are recorded as they are in the last section.

## Background

In the 2024 final test of [PREREGISTRATION_VOLUME.en.md](PREREGISTRATION_VOLUME.en.md), the primary hypothesis H1 (log volume pooled across three stocks) passed
with CW t = 3.76; individually, only Evergreen passed after Holm correction. 2024 has only 242 trading days, and that year has already been used for this hypothesis.
All the data from 2025-01 to 2026-09 has now been collected, and the test is run again on this unseen period with **exactly the same settings**.

## Same as the 2024 test

Features, targets, baseline model, model, test, hypotheses and decision rules all follow "Fixed settings"
and "Hypotheses and decision rules" in [PREREGISTRATION_VOLUME.en.md](PREREGISTRATION_VOLUME.en.md). `12_volume_replication.py` rewrites no logic; it loads `build` and `evaluate` directly from `11_volume_prereg.py`.

## Differences

| Item | 2024 test | This test |
|---|---|---|
| Evaluation period | Target dates 2024-01-01 .. 2024-12-31 (242 trading days) | Target dates 2025-01-02 .. 2026-09-24 (420 trading days; the market was closed on 9/25 for the Mid-Autumn Festival) |
| Training data | From 2019-02 up to the start of each block | Same, so it includes 2024 |
| US data | `data/prices/us/` (to 2025-06-30) | `data/prices/us_2026/` (to 2026-09-25), downloaded with the same `fetch_us.py` plus `--out` |

## New data

| Data | Source and method | Checks |
|---|---|---|
| PTT 2025 to 2026/09/26 | `ptt-stock-crawler` (the same crawler that produced pttcc 2020-2024, `-w 4 -d 0.5`), saved as `pttcc/stock_2025.jsonl` and `stock_2026.jsonl`; 2025 and 2026 added to `ptt_years` in `config.yaml` | 45,495 posts, 5.56M comments; no duplicate IDs (3 duplicated pinned posts removed), all dates in range |
| Per-stock texts | `01_prepare_ptt.py` rebuilds `texts_{ticker}.parquet` | Every row before 2024 (uid, time, account, type) is identical to before the rebuild |
| cnyes news 2025-04 .. 2026-09-26 | `collect_cnyes_news.py` from `taiwan-attention-forecast` (the same script that produced the old data) | All 544 days complete, none truncated; after deduplicating by newsId, about 3,000-3,700 items a month, similar to 2024 |
| TAIEX futures 2025-04 .. 2026-09 | TAIFEX futDataDown, one file per month | Re-downloaded 2025-03 first; its sha256 matches the old file exactly, confirming the same method; contract format unchanged |
| Disposition stocks, ex-rights/ex-dividend 2026 | `fetch_range` in `collect_disposition.py` and `collect_exrights.py` from `taiwan-market-attention` | Same file names and format; Q3 fetched only up to 2026-09-27 |
| Price panel 2025-07 .. 2026-09-24 | `fetch_prices.py` + `build_panel.py` | The 4.75M rows up to 2025-06-30 are identical, column by column, to before the rebuild; the comparison with FinMind is unchanged |

After rebuilding the data, rerunning `python scripts/11_volume_prereg.py` (development period) gives exactly the registered reference values.

sha256: `us_2026/TSM.json` `a3c5b9f622e91cf4f48761b31dafcca7f2a2b41b70f7fd3ec2a37caf20056c05`,
`us_2026/SOX.json` `5e489187d52347ee33aae47dd20f4278b9c54593d3c1ebb7855667fabae3d0f2`,
`pttcc/stock_2025.jsonl` `9254575788bdb218fa14f8684e2b19b3016b422fced3396f81301dccb00b89c3`,
`pttcc/stock_2026.jsonl` `6e61993a8dc4c73e8fbed58c736619df2e0b7fea61244cd2f770b74a5ad70b82`.

## What has already been seen in 2025-2026

While preparing the data I looked at: monthly counts of PTT posts and cnyes news, per-stock text counts, row counts for prices and TAIEX futures, counts of disposition and ex-rights/ex-dividend records,
the usable days per stock printed by `12_volume_replication.py` (without `--final`), and the reasons for missing values.
I have **not** looked at the relation between discussion volume and trading volume or range, nor at any trading volume or range values (I only confirmed that the high equals the low on 2025-04-07 and 04-10).

## Differences known in advance

- **Much less Evergreen discussion**: about 380k comments in 2019-2024, only 13k in 2025-01 to 2026-09 (TSMC 320k, Hon Hai 36k).
  With the shipping boom over, Evergreen's pre-open discussion will be noisier, and its individual power will be lower than the table below estimates.
- **Usable days**: for volume, 418 days each for 2330 and 2603 and 395 for 2317; for range, 394, 394 and 371.
  On 2025-04-07 and 04-10, during the tariff shock, TSMC and Hon Hai were locked limit-down / limit-up all day, so under the registered rule their range is missing, and so is the 22-day mean for about 22 days afterwards.
- **Deleted posts**: the new data comes from the same crawler as pttcc, but it was fetched closer to the posting date, so fewer deleted posts may be missing.

## Decision

Same as in 2024: if H1 passes, the replication succeeds; H2, H3 and the opening-gap interpretation are reported separately and do not change the H1 verdict.
If H1 fails, the conclusion becomes "the 2024 result did not recur in 2025-2026", written as is into the README, with a review of differences such as the drop in Evergreen discussion.

## Power

Power at one-sided α = 0.05 if the effect is real (volume on about 395 common days, range on about 371):

| | From the development-period effect | From the 2024 effect |
|---|---|---|
| H1 pooled log volume | Expected t 3.60, 97% | Expected t 4.80, about 100% |
| H2 2330 / 2603 / 2317 (uncorrected) | 46% / 82% / 70% | 69% / 96% / 68% |
| H3 pooled log range | Expected t 2.02, 65% | Expected t 3.61, 98% |

The development-period effect was picked after exploration and is optimistic; the 2024 effect is used only because H1 passed, so it may also be inflated (winner's curse).
Both columns are upper bounds, and the drop in Evergreen discussion will lower the actual power further.

## Change log

No changes.

## Replication results

Run on 2026-09-27, after preregistration commit `7937159`: `python scripts/12_volume_replication.py --final`
(target dates 2025-01-02 .. 2026-09-24; 395 days common to all three for volume).

| | CW t | One-sided p | Out-of-sample MSE change | Effect of 1 SD | Verdict |
|---|---|---|---|---|---|
| **H1** pooled log volume | +2.94 | 0.002 | | | **Passed** |
| H2 2330 log volume | +2.22 | 0.013 | −1.06% | +4.0% | Passed (Holm) |
| H2 2603 log volume | +2.49 | 0.006 | −1.99% | +10.0% | Passed (Holm) |
| H2 2317 log volume | +1.68 | 0.046 | −0.48% | +3.6% | Passed (Holm, threshold 0.05) |
| **H3** pooled log range | +2.47 | 0.007 | | | **Passed** |
| 2330 log range (descriptive) | +1.75 | 0.040 | −0.93% | +4.3% | |
| 2603 log range (descriptive) | +1.56 | 0.059 | −0.41% | +4.7% | |
| 2317 log range (descriptive) | +0.78 | 0.22 | −0.15% | +1.8% | |

For interpretation (not a hypothesis): after also adding the opening gap, the pooled log-volume CW t = +2.78 (one-sided p = 0.003), still significant.

**Conclusion**: the replication succeeded. The incremental predictive power of pre-open PTT discussion volume for same-day trading volume holds in two independent out-of-sample periods, 2024 (CW t = 3.76) and 2025-2026 (CW t = 2.94),
and it stays significant after adding the opening gap. This time each of the three stocks also passes Holm correction (in 2024 only Evergreen did).
The effect is still small: out-of-sample MSE falls 0.5-2%. Although Evergreen discussion is far lower, its increment is still the largest of the three. The range result is weaker and comes mainly from TSMC.
