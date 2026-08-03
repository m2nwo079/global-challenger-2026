# Exposure decay, attribution windows, and incrementality in display ads

Three measurements on publicly released Criteo traffic logs:

1. How click-through rate changes as the same ad is shown repeatedly
2. How much campaign performance rankings shift when the attribution rule
   or window changes
3. How many conversions the advertising actually caused

## Motivation

An interview with a practitioner in the advertising division of a major
Korean messaging platform surfaced several operational judgments that were
described qualitatively but never quantified — where user fatigue starts to
conflict with revenue, and how attribution windows are chosen. This project
measures those quantities on public data instead.

The point is not to transfer the numbers to that platform. Criteo operates a
web display retargeting network; the interview concerned an in-app messenger
placement. See Limitations.

## Modules

| Module | Question | Dataset |
| --- | --- | --- |
| A | Click-through decay across impression order | Criteo Attribution |
| B | Sensitivity of performance to attribution rules and windows | Criteo Attribution |
| C | Incremental effect of advertising | Criteo Uplift |

## Verified baselines

Every figure below was counted twice: once through SQLite after loading, and
once by streaming the original gzipped file directly.

| Metric | Value |
| --- | --- |
| Impressions | 16,468,027 |
| Users | 6,142,256 |
| Campaigns | 675 |
| Clicks | 5,947,563 (CTR 36.12%) |
| Distinct conversions | 435,810 |
| Distinct attributed conversions | 236,331 |
| Attributed share | 54.2% |
| Impressions per user | 2.68 |
| Max timestamp | 2,671,199 (30.9 days) |

Impression count, campaign count, and impressions-per-user match the
published figures. The dataset README states 45K conversions, which does not
match the distributed file under any counting definition tested; 435,810 is
used and the discrepancy is noted rather than reconciled.

## Aggregation rules

See `AGGREGATION_RULES.md`. Definitions were fixed before analysis and
cross-checked against `Experiments.ipynb`, the reproduction notebook shipped
with the dataset.

## Layout

    notebooks/     analysis, one notebook per module
    output/        aggregated results as JSON
    figures/       plots
    data/          not tracked — see Data

## Data

Criteo Attribution Modeling for Bidding Dataset and Criteo Uplift Modeling
Dataset, released by Criteo AI Lab under CC BY-NC-SA 4.0. Neither dataset is
redistributed here; `notebooks/00_setup.ipynb` downloads and loads them.

Diemert Eustache, Meynet Julien, Galland Pierre, Lefortier Damien.
"Attribution Modeling Increases Efficiency of Bidding in Display
Advertising." AdKDD & TargetAd Workshop, KDD 2017.

Aggregated outputs in `output/` are derived from this data and are shared
under the same CC BY-NC-SA 4.0 terms.

## Limitations

- Criteo runs a web display retargeting network. The interview concerned an
  in-app messenger placement. Traffic is global and the logs are from 2017.
  The measured decay coefficient is not presented as an optimal value for
  any specific platform.
- Measured CTR is 36.12%, far above typical display rates. The dataset
  README states the data was subsampled. Absolute click rates are not
  interpreted; only relative change across impression order is used. The
  same sampling may bias the decay estimate.
- `cost` and `cpo` are transformed values, not real prices. Only relative
  comparison is valid.
- The attribution window cannot exceed 30 days, which is the observation
  horizon built into the `conversion` field.
