# Aggregation Rules

Definitions fixed before analysis, cross-checked against `Experiments.ipynb`
(the reproduction notebook shipped with the Criteo Attribution dataset).
Every query in this repository follows these rules.

## 1. Impression order

Impression order `k` is counted per `(uid, campaign)`, not per `uid`.
Counting per `uid` mixes impressions from different campaigns into one
sequence, which is not what "repeated exposure to the same ad" means.

## 2. Time to conversion

The gap is `conversion_timestamp - timestamp`, applied only to rows where
`conversion = 1`. There is no separate click-timestamp column in this
dataset; the impression timestamp is used. This matches the authors' code:

    df.loc[df.conversion == 1, 'gap_click_sale'] = \
        df.conversion_timestamp - df.timestamp

Verified: zero rows with `conversion = 1` have
`conversion_timestamp < timestamp`.

## 3. Conversion counting

Conversions are deduplicated by `conversion_id`. A single conversion spans
multiple impression rows, so counting rows inflates the total. Note that
`conversion_id` is `-1` when no conversion occurred, so a bare
`COUNT(DISTINCT conversion_id)` counts `-1` as one conversion.

## 4. CTR denominator

The denominator is the number of impressions, not the number of users.

## 5. The `-1` sentinel

`-1` marks a missing value in `conversion_timestamp`, `conversion_id`,
`click_pos`, `click_nb`, and `time_since_last_click`. Always filter with
`>= 0` before computing means, minima, maxima, or time differences. The
authors apply the same mask:

    previous_tslc_mask = (df.time_since_last_click >= 0)

## 6. Day derivation

`day = floor(timestamp / 86400)`. Timestamps are second offsets from the
first impression, matching the authors' definition.

## Attribution rules (module B)

Taken verbatim from the authors' notebook rather than defined ad hoc:

    last_click  = attribution * (click_pos == click_nb - 1)
    first_click = attribution * (click_pos == 0)
    all_clicks  = attribution
    uniform     = attribution / click_nb

All four are conditioned on `attribution = 1`, so the population for
module B is 236,331 attributed conversions, not the 435,810 total
conversions.

## Measured baselines

| Metric | Value |
| --- | --- |
| Impressions | 16,468,027 |
| Users | 6,142,256 |
| Campaigns | 675 |
| Clicks | 5,947,563 (CTR 36.12%) |
| Distinct conversions (`conversion = 1`) | 435,810 |
| Distinct attributed conversions | 236,331 |
| Attributed share | 54.2% |
| Max timestamp | 2,671,199 (30.9 days) |

The dataset README states 45K conversions, which does not match the
distributed file under any counting definition tested. Impression count,
campaign count, and impressions-per-user (2.68) all match published
figures exactly, so the file is treated as intact and 435,810 is used.

CTR of 36.12% reflects the subsampling described in the dataset README.
Absolute click rates are not interpreted; only relative change across
impression order is used.
