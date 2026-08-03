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
