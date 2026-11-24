# Paired differences on D4

Archived Supplementary Table 11 from the EEGJev-Bench v1.3 manuscript.

The table reports paired differences in the D4 group-balanced score, in percentage points. Each difference is EEGJev minus the named comparator. Both models are scored on the rows they share. Within each task, scores are averaged over seeds. Tasks are averaged within sources and sources are weighted equally. Parentheses give 95% intervals from 10,000 paired bootstrap resamples of tasks within sources, conditional on the evaluated participants. Scores and intervals were obtained from the saved model outputs.

The D4 group-balanced score averages refusal recall on questions missing the correct option and answer rate on matched answerable controls. Groups share source, attribute, window length and unordered offered options. Only groups with both kinds of rows enter the task score.

The minuend uses its separately trained Act head. Comparators use their validation-fitted confidence threshold, including the corresponding EEGJev variant in the own-threshold rows. The shared EEGJev comparator in the per-source section uses its Act head. Test and validation are reported separately.

## EEGJev minus

| Comparator | Tasks | Test difference (95% interval) | Validation difference (95% interval) |
| --- | ---: | ---: | ---: |
| EEGJev, own threshold rule | 31 | +4.8 (+4.2, +5.4) | +5.1 (+4.5, +5.8) |
| Ordinary classifier | 31 | +0.8 (+0.3, +1.3) | +1.1 (+0.5, +1.6) |
| Ordinary, per-source | 31 | +0.7 (0.0, +1.5) | +0.3 (-0.4, +1.0) |
| EEGNet | 31 | +6.2 (+5.1, +7.3) | +5.5 (+4.5, +6.4) |
| LaBraM | 24 | 0.0 (-0.8, +0.9) | -0.3 (-1.3, +0.6) |
| NeuroLM | 24 | +7.8 (+6.0, +9.6) | +7.4 (+5.8, +8.9) |
| METIS | 31 | +9.2 (+7.7, +10.8) | +8.6 (+7.2, +10.0) |
| LEAF level 0 | 23 | +8.3 (+6.3, +10.3) | +7.5 (+5.6, +9.4) |
| LEAF level 1 | 23 | +8.4 (+6.4, +10.4) | +7.8 (+6.0, +9.7) |
| LEAF level 2 | 23 | +8.4 (+6.4, +10.4) | +7.8 (+6.1, +9.6) |

## EEGJev per-source minus

| Comparator | Tasks | Test difference (95% interval) | Validation difference (95% interval) |
| --- | ---: | ---: | ---: |
| EEGJev per-source, own threshold rule | 31 | +5.1 (+4.5, +5.8) | +4.7 (+4.1, +5.4) |
| EEGJev (shared encoder) | 31 | +1.3 (+0.9, +1.6) | +1.3 (+0.9, +1.7) |
| Ordinary classifier | 31 | +2.1 (+1.6, +2.5) | +2.4 (+1.8, +2.9) |
| Ordinary, per-source | 31 | +2.0 (+1.4, +2.5) | +1.6 (+1.0, +2.2) |
| EEGNet | 31 | +7.5 (+6.4, +8.6) | +6.8 (+5.7, +7.8) |
| LaBraM | 24 | +1.2 (+0.5, +1.9) | +0.9 (0.0, +1.7) |
| NeuroLM | 24 | +9.0 (+7.3, +10.7) | +8.6 (+7.0, +10.1) |
| METIS | 31 | +10.5 (+9.0, +12.1) | +9.9 (+8.4, +11.4) |
| LEAF level 0 | 23 | +8.3 (+6.4, +10.2) | +7.9 (+6.1, +9.7) |
| LEAF level 1 | 23 | +8.4 (+6.5, +10.3) | +8.2 (+6.5, +10.0) |
| LEAF level 2 | 23 | +8.4 (+6.6, +10.3) | +8.2 (+6.4, +10.0) |
