# EEGJev v1.3 figure result tables

The tables below preserve the manuscript scores, blank cells and comparison intervals. The original captions describe the PDF presentation.

- [Paired questions and refusal](#paired-questions-and-refusal)
- [Ordered bins](#ordered-bins)
- [Two-window and long-window questions](#two-window-and-long-window-questions)

## Paired questions and refusal

**Supplementary Table 12 caption**

Paired, binned and refusal questions. **a**, D2 source-balanced mean balanced accuracy over 59 tasks, the D2 joint score, the D4 group-balanced criterion over 31 tasks and the D1 expected calibration error after temperature scaling, all in %, seed means, each model over the tasks it covers. Chance is 50% for the D4 criterion. The D4 thresholds of LEAF levels 0 and 1 are fitted on uncalibrated probabilities, because their calibrated probabilities equal a uniform guess. **b**, D3 source-balanced mean balanced accuracy (%) and mean absolute bin distance (Dist., lower is better), seed means, each model over the tasks it covers. Differences to EEGNet are paired over the Tasks column on common rows. LaBraM, NeuroLM and LEAF cover 14 of the 22 tasks. In **b**, shading shows BA on all 22 tasks, with the highest BA in each split in bold.

### Supplementary Table 12a

| Model | D2 BA test | D2 BA val | D2 joint test | D2 joint val | D4 group-balanced test | D4 group-balanced val | D1 ECE test | D1 ECE val |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EEGJev, Act head | 61.1 | 62.9 | 39.3 | 41.7 | 59.2 | 58.6 | 2.0 | 2.4 |
| EEGJev, threshold |  |  |  |  | 54.4 | 53.4 |  |  |
| Ordinary classifier, threshold | 62.2 | 64.3 | 41.0 | 44.4 | 58.4 | 57.5 | 1.7 | 1.6 |
| EEGJev, full-label check |  |  |  |  | 65.4 | 65.1 |  |  |
| Ordinary classifier, full-label check |  |  |  |  | 65.5 | 65.3 |  |  |
| EEGJev and ordinary, averaged | 63.2 | 65.1 | 42.1 | 45.4 |  |  |  |  |
| EEGNet, threshold | 63.0 | 65.3 | 44.2 | 46.2 | 53.0 | 53.1 | 3.3 | 1.3 |
| LaBraM, threshold | 61.7 | 62.9 | 41.2 | 43.2 | 57.9 | 57.9 | 9.7 | 11.3 |
| NeuroLM, threshold | 56.6 | 57.5 | 34.0 | 35.3 | 50.1 | 50.2 | 3.2 | 3.0 |
| METIS, threshold | 44.4 | 44.7 | 20.3 | 20.0 | 50.0 | 50.0 | 3.1 | 2.0 |
| LEAF level 0, threshold | 43.9 | 43.4 | 20.0 | 18.8 | 50.1 | 50.4 | 3.6 | 2.6 |
| LEAF level 1, threshold | 43.9 | 44.0 | 20.6 | 19.4 | 50.0 | 50.0 | 2.9 | 2.5 |
| LEAF level 2, threshold | 43.5 | 44.1 | 20.1 | 19.6 | 50.0 | 50.0 | 3.1 | 2.4 |
| EEGJev per-source, Act head | 61.2 | 62.7 | 39.6 | 41.9 | 60.5 | 59.9 | 2.4 | 2.7 |
| EEGJev per-source, threshold |  |  |  |  | 55.3 | 55.1 |  |  |
| Ordinary per-source, threshold | 62.7 | 65.4 | 41.2 | 44.9 | 58.5 | 58.3 | 2.9 | 1.7 |

## Ordered bins

### Supplementary Table 12b

| Model | Tasks | Test BA | Test difference to EEGNet | Test distance | Validation BA | Validation distance |
| --- | --- | --- | --- | --- | --- | --- |
| EEGNet, one model per task | 22 | **42.5** |  | 0.75 | **42.7** | 0.75 |
| EEGJev, D3 branch | 22 | 41.5 | $-1.0$ ($-2.8$, $+0.8$) | 0.80 | 41.3 | 0.82 |
| Ordinary classifier, bin heads | 22 | 42.4 | $0.0$ ($-1.9$, $+1.7$) | 0.83 | 39.5 | 0.85 |
| LaBraM, one model per task | 14 | 39.7 | $+2.8$ ($-1.8$, $+7.6$) | 0.79 | 40.1 | 0.77 |
| NeuroLM, instruction-tuned | 14 | 34.5 | $-2.4$ ($-6.0$, $+0.7$) | 1.00 | 32.4 | 0.96 |
| METIS, fixed weights | 22 | 28.0 | $-14.4$ ($-16.2$, $-12.8$) | 1.27 | 28.8 | 1.28 |
| LEAF, level 0 | 14 | 29.8 | $-7.0$ ($-8.6$, $-5.4$) | 1.23 | 27.2 | 1.37 |
| LEAF, level 1 | 14 | 30.8 | $-6.0$ ($-7.9$, $-3.9$) | 1.15 | 27.3 | 1.30 |
| LEAF, level 2 | 14 | 30.7 | $-6.2$ ($-8.1$, $-4.0$) | 1.16 | 26.9 | 1.31 |
| EEGJev per-source, D3 branch | 22 | 41.6 | $-0.8$ ($-2.3$, $+0.6$) | 0.82 | 39.6 | 0.84 |
| Ordinary per-source, bin heads | 22 | 42.2 | $-0.3$ ($-2.5$, $+1.6$) | 0.83 | 39.0 | 0.87 |

## Two-window and long-window questions

**Supplementary Table 13 caption**

T1 and T2 source-balanced accuracy (%), seed means, each model over the tasks it covers. Chance is 50%. Native rows are answered by EEGJev's decision head. Composed rows are built without training from a model's D1 outputs. Differences to the native EEGJev answers are paired over the Tasks column on the test split. The per-source EEGJev has no composed T1 row. Shading shows accuracy on the full task set for each question type. Bold marks the highest of these scores in each column.

| Model | T1 test | T1 tasks | T1 test difference to native EEGJev | T1 val | T2 test | T2 tasks | T2 test difference to native EEGJev | T2 val |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EEGJev, native | 77.7 | 93 |  | 77.7 | **69.8** | 17 |  | **67.1** |
| EEGJev, composed from its D1 answers | 78.1 | 93 | $+0.4$ ($0.0$, $+0.8$) | 78.2 | 61.8 | 15 | $-6.5$ ($-9.2$, $-3.7$) | 60.2 |
| Ordinary classifier, composed | 78.9 | 93 | $+1.1$ ($+0.3$, $+2.0$) | 78.7 | 61.2 | 15 | $-7.1$ ($-9.1$, $-5.2$) | 60.9 |
| EEGNet, composed | **80.8** | 93 | $+3.1$ ($+1.8$, $+4.4$) | **81.9** | 61.8 | 15 | $-6.5$ ($-9.4$, $-3.8$) | 62.2 |
| LaBraM, composed | 77.5 | 80 | $-1.1$ ($-2.8$, $+0.6$) | 77.8 | 61.4 | 7 | $-13.4$ ($-16.8$, $-10.0$) | 61.2 |
| NeuroLM, composed | 70.1 | 83 | $-7.1$ ($-8.5$, $-5.7$) | 71.4 | 60.2 | 10 | $-10.1$ ($-13.0$, $-7.3$) | 54.8 |
| METIS, composed | 48.9 | 93 | $-28.9$ ($-31.0$, $-26.7$) | 48.3 | 53.5 | 15 | $-14.8$ ($-17.4$, $-12.2$) | 53.9 |
| LEAF level 0, composed | 51.1 | 74 | $-24.9$ ($-27.8$, $-21.8$) | 52.1 | 49.3 | 7 | $-25.5$ ($-27.5$, $-23.5$) | 55.3 |
| LEAF level 1, composed | 50.6 | 74 | $-25.4$ ($-28.5$, $-22.2$) | 51.7 | 50.1 | 7 | $-24.7$ ($-27.3$, $-22.2$) | 54.7 |
| LEAF level 2, composed | 51.2 | 74 | $-24.7$ ($-27.7$, $-21.8$) | 51.6 | 51.4 | 7 | $-23.4$ ($-25.6$, $-21.2$) | 55.8 |
| EEGJev per-source, native | 78.0 | 93 | $+0.2$ ($-0.5$, $+1.0$) | 78.7 | 67.5 | 17 | $-2.3$ ($-3.1$, $-1.5$) | 67.0 |
| EEGJev per-source, composed from its D1 answers |  |  |  |  | 60.9 | 15 | $-7.4$ ($-10.0$, $-4.8$) | 59.5 |
| Ordinary per-source, composed | 79.3 | 93 | $+1.6$ ($+0.6$, $+2.5$) | 80.0 | 63.4 | 15 | $-4.9$ ($-7.7$, $-2.3$) | 61.0 |

### Comparison with trained T2 readouts

The additional comparison with trained T2 readouts covered 3,307 test events, each placed once in each half of a window. On the 13 state-transition tasks, retaining the order of one-second probabilities improved the trained readouts by 1.86 percentage points for EEGNet and 0.99 points for the shared ordinary classifier. On the two tasks with a target in a stimulus stream, the gains were 14.29 and 18.51 points, respectively.
