# Inference rules for known tasks

This is the complete per-seed content of Supplementary Table 7 from the EEGJev v1.3 manuscript at Overleaf revision b942f5b. Values are source-balanced means in percent.

The D4 full-label check uses the group-balanced criterion over 31 tasks. D1 and D2 accuracy are mean balanced accuracy over 109 and 59 tasks, respectively. The last row shows the D2 joint score.

| Rule and model | Test, seeds 0 / 1 / 2 | Validation, seeds 0 / 1 / 2 |
| --- | ---: | ---: |
| D4 full-label check, EEGNet | **66.4 / 66.8 / 66.8** | **66.6 / 66.3 / 66.9** |
| D4 full-label check, EEGJev | 65.2 / 65.7 / 65.3 | 65.0 / 65.2 / 65.0 |
| D4 full-label check, ordinary classifier | 65.8 / 65.6 / 65.2 | 65.1 / 65.4 / 65.4 |
| D1 averaging, EEGJev and ordinary | 65.6 / 65.2 / 65.0 | 65.7 / 65.7 / 66.4 |
| D2 averaging, EEGJev and ordinary | 63.5 / 63.4 / 62.7 | 64.6 / 65.3 / 65.3 |
| D2 joint score of the averaged answers | 42.4 / 42.5 / 41.5 | 45.0 / 45.5 / 45.8 |

[Download the per-seed scores as CSV](source-data/known-task-inference-rules.csv).

In the matched D4 test comparison, the means are 65.4% for EEGJev, 66.7% for EEGNet and 65.5% for the ordinary classifier. The paired EEGJev-minus-EEGNet difference is −1.3 percentage points (95% interval, −2.0 to −0.7, with tasks resampled within sources). EEGJev's Act head scores 59.2%.

Averaging EEGJev and the ordinary classifier gives 65.2% on test D1 and 63.2% on test D2. Relative to EEGJev, the paired gains are 1.4 points (95% interval, 1.0 to 1.8) on D1 and 2.1 points (1.4 to 2.9) on D2. Relative to EEGNet, the differences are 0.7 points (−0.5 to 1.9) and 0.2 points (−1.3 to 1.6), respectively. The averaged D2 answers attain a test joint score of 42.1%.
