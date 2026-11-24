# EEGJev v1.3 supplementary result tables

These tables contain the aggregate results reported as Supplementary Tables 6 and 10 in manuscript revision `b0248b4`.

## Few-shot calibration on two unseen datasets

These are the complete aggregate results for the left-versus-right-hand motor imagery task on two new external datasets, Weibo2014 (10 participants) and Dreyer2023A (60 participants). Windows begin at the cue and last 1, 2 or 4 s. For each person and window length, a logistic regression is fitted to frozen model features using k labelled windows per class, then evaluated on that person's other windows. The results are balanced accuracy in percent, averaged across participants, 20 draws of the labelled windows and three training seeds. The rows with k = 0 are the results without calibration.

[Download the few-shot calibration CSV](source-data/few-shot-external-calibration.csv).

| Model and labelled windows per class | Weibo2014, 1 s | Weibo2014, 2 s | Weibo2014, 4 s | Dreyer2023A, 1 s | Dreyer2023A, 2 s | Dreyer2023A, 4 s |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| EEGJev, k = 0 | 51.0 | 50.2 | 49.9 | 52.4 | 50.6 | 49.6 |
| EEGJev, k = 5 | 50.7 | 50.4 | 52.2 | 54.9 | 52.4 | 52.3 |
| EEGJev, k = 10 | 51.1 | 51.3 | 53.1 | 57.5 | 54.3 | 53.5 |
| EEGJev, k = 20 | 51.9 | 52.9 | 54.2 | 60.8 | 57.0 | 55.4 |
| Ordinary classifier, k = 0 | 53.1 | 55.5 | 56.9 | 58.1 | 57.2 | 55.2 |
| Ordinary classifier, k = 5 | 52.2 | 55.2 | 56.4 | 60.4 | 58.5 | 55.7 |
| Ordinary classifier, k = 10 | 53.5 | 57.8 | 58.4 | 65.1 | 62.1 | 58.1 |
| Ordinary classifier, k = 20 | 55.4 | 60.7 | 60.6 | 69.9 | 66.3 | 61.2 |
| EEGNet (bciciv models), k = 0 | 52.5 | 57.5 | 57.3 | 53.5 | 53.3 | 55.9 |
| EEGNet (bciciv models), k = 5 | 51.9 | 53.9 | 57.7 | 56.1 | 55.1 | 54.3 |
| EEGNet (bciciv models), k = 10 | 53.0 | 55.2 | 60.0 | 58.6 | 57.2 | 56.6 |
| EEGNet (bciciv models), k = 20 | 54.3 | 57.4 | 62.5 | 61.4 | 60.0 | 59.1 |
| LaBraM (bciciv models), k = 0 | 50.0 | 50.0 | 49.9 | 50.0 | 50.0 | 50.5 |
| LaBraM (bciciv models), k = 5 | 50.6 | 50.0 | 50.5 | 53.4 | 51.4 | 50.4 |
| LaBraM (bciciv models), k = 10 | 51.1 | 49.9 | 50.9 | 55.4 | 52.0 | 50.9 |
| LaBraM (bciciv models), k = 20 | 51.3 | 50.6 | 51.4 | 57.8 | 52.9 | 51.3 |

## Paired differences on D2

D2 asks two questions about the same EEG window. These paired differences use per-question balanced accuracy from saved outputs on common rows. Means give equal weight to sources over the tasks covered by each model. Test and validation are reported separately. Values are in percentage points and are shown as estimate (95% interval lower bound, upper bound). The intervals come from bootstrapping tasks within sources. A blank cell denotes a comparison of a reference model with itself. The averaged model combines the same-seed EEGJev and ordinary classifier log-softmax scores over the offered options, with weight 0.5 each.

[Download the paired D2 differences CSV](source-data/d2-paired-differences.csv).

| Model | Tasks | Minus EEGNet, test | Minus EEGNet, validation | Minus EEGJev, test | Minus EEGJev, validation |
| --- | ---: | :---: | :---: | :---: | :---: |
| EEGNet, one model per task | 59 |  |  | +2.0 (+0.2,+3.7) | +2.4 (+0.9,+4.0) |
| EEGJev | 59 | -2.0 (-3.7,-0.2) | -2.4 (-4.0,-0.9) |  |  |
| Ordinary classifier | 59 | -0.8 (-2.2,+0.6) | -1.0 (-2.2,+0.2) | +1.1 (-0.2,+2.5) | +1.5 (+0.6,+2.4) |
| EEGJev and ordinary, averaged | 59 | +0.2 (-1.3,+1.6) | -0.2 (-1.4,+1.0) | +2.1 (+1.4,+2.9) | +2.2 (+1.6,+2.9) |
| LaBraM, one model per task | 59 | -1.3 (-2.5,-0.2) | -2.4 (-3.1,-1.7) | +0.6 (-0.7,+2.0) | 0.0 (-1.5,+1.5) |
| NeuroLM, instruction-tuned | 59 | -6.4 (-8.1,-4.7) | -7.8 (-9.7,-5.8) | -4.4 (-6.0,-2.9) | -5.3 (-6.6,-4.1) |
| METIS, fixed weights | 59 | -18.6 (-21.6,-15.8) | -20.6 (-22.9,-18.5) | -16.7 (-19.2,-14.2) | -18.2 (-20.6,-15.8) |
| LEAF, level 0 | 51 | -14.8 (-17.8,-12.0) | -17.5 (-20.0,-15.0) | -12.5 (-15.0,-10.0) | -14.5 (-17.4,-11.7) |
| LEAF, level 1 | 51 | -14.8 (-17.8,-11.9) | -17.0 (-19.7,-14.2) | -12.5 (-15.2,-10.0) | -13.9 (-16.6,-11.4) |
| LEAF, level 2 | 51 | -15.2 (-18.4,-12.2) | -16.8 (-19.6,-14.0) | -12.9 (-15.5,-10.2) | -13.8 (-16.4,-11.3) |
| EEGJev, per-source encoders | 59 | -1.8 (-3.0,-0.7) | -2.6 (-3.7,-1.5) | +0.1 (-1.2,+1.4) | -0.2 (-0.8,+0.5) |
| Ordinary, per-source encoders | 59 | -0.3 (-1.5,+0.9) | +0.1 (-1.6,+1.8) | +1.7 (+0.2,+3.1) | +2.5 (+1.7,+3.3) |
