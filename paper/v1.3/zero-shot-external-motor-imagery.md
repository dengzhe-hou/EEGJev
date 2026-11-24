# Zero-shot motor imagery on unseen datasets

Complete results from Supplementary Table 4 of EEGJev v1.3, Overleaf revision 819b592.
Frozen models distinguish left-hand and right-hand motor imagery on Weibo2014 (10 participants) and Dreyer2023A (60 participants). Scores are balanced accuracy in percent, averaged over participants and over the three seeds. Chance is 50%.

Models with bciciv components used those components on both datasets. The EEG was mapped to the 59 bciciv electrodes. Each missing electrode was filled by averaging its two nearest recorded neighbours. LEAF used its own electrode mapping. Windows start at the cue.

| Model | Weibo2014, 1 s | Weibo2014, 2 s | Weibo2014, 4 s | Dreyer2023A, 1 s | Dreyer2023A, 2 s | Dreyer2023A, 4 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EEGJev | 51.0 | 50.2 | 49.9 | 52.4 | 50.6 | 49.6 |
| Ordinary classifier | 53.1 | 55.5 | 56.9 | 58.1 | 57.2 | 55.2 |
| EEGNet (bciciv models) | 52.5 | 57.5 | 57.3 | 53.5 | 53.3 | **55.9** |
| LaBraM (bciciv models) | 50.0 | 50.0 | 49.9 | 50.0 | 50.0 | 50.5 |
| LEAF level 0 | 49.7 | 58.1 | 55.0 | 49.8 | 49.2 | 48.7 |
| LEAF level 1 | 52.4 | 59.0 | 56.9 | 50.3 | 49.2 | 49.2 |
| LEAF level 2 | **54.9** | **63.6** | **60.0** | 50.8 | 49.3 | 49.3 |
| EEGJev, per-source | 51.9 | 52.9 | 53.1 | 52.0 | 51.4 | 52.9 |
| Ordinary, per-source | 52.5 | 56.8 | 57.5 | **59.7** | **57.5** | 55.8 |

[Download the scores as CSV](source-data/zero-shot-external-motor-imagery.csv).

EEGJev scores 49.6% to 52.4%. The ordinary classifier scores 2.2 to 7.0 percentage points higher across the six matched conditions, with intervals that exclude zero.

[Few-shot calibration on the same datasets](https://github.com/dengzhe-hou/EEGJev/blob/4bd482413909d7b764d22f27798f2bfa425956dc/paper/v1.3/supplementary-d2-and-few-shot.md#few-shot-calibration-on-two-unseen-datasets) reports the separate experiment with labelled windows from each person.
