# EEGJev v1.3 reproduction details

These notes accompany version v1.3 of the EEGJev manuscript.

- [Training details of EEGJev](#training-details-of-eegjev)
- [Baselines](#baselines)
- [Training compute and parameter inventory](#training-compute-and-parameter-inventory)
- [Comparison with CBraMod on its official splits](#comparison-with-cbramod-on-its-official-splits)

## Training details of EEGJev

This appendix gives the training settings summarized in the EEGJev subsection of Methods.

### Step 1.

One CBraMod encoder is fine-tuned on the D1 labels of all 19 sources together, with the hidden layers per source and window length and the output layer per cell of the EEGJev subsection of Methods. This step runs for 50 epochs with AdamW, a learning rate of $10^{-4}$ for the encoder and $5 \times 10^{-4}$ for the heads, weight decay 0.05, label smoothing 0.1, batches of 64, cosine decay to $10^{-6}$, gradient clipping at 1.0 and 32-bit precision. The epoch with the highest mean validation balanced accuracy over cells is kept.

### Decision head.

The decision head is trained with soft cross-entropy over the offered candidates while the encoder stays frozen. The D1 and D2 part uses the 3,258,618 D1 and D2 questions of the training persons for 50 epochs of 378 updates with 64 questions per update (18,900 updates). Each draw picks a source with probability proportional to the square root of its question count and a window length in proportion to its question count. Within a source and length, a question is drawn with a weight inverse to the frequency of its answer in its cell, and the questions of one D2 window are never split. Options are shuffled, 25% of the questions offer a random subset of their options, and 10% replace the option texts by letters whose mapping is stated in the question. A D1 question whose source asks another attribute at the same window length, with no label in common, is drawn with the labels of both attributes as options with probability 0.25, up to 12 options, and the subset and letter augmentations are not applied to these draws.

The same schedule holds 10,463 updates on T1 and 1,913 updates on T2, spread evenly over the epochs, so that training has 31,276 updates. A T1 update draws pairs from the 557,933 training pairs and a T2 update event groups from the 64,465 training groups, with the same weighting of sources, lengths and answers, and each draw takes one of the linked views of its pair or group. The optimizer is AdamW with a learning rate of $10^{-4}$, weight decay 0.01, cosine decay to $10^{-6}$ over all updates, gradient clipping at 1.0 and 32-bit precision. The embeddings that mark the two windows of T1 and the blocks of T2 are trained with the head. The epoch is selected on D1 and D2 validation questions, at most 300 per task. Three seeds are trained, and each seed has its own step-1 encoder.

### Act head.

The Act head is trained after the decision head, which stays frozen, on pairs of D4 training rows. The target is to refuse on refusal rows and to answer on control rows. The pool holds at most 2,000 pairs per cell, and the pairs of 15% of the training windows are held out to select the epoch. Training runs for 11 epochs of 788 steps with 64 rows per step, with AdamW at a learning rate of $10^{-4}$, weight decay 0.05, one warmup epoch and cosine decay to a tenth of the peak.

### D3 branch.

The D3 branch continues from the trained head for 6 epochs of 400 steps. Half of the draws in each step are D3 training questions and half D1 and D2 questions, and the first epoch is a warmup. The branch answers the D3 rows only.

## Baselines

This appendix gives the settings of the baselines summarized in the baselines subsection of Methods.

### EEGNet.

EEGNet ([EEGNet paper](https://doi.org/10.1088/1741-2552/aace8c)) is the architecture of the official implementation (arl-eegmodels, revision 4a512e5), ported to PyTorch with $F_1 = 8$, $D = 2$, $F_2 = 16$, a kernel length of 64 and dropout 0.25. One model is trained for each D1 task, attribute and window length, and for each D3 task. Training uses Adam with a learning rate of $10^{-3}$, batches of 64 and 500 epochs, and the epoch with the lowest validation cross-entropy is kept. Over 134 tasks and three seeds this gives 402 models. The 134 tasks are the 109 D1 and 22 D3 tasks and three bciciv tasks that are not scored.

### LaBraM.

LaBraM ([LaBraM paper](https://arxiv.org/abs/2405.18765v1)) starts from the published pretrained weights and is fine-tuned for each task with the official 50-epoch TUAB recipe, an effective batch of 512 (64 with eight accumulation steps), a learning rate of $5 \times 10^{-4}$ with layer decay 0.65, weight decay 0.05, five warmup epochs, label smoothing 0.1 and mixed precision. The epoch with the highest validation accuracy is kept. LaBraM's channel table does not cover the chbmit bipolar derivations, the two sleep channels or the hbn channels, so it has no models for those sources. This leaves 339 models over 113 tasks and three seeds. Its validation and test outputs are computed with deterministic GPU settings, because the training-time settings did not give identical outputs in repeated runs.

### Ordinary classifier.

The ordinary classifier is EEGJev's step-1 model, the shared CBraMod encoder with one hidden stack per source and window length and one linear output layer per cell (the EEGJev subsection of Methods). It does not read the question, so it answers every phrasing of a question the same way. It answers a D2 row with the output layer of the attribute asked, and it refuses in D4 only through the confidence threshold. For D3 it receives bin heads on the frozen step-1 features, trained on the same D3 training questions as the D3 branch of EEGJev.

### NeuroLM.

NeuroLM ([NeuroLM paper](https://arxiv.org/abs/2409.00101v3)) starts from the released NeuroLM-B weights (255.1 million parameters) and is instruction-tuned on the D1 and D2 training questions with the official recipe, namely 5 epochs, a global batch of 512, a peak learning rate of $5 \times 10^{-4}$ and the model of the final epoch. A stage of 300 steps, half on D3 questions, follows for the D3 rows. Its inputs are the benchmark windows with the filter of its official preprocessing. It has no input for hbn and sleep. Three seeds are trained.

### METIS.

We use the released METIS checkpoint ([METIS paper](https://doi.org/10.1002/aisy.70486)) with fixed weights. Following its inference example, we prompt the model with the question and lettered options. It selects the letter with the highest next-token logit in one forward pass. Windows are supplied in volts because the model standardizes each channel internally.

### LEAF.

LEAF ([LEAF paper](https://arxiv.org/abs/2509.24302v2)) is run with its released instruction-tuned checkpoint (v1.0, MPNet text encoder) and fixed weights, with no training on the benchmark. It maps channels to its own electrode table and has no predictions for chbmit, deprest, hbn, mom and sleep. We report its three instruction levels separately. Level 0 uses no instruction, level 1 reads the question, and level 2 additionally mixes in the candidate prototypes of the row. The levels are modes of one checkpoint, not seeds.

### Per-source variants.

To see what sharing the encoder costs, we also train one CBraMod encoder per source in step 1, each on its own source's D1 labels, and build both the ordinary classifier and EEGJev on top of these encoders with otherwise the same recipe. We report them as comparison rows.

## Training compute and parameter inventory

Trainable parameters, hardware and training time of the trained models for data version v1.3. Times are the wall-clock times of the formal runs in the cluster records. A100 runs used one MIG slice (4g or 3g) of an 80 GB card, and RTX PRO 6000 runs one full card. EEGNet and LaBraM give the median time per model and the total over all models.

| Model and stage | Trainable parameters | Hardware | Time per run | Runs |
| --- | --- | --- | --- | --- |
| EEGJev, step 1 (shared CBraMod fine-tuning) | 769.2 M | RTX PRO 6000 | 4.4 to 5.0 h | 3 seeds |
| EEGJev, decision head with T1 and T2 updates | 806.8 M | A100 slice | 2.1 to 2.3 h | 3 seeds |
| EEGJev, D3 branch | 806.8 M | A100 slice | 13 to 14 min | 3 seeds |
| EEGJev, Act head (features, then training) | 67.3 k | CPU, 2 cores | 1.9 to 2.0 h, then under 1 min | 3 seeds |
| Per-source step 1 (19 encoders) | 857.1 M in total | RTX PRO 6000 | 4.4 to 4.5 h | 3 seeds |
| Per-source decision head | 806.8 M | A100 slice | 2.2 to 2.4 h | 3 seeds |
| Per-source D3 branch | 806.8 M | A100 slice | 14 to 16 min | 3 seeds |
| Per-source Act head | 67.3 k | CPU | 1.9 to 2.0 h, then under 1 min | 3 seeds |
| Ordinary classifier, D3 bin heads | 12.7 k | CPU | 1.3 to 2.2 h | 3 + 3 seeds |
| EEGNet, one model per task | 1.5 to 4.6 k | A100 slice | 17.6 min (216.8 h in total) | 402 |
| LaBraM, one model per task | 5.8 M | A100 slice | 5.9 min (55.9 h in total) | 339 |
| NeuroLM-B, instruction tuning | 169.6 M (255.1 M with its frozen tokenizer) | A100 slice | 22.8 to 23.3 h | 3 seeds |
| METIS, LEAF | none (released checkpoints) |  |  |  |

## Comparison with CBraMod on its official splits

EEGJev-Bench has its own splits and preprocessing, so the numbers of the main tables cannot be set next to published ones. We therefore ran EEGJev and CBraMod ([CBraMod paper](https://openreview.net/forum?id=NPNUHgHF2w)) on the same splits of two datasets, as generated by the released CBraMod preprocessing ([the comparison table](#cbramod-comparison-table)). PhysioNet-MI ([EEG Motor Movement/Imagery Dataset v1.0.0](https://doi.org/10.13026/C28G6P), [BCI2000](https://doi.org/10.1109/TBME.2004.827072), [PhysioNet](https://doi.org/10.1161/01.cir.101.23.e215)) has four imagined movements in windows of 4 s from 109 persons, with persons 1 to 70 for training, 71 to 89 for validation and 90 to 109 for test. Mumtaz2016 ([Mumtaz2016 v2](https://doi.org/10.6084/m9.figshare.4244171.v2), [Mumtaz et al. 2017](https://doi.org/10.1371/journal.pone.0171409)) separates persons with major depressive disorder from healthy controls in windows of 5 s, with 43 training, 9 validation and 11 test persons. Appendix E.4 of the [CBraMod paper](https://arxiv.org/html/2412.07236v6#A5.SS4) reports 10 Mumtaz2016 test participants. CBraMod was run from its official code with the defaults of the repository. EEGJev was trained with the two steps of [Training details of EEGJev](#training-details-of-eegjev) on the training split of each dataset alone, as one single-attribute task whose candidates are the class names. Each dataset has one attribute and single windows, so the union candidates and the two-window and long-window updates of the main model have nothing to act on. Each model used its own validation selection rule and was scored once on the resulting test split, with seeds 0, 1 and 2. The runs were made on 14 November 2026. The PhysioNet-MI page states no ethics approval, and the publication that describes the Mumtaz2016 data reports the approval of its ethics committee and the written consent of the participants ([Mumtaz et al. 2017](https://doi.org/10.1371/journal.pone.0171409)).

### CBraMod comparison table

Test results on the official CBraMod splits (%, Cohen's $\kappa$ as a fraction), three seeds, with the sample standard deviation of the balanced accuracy. The second and third metrics are named in the header row of each dataset.

#### PhysioNet-MI, four classes

| Model | Balanced accuracy by seed | Mean ± SD | Cohen’s $\kappa$ | Weighted F1 |
| --- | --- | --- | --- | --- |
| CBraMod, official code | 62.1 / 60.7 / 62.2 | 61.7 $\pm$ 0.9 | 0.489 | 61.7 |
| EEGJev | 58.8 / 60.0 / 59.1 | 59.3 $\pm$ 0.6 | 0.457 | 59.4 |
| EEGJev, inputs divided by 100 | 58.1 / 59.4 / 60.4 | 59.3 $\pm$ 1.2 | 0.457 | 59.4 |

#### Mumtaz2016, two classes

| Model | Balanced accuracy by seed | Mean ± SD | AUC-PR | AUROC |
| --- | --- | --- | --- | --- |
| CBraMod, official code | 90.5 / 88.1 / 89.5 | 89.3 $\pm$ 1.2 | 97.2 | 96.9 |
| CBraMod, learning rate of its paper | 87.5 / 88.2 / 89.2 | 88.3 $\pm$ 0.8 | 97.3 | 96.8 |
| EEGJev | 93.1 / 93.8 / 93.8 | 93.6 $\pm$ 0.4 | 99.0 | 98.6 |

On PhysioNet-MI, CBraMod reaches 61.7% balanced accuracy and EEGJev 59.3%, so EEGJev is 2.4 points lower. Giving EEGJev the input scaling of CBraMod instead of its own standardization per window leaves its test result at 59.3%. On Mumtaz2016, EEGJev reaches 93.6% and CBraMod 89.3%. All runs classify at least 85% of the windows correctly for 10 of the 11 test persons. The difference of 4.2 points comes from one person with depression, for whom EEGJev classifies 37 to 47% of the windows correctly and CBraMod 0 to 12%. Our CBraMod runs are below the values of its paper on both datasets, which reports 64.2% on PhysioNet-MI and 95.6% on Mumtaz2016 over five seeds. With the single learning rate given in that paper, CBraMod reaches 88.3% on Mumtaz2016. We did not find the cause of this gap, so [the comparison table](#cbramod-comparison-table) compares the two models under one protocol and does not compare EEGJev with the published numbers.
