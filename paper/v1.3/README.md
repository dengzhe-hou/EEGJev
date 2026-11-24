# EEGJev v1.3 supplementary resources

[Result figure tables](result-figure-tables.md) gives the complete test and validation tables for Figures 5 and 6, with coverage, paired intervals and the original captions.

[Data quality audit](data-quality-v13.md) gives the signal rules, window-level checks and amplitude comparisons.

[Reproduction details](reproduction-details.md) gives the full EEGJev training recipe, baseline settings, training hardware and times, and the results on the official CBraMod splits.

[Related benchmarks and candidate scoring](related-benchmarks.md) provides the additional protocol comparison from the manuscript's original Supplementary Note 1.

[Few-shot calibration and paired D2 differences](supplementary-d2-and-few-shot.md) gives the complete aggregate tables and downloadable CSVs.

[Paired D4 comparisons](d4-paired-comparisons.md) gives matched task counts and paired intervals for test and validation.

[Zero-shot motor imagery on unseen datasets](zero-shot-external-motor-imagery.md) gives all six dataset and window-length conditions for the nine model settings.

[Inference rules for known tasks](known-task-inference-rules.md) gives all three seeds on test and validation for the D4 full-label check and D1 and D2 averaging.

| File | Contents |
| --- | --- |
| [Fixed-option scores by source and seed](source-data/fixed-options-by-source.csv) | The original Supplementary Table 17. Each split has 13 sources and one source-mean row, with separate attribute selection and answer balanced accuracy. |
| [Figure 3a plot data](source-data/figure-3a-by-source.csv) | The per-source means across the three training seeds. |
| [Figure 4 matched-task scores](source-data/figure-4-matched-scores.csv) | The plotted score pair and task count for each baseline and question type. |
| [Figure 5 result summaries](source-data/figure-5-scores.json) | D2, D3 and D4 scores and matched comparisons. |
| [Figure 6 temporal result summaries](source-data/figure-6-by-task-and-paired-scores.json) | T1 and T2 task scores and comparisons using fixed combination rules. |
| [Figure 6 matched T2 readouts](source-data/figure-6-matched-readouts.json) | The comparison trained on the same T2 labels, including the two participant-level interval estimates. |

From this directory, reproduce Fig. 3a with Python and Matplotlib.

    python -I -B figures/reproduce_fig3a.py figure3a-output

The script reads the aggregate source table and produces a PDF, an editable SVG and a PNG. The included font faces are covered by the [font licence](figures/fonts/LICENSE).
