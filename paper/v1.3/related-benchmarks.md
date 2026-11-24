# Related benchmarks and candidate scoring

[PhysioBench](https://arxiv.org/abs/2609.20836) evaluates seizure detection and seizure state on 30-s EEG windows. On EEG windows
carrying both questions, the seizure-state answer determines whether a seizure is present. Windows with less than
20% seizure overlap omit the detection question, and several states share the same detection answer.
PhysioBench and EEGJev-Bench both hold out participants and phrasings for evaluation. Their window durations,
seizure label definitions and participant splits differ. For example, PhysioBench excludes the five minutes before
seizure onset from its seizure-state task, and our CHB-MIT pre-seizure windows lie within 32 s of onset.

[BrainBench](https://arxiv.org/abs/2608.04156) grades EEG analysis reports and includes cases with missing inputs or false premises.
EEGJev-Bench pairs each refusal item with an answerable item using the same EEG window, question and option count.
[UniEQA](https://huggingface.co/datasets/DoYangTan/EQA_DATASET) supplies source segments for our benchmark, and [EEG-Captioner](https://arxiv.org/abs/2610.30995) poses
multiple questions about individual segments. We check each D2 attribute pair on training segments so that neither
answer determines the other. We also score the distance between ordered answers for D3.

EEGJev uses the pretrained [CBraMod encoder](https://openreview.net/forum?id=NPNUHgHF2w). Its decision head follows Laya's design for scoring
offered answers together at their marker [positions](https://github.com/NandhaKishorM/laya/tree/1e28ac20c0896b1c37a744cd11f740eb98f8b178). A separate Act head predicts whether to answer.
We train both heads for EEG question answering.
