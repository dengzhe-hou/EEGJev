# Contributing to EEGJev-Bench

Use the [benchmark correction form](https://github.com/dengzhe-hou/EEGJev/issues/new?template=benchmark-correction.yml) to report a label, source or scoring problem. Use the [new question paradigm form](https://github.com/dengzhe-hou/EEGJev/issues/new?template=new-question-paradigm.yml) to propose a new task. Include a public source, an example question and answer, the split and metric, and a control for a plausible shortcut.

From the repository root, run the example to check the scoring interface.

~~~sh
python -I -B bench/examples/score_linked_demo.py
~~~

It checks T1 and T2 on synthetic questions. See the [benchmark guide](bench/README.md) for reproducing a score on your own inputs. In a result proposal, report the model and version, source and split, question type, task coverage, and scoring command. Use aggregate results and a synthetic example for public discussion.
