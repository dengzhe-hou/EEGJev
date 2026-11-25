# Contributing to EEGJev-Bench

Use the [benchmark correction form](https://github.com/dengzhe-hou/EEGJev/issues/new?template=benchmark-correction.yml) to report a label, source or scoring problem. Use the [new question paradigm form](https://github.com/dengzhe-hou/EEGJev/issues/new?template=new-question-paradigm.yml) to propose a new task. Include a public source, an example question and answer, the split and metric, and a control for a plausible shortcut.

From the repository root, run the example to check the scoring interface.

~~~sh
python -I -B bench/examples/score_linked_demo.py
~~~

It checks T1 and T2 on synthetic questions. See the [benchmark guide](bench/README.md) for reproducing a score on your own inputs.

## Submit a validation result

Build the v1.3 validation inputs under the applicable source terms. Link a versioned public evaluation recipe that identifies the source, split, selected questions and task coverage, with retrieval instructions. A reproducible recipe lets other submissions enter the same table. Link the model version and document its training data and prediction command.

Run this from the repository root with NumPy installed. Provide one prediction per selected question with `question_id` and a zero-based `prediction_index`. Use `null` for a D4 refusal.

~~~sh
python -I -B bench/tools/community_v13.py pack \
  --root /path/to/built-corpus --track T2 --predictions predictions.jsonl \
  --model "Example model v1" --model-url https://example.org/model-v1 \
  --evaluation-url https://example.org/versioned-validation-recipe \
  --out bench/community/submissions/example-model-v1-t2.json
~~~

For a single source, also pass `--source` and use a recipe that describes that scope. The package uses the existing v1.3 scorer with the validation split. The board reports D1/D3 balanced accuracy, D2 balanced joint score, D4 matched refusal score, and T1/T2 linked-view accuracy. The [full scoring tool](bench/tools/benchmark_v13.py) also produces the other metrics.

Open a pull request with the generated aggregate JSON and describe how to reproduce it. Reviewers check the public recipe, coverage, and permissions to share. Keep question rows, predictions, subject IDs, and raw recordings in your local workspace. After merge, the workflow refreshes the [community validation leaderboard](bench/community/LEADERBOARD.md). It groups results by evaluation recipe, track, question count, and scored task IDs. The paper's test tables are available in [the paper archive](paper/v1.3/README.md).

Verify the packaging and rendering with the built-in invented inputs.

~~~sh
python -I -B bench/examples/check_community_submission.py
~~~
