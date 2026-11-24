"""Inspect the frozen v1.3 corpus and score its two new question types."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eeg_jev.benchmark.v13_dataset import BenchmarkV13,score_linked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["inspect","score-new"])
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--track",choices=["T1","T2"])
    parser.add_argument("--split",choices=["train","val","test"],default="val")
    parser.add_argument("--source")
    parser.add_argument("--predictions",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--test-model-scoring-authorized",action="store_true")
    args=parser.parse_args()
    data=BenchmarkV13(args.root)
    if args.command=="inspect":
        print(json.dumps(dict(frozen=data.frozen,tracks=data.tasks,
                              test_scope=data.manifest["test_authorization"]),ensure_ascii=False))
        return
    data.require_frozen()
    if not args.track or not args.predictions or not args.out:
        parser.error("score-new requires --track, --predictions and --out")
    if args.split=="test" and not args.test_model_scoring_authorized:
        raise ValueError("Construction-only test approval does not authorize model scoring")
    predictions={}
    for line in args.predictions.open():
        row=json.loads(line)
        if row["question_id"] in predictions:raise ValueError("Duplicate prediction question_id")
        predictions[row["question_id"]]=row["prediction_index"]
    report=score_linked(data.new_questions(args.track,args.split,args.source),predictions)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":main()
