"""EEGJev-Bench v1.3 final-build reader, with no model dependency."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np


def pair_questions(pair):
    """Linked A/B swaps and target-label questions for one base EEG pair."""
    for swap in (0,1):
        inputs=pair["windows"][::-1] if swap else pair["windows"]
        labels=pair["labels"][::-1] if swap else pair["labels"]
        for i,target in enumerate(pair["labels"]):
            yield dict(question_id=f'{pair["pair_id"]}/swap{swap}/{i}',
                group_id=pair["pair_id"],source=pair["source"],task=pair["task"],
                subject=pair["subject"],split=pair["split"],track="T1",
                inputs=inputs,options=["window A","window B"],
                question=f'For {pair["attribute"].replace("_"," ")}, which EEG window shows "{target}"?',
                answer_index=labels.index(target))


def score_linked(rows,predictions):
    """User-approved group -> person -> task -> source equal-weight means."""
    groups=defaultdict(list)
    for r in rows:
        if r["question_id"] not in predictions:
            raise ValueError("Missing prediction "+r["question_id"])
        p=predictions[r["question_id"]]
        if p not in (0,1):raise ValueError("Prediction must select option0 or1")
        k=r["source"],r["task"],r["subject"],r["group_id"]
        groups[k].append(float(p==r["answer_index"]))
    subjects=defaultdict(list)
    for (source,task,subject,group),v in groups.items():
        subjects[source,task,subject].append(sum(v)/len(v))
    tasks=defaultdict(list)
    for (source,task,subject),v in subjects.items():
        tasks[source,task].append(sum(v)/len(v))
    sources=defaultdict(list);bytask={}
    for (source,task),v in tasks.items():
        bytask[task]=sum(v)/len(v)
        sources[source].append(bytask[task])
    means={s:sum(v)/len(v) for s,v in sources.items()}
    return dict(source_balanced_mean=sum(means.values())/len(means) if means else None,
                by_source=means,by_task=bytask,distinct_event_or_pair_groups=len(groups),
                aggregation="linked views, then persons, tasks, sources")


class BenchmarkV13:
    """Reader of a built corpus. Publication and model-test permission are separate."""
    def __init__(self,root):
        self.root=Path(root)
        self.manifest=json.loads((self.root/"MANIFEST.json").read_text())
        self.tasks=json.loads((self.root/"metadata/tasks_v13.json").read_text())["tracks"]
        self._windows={}
        self._arrays={}

    @property
    def frozen(self):
        return (self.root/"FROZEN.json").exists()

    def require_frozen(self):
        if not self.frozen:raise ValueError("Dataset construction is not yet frozen")

    def windows(self,source,split):
        if split not in ("train","val","test"):raise ValueError(split)
        key=source,split
        if key not in self._windows:
            part="test" if split=="test" else "train"
            file=self.root/f"frozen_v13/{part}/{source}_windows.jsonl"
            rows={}
            if file.exists():
                for line in file.open():
                    r=json.loads(line)
                    if r["split"]==split:rows[r["window_id"]]=r
            self._windows[key]=rows
        return self._windows[key]

    def _read_ref(self,ref):
        file=self.root/ref["file"]
        if ref["format"]=="npy":
            if file not in self._arrays:
                self._arrays[file]=np.load(file,mmap_mode="r")
            return np.array(self._arrays[file][ref["row"]],copy=True)
        if ref["format"]=="f32":
            shape=tuple(ref["shape"])
            size=int(np.prod(shape))*4
            a=np.memmap(file,dtype=np.float32,mode="r",
                        offset=ref["row"]*size,shape=shape)
            return np.array(a,copy=True)
        raise ValueError("Unsupported stored EEG format "+ref["format"])

    def core_eeg(self,source,split,window_id):
        return self._read_ref(self.windows(source,split)[window_id]["cache_ref"])

    def core_questions(self,split,source=None,track=None,eligible_only=True):
        part="test" if split=="test" else "train"
        sources=[source] if source else self.manifest["sources"]
        for src in sources:
            kinds=["questions"]
            if split=="test":kinds.append("questions_heldout_phrasing")
            for kind in kinds:
                file=self.root/f"frozen_v13/{part}/{src}_{kind}.jsonl"
                if not file.exists():continue
                for index,line in enumerate(file.open()):
                    row=json.loads(line)
                    if row["split"]!=split or (track and row["track"]!=track):continue
                    if eligible_only and not row["v13_task"]:continue
                    row["question_id"]=f'{row["track"]}/{src}/{kind}/{index}'
                    yield row

    def new_questions(self,track,split,source=None,eligible_only=True):
        if track not in ("T1","T2"):raise ValueError(track)
        eligible=set(self.tasks[track]["cells"])
        if track=="T1":
            files=sorted((self.root/"new_tracks/data").glob("t1_*.jsonl"))
        else:
            files=sorted((self.root/"new_tracks/validated").glob("T2_*.jsonl"))
        for file in files:
            for line in file.open():
                row=json.loads(line)
                if row["split"]!=split or (source and row["source"]!=source):continue
                if eligible_only and row["task"] not in eligible:continue
                if track=="T1":
                    yield from pair_questions(row)
                else:
                    yield dict(row,track="T2")

    def new_eeg(self,row):
        if row["track"]=="T1":
            return [self.core_eeg(row["source"],row["split"],wid) for wid in row["inputs"]]
        if row["track"]=="T2":return [self._read_ref(row["cache_ref"])]
        raise ValueError(row["track"])
