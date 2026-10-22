"""Atomic slot-boundary checkpoints with conservative crash budget accounting."""
from pathlib import Path
import json,os,time

VERSION="multiday_checkpoint_v1"

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,default=str)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(str(path.parent),os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

class SlotCheckpoint:
    def __init__(self,path,identity):
        self.path=Path(path);self.identity=identity
    def load(self):
        if not self.path.exists():return None
        state=json.loads(self.path.read_text())
        if state.get("identity")!=self.identity:raise ValueError("checkpoint fingerprint mismatch")
        if state.get("version")!=VERSION:raise ValueError("checkpoint version mismatch")
        if state.get("phase")=="inflight":
            # There is no saved optimizer/RNG state or trustworthy final elapsed
            # time after SIGKILL/host crash. Never give another free search slot.
            state["consumed_s"]=state["task_wall_s"]
            state["recovery_note"]="unclean_interruption_remaining_run_budget_exhausted"
        return state
    def save(self,state):
        atomic_json(self.path,{**state,"identity":self.identity,"version":VERSION,"saved_unix":time.time()})
