"""Generic, audited multi-day CLI on a trusted fixed-pool snapshot."""
import argparse,json,hashlib,os
from pathlib import Path
import yaml
from dmp4p_runtime.factory import SUPPORTED_METHODS
from dmp4p_runtime.pool_snapshot import load_snapshot
from dmp4p_runtime.evaluator import UnifiedEvaluator
from dmp4p_runtime.multiday_common import run_multiday_system

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot",type=Path,required=True,help="Trusted snapshot directory (parquet/meta/personal); do not load untrusted pickle snapshots")
    p.add_argument("--config",type=Path,default=Path("config/ga_multiobj_weights.yaml"))
    p.add_argument("--algorithm",choices=SUPPORTED_METHODS,default="v2_quality_recovery")
    p.add_argument("--device",default="cpu");p.add_argument("--seed",type=int,default=0)
    p.add_argument("--task-seconds",type=float,default=210);p.add_argument("--slot-seconds",type=float,default=30)
    p.add_argument("--max-days",type=int,default=7);p.add_argument("--population-size",type=int)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    if min(a.task_seconds,a.slot_seconds,a.max_days)<=0:p.error("Budgets/days must be positive")
    if a.out.exists():p.error("Refusing to overwrite output")
    snap=load_snapshot(a.snapshot);cfg=yaml.safe_load(a.config.read_text())
    if a.population_size is not None:cfg["evidence_population_size"]=a.population_size
    ev=UnifiedEvaluator(snap["target"],snap["constraint_spec_obj"],config=cfg,structure_prior=snap.get("structure_prior") or {},personal_category_structure=snap.get("personal"))
    if a.dry_run:
        print(json.dumps({"solvers_invoked":False,"algorithm":a.algorithm,"rows":len(snap["pool"]),"target":snap["target"],"population_override":a.population_size}));return
    result=run_multiday_system(method=a.algorithm,snap=snap,evaluator=ev,cfg=cfg,device=a.device,seed=a.seed,task_wall_s=a.task_seconds,slot_cap_s=a.slot_seconds,max_days=a.max_days)
    result["runtime_interface_version"]="v21_runtime_v1"
    result["config_sha256"]=hashlib.sha256(a.config.read_bytes()).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True);tmp=a.out.with_suffix(a.out.suffix+".tmp")
    tmp.write_text(json.dumps(result,ensure_ascii=False,default=str,indent=2));os.replace(tmp,a.out)
    print(json.dumps({"algorithm":a.algorithm,"delivered_days":len(result.get("days",[])),"out":str(a.out)}))

if __name__=="__main__":main()
