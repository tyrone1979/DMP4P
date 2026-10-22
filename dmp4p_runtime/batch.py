"""Serial batch runner with frozen inputs, atomic day/run checkpoints and resume."""
import argparse,fcntl,hashlib,json,os,signal,time,traceback
from pathlib import Path
import pandas as pd,yaml
from dmp4p_runtime.checkpoint import SlotCheckpoint,atomic_json
from dmp4p_runtime.pool_snapshot import load_snapshot
from dmp4p_runtime.evaluator import UnifiedEvaluator
from dmp4p_runtime.multiday_common import run_multiday_system,_rebuild_day_from_rows

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--plan",type=Path,required=True);parser.add_argument("--out",type=Path,required=True);parser.add_argument("--dry-run",action="store_true");parser.add_argument("--resume",action="store_true")
    a=parser.parse_args();plan=json.loads(a.plan.read_text());root=Path(__file__).resolve().parents[1]
    files=list((root/"dmp4p_runtime").rglob("*.py"))+list((root/"algo").glob("*.py"))+[root/"ga_gpu.py",root/"ga_gpu_v2.py",root/plan["ga_config"]]
    for job in plan["jobs"]:files+=list((root/job["snapshot"]).glob("*"))
    for rel in plan.get("input_files", []):
        p = root / rel
        if not p.is_file():
            raise FileNotFoundError(p)
        files.append(p)
    digest={str(p.relative_to(root)):sha(p) for p in sorted(set(files)) if p.is_file()}
    payload={"plan":plan,"files":digest};fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    if a.dry_run:
        print(json.dumps({"solvers_invoked":False,"jobs":len(plan["jobs"]),"nominal_hours":len(plan["jobs"])*plan["task_seconds"]/3600,"fingerprint":fingerprint}));return
    if a.out.exists() and not a.resume:parser.error("Output exists; use --resume with matching fingerprint")
    a.out.mkdir(parents=True,exist_ok=True);lock=(a.out/"batch.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    record=a.out/"freeze.json"
    if record.exists():
        if json.loads(record.read_text())["fingerprint"]!=fingerprint:raise ValueError("Code/config/pool fingerprint mismatch")
    else:atomic_json(record,{"fingerprint":fingerprint,"payload":payload})
    atomic_json(a.out/"process.json",{"pid":os.getpid(),"started_unix":time.time(),"fingerprint":fingerprint})
    stopping=[False]
    def stop(signum,frame):stopping[0]=True
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,stop)
    cfg=yaml.safe_load((root/plan["ga_config"]).read_text());cfg.update(plan.get("config_overrides",{}))
    start=time.time()
    for i,job in enumerate(plan["jobs"]):
        key=job["key"];target=a.out/"run_results"/(key+".json")
        if target.exists():continue
        if stopping[0]:raise SystemExit(75)
        snap=load_snapshot(root/job["snapshot"]);ev=UnifiedEvaluator(snap["target"],snap["constraint_spec_obj"],config=cfg,structure_prior=snap.get("structure_prior") or {},personal_category_structure=snap.get("personal"))
        cp=SlotCheckpoint(a.out/"checkpoints"/(key+".json"),{"experiment":fingerprint,"run":key})
        print(f"RUN {i+1}/{len(plan['jobs'])} {key}",flush=True)
        try:
            result=run_multiday_system(method=job["method"],snap=snap,evaluator=ev,cfg=cfg,device=plan["device"],seed=job["seed"],task_wall_s=plan["task_seconds"],slot_cap_s=plan["slot_seconds"],max_days=plan["max_days"],checkpoint=cp,resume_state=cp.load(),stop_requested=lambda:stopping[0])
            # Reconstruct committed plans independently, including cross-day bans.
            seen=set();audit=[]
            for day in result["days"]:
                rebuilt=_rebuild_day_from_rows(snap["pool"],day["selected_rows"])
                if rebuilt is None:raise ValueError("reconstruction failed")
                verdict=ev.evaluate_solution(rebuilt);ids=set(rebuilt.food_id.astype(int))
                if not verdict.feasible or not verdict.bld_ok or seen.intersection(ids) or not day["prepared_in_budget"]:raise ValueError("delivery audit failed")
                seen|=ids;audit.append({"day":day["day_index"],"verified":True})
            result.update(run_key=key,instance_id=job["instance_id"],difficulty=job["difficulty"],fingerprint=fingerprint,independent_delivery_audit=audit,phase=plan["phase"],observation_mode=cfg.get("evidence_observation_mode"),population=cfg.get("evidence_population_size"))
            if any(s.get("solver_status")=="error" for s in result["slots"]):raise RuntimeError("program error in solver slot; see checkpoint slot diagnostics")
            atomic_json(target,result)
            print("DONE",key,"days",result["n_days_delivered"],"all_band",result["n_all_bands_days"],"error",result.get("program_error",False),flush=True)
        except InterruptedError:
            print("Graceful pause; slot checkpoint saved",flush=True);raise SystemExit(75)
        except Exception as exc:
            atomic_json(a.out/"errors"/(key+".json"),{"error":str(exc),"traceback":traceback.format_exc(),"fingerprint":fingerprint});raise
    records=[json.loads(f.read_text()) for f in sorted((a.out/"run_results").glob("*.json"))]
    frame=pd.DataFrame([{k:d.get(k) for k in ["run_key","method","instance_id","difficulty","seed","n_days_delivered","n_all_bands_days","ge7","program_error"]} for d in records]);frame.to_csv(a.out/"run_summary.csv",index=False)
    summary={"completed":len(records),"planned":len(plan["jobs"]),"program_errors":sum(bool(d.get("program_error")) for d in records),"fingerprint":fingerprint,"wall_since_launch_s":time.time()-start,"phase":plan["phase"]}
    atomic_json(a.out/"summary.json",summary);print(json.dumps(summary),flush=True)

if __name__=="__main__":main()
