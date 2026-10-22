# V2.1 generic runtime and named supplementary algorithms

This package exposes fixed-pool multi-day search, independent validation, in-budget online selection, food-ID exclusion and resumable batches. It isolates new development from the original frozen benchmark modules. The original full V2.1 algorithm remains the default; additional methods are explicitly named.

## Environment

Python3.11; numpy, pandas, torch, scipy, scikit-learn, PyYAML, tqdm, pyarrow. CP-SAT additionally requires ortools. Choose a compatible PyTorch/CUDA installation for GPU execution.

## Run a trusted snapshot

```bash
PYTHONPATH=. python -m dmp4p_runtime.cli --snapshot PATH_TO_TRUSTED_SNAPSHOT --algorithm v2_quality_recovery --device cuda:0 --task-seconds 210 --slot-seconds 30 --max-days 7 --out result.json
```

Use `--dry-run` to load inputs and inspect without invoking a solver. `--population-size` explicitly overrides both algorithm population settings; it changes the experimental setting. Existing output files are never overwritten. Snapshot pickle loading is for trusted local snapshots only; prefer parquet for distribution. Config defaults to config/ga_multiobj_weights.yaml.

CLI example fixture (synthetic values, not nutritional advice): `--snapshot examples/runtime_toy --algorithm cpsat_band --max-days 1 --task-seconds 5 --slot-seconds 5 --out toy_result.json`.

## Batch

`python -m dmp4p_runtime.batch --plan PLAN.json --out NEW_DIRECTORY`; add `--resume` for a matching checkpoint/fingerprint. Use `--dry-run` before execution. Plans specify trusted snapshot paths, algorithm, seed, device, task/slot budgets and config overrides. Each committed day is reconstructed and checked. SIGUSR1/SIGTERM stops at the next day boundary; a hard crash exhausts remaining run budget conservatively.

For >2h jobs use a detached process, closed stdin and saved console/PID, then use the same plan/output with --resume. Checkpoints are day boundaries, not optimizer-state snapshots mid-generation.

## Scientific scope

No guarantee that seven days exist or will be found. Independent acceptance is separate from float32 search proxies. Food identity exclusion is not semantic dish uniqueness. The learned category bands are not evidence of clinical adequacy or cooking/portion reasonableness. Later practical-meal modifications require a new recorded method/data version.

## Reconciled NHANES priors

The default config/ga_multiobj_weights.yaml and config/structure_learn.yaml now select the reconciled priors. The explicitly named config/ga_reconciled_v1.yaml records the same source-release reconciliation that retains 2017–March 2020 and excludes standalone 2017–2018. Aggregate category/plate models and meal-share, weight and frequency summaries are versioned under outputs/nhanes_release_reconciled_1720_v1; no raw respondent tables are included. Category and plate models use 7459 matched food-only training days. The portion dictionary is a shared fixed modeling resource. The configuration now routes all prior loads explicitly; frequency caches are keyed by configured file and thresholds. Batch plans may declare input_files so referenced model bytes are included in freeze fingerprints.

Rebuild candidate snapshots when changing priors; replacing a score model alone is insufficient. Historical result artifacts were generated with their original models and must not be pooled with a new-prior run. The new-prior comparison is currently in progress; no new performance ranking is claimed here.
