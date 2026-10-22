# DMP4P V2.1 — V2-Quality

**Current default branch.** V2.1 extends the structure-guided Bronze search with bounded distinct elites, feasible-parent selection, structure-directed same-cell improvement, nutrient-gap recovery and scheduled structured immigrants. These mechanisms improve search beyond first feasibility; production defaults are not silently replaced.

## Version identity

- Public method name: **V2-Quality (V2.1)**.
- Frozen experiment identifier: `v2_quality_recovery`.
- Algorithm source anchor: `0effdf5`; frozen seven-day protocol: `829a793`.
- V2.0 denotes the historical Bronze production generation.
- GitHub public core-code export anchor: `cd89bb1`; public module paths differ from the internal development layout.

## Runtime components

Use Python 3.11 with NumPy, pandas, PyTorch, SciPy, scikit-learn, PyYAML, tqdm and pyarrow. Supply food/portion rows, nutrient targets and learned personal structure priors separately. CUDA requires a compatible PyTorch build and GPU driver.

```python
from algo.v2_quality_recovery import FoodCombinerV2QualityRecovery
```

The recovery class inherits the food-pool, target and configuration interfaces of `FoodCombinerV2`. Independently verified ranking uses the injected `quality_evaluator`. The generic command below still invokes the **production V2** CLI; it is not an end-to-end V2.1 launcher:

```bash
PYTHONPATH=. python find_bronze_multiobj.py --help
```

## Reproducibility status

The internal frozen seven-day evaluation used 24 structure-target units, four difficulty strata, four methods and three seeds (288 runs). V2.1 completed seven days in 54/72 runs and delivered 447 days, including 319 satisfying all tested category bands. These are system-level benchmark results, not whole-population generation or clinical validation. All tested category lower bounds were zero. Preparation within budget is distinct from strict response latency.

This public branch now includes `dmp4p_runtime`: independent validation, online selection, fixed-pool multi-day search, and fingerprinted day-checkpointed batches. It intentionally excludes manuscripts and paper-specific experimental/analysis scripts. A synthetic input fixture and algorithm correctness tests are included. Reproducing the original reported benchmark additionally requires its exact frozen pools, configuration, code revision and run plan; a later runtime package is not a claim that all original runs have been repeated. See V2_RUNTIME.md and RELEASE_VERSION.md for export scope. The existing MIT license applies to the public code; it does not relicense third-party data.

## Data and follow-up scope

The earlier Figshare 30153778 deposit belongs to **V1**. Historical V2.0 exports, the current V2.1 seven-day benchmark and any future full-population V2.1 release must be versioned separately. A full-population V2.1 corpus has not been generated in this phase. Practical meal/portion reasonableness remains follow-up work; new rules require a separately recorded version and validation.

## Versions and branches

- [V1](https://github.com/tyrone1979/DMP4P/tree/V1): the earlier DMP4P generation. Its published data are at [Figshare](https://doi.org/10.6084/m9.figshare.30153778).
- [V2.0](https://github.com/tyrone1979/DMP4P/tree/V2.0): the historical structure-guided Bronze production generation.
- [V2.1](https://github.com/tyrone1979/DMP4P/tree/V2.1): V2-Quality, the current default branch.

Branch names are case-sensitive. Figshare repository release `v7` is a revision of the **V1 deposit**, not algorithm V7. Neither V2.0 nor V2.1 should cite that DOI as its generated output.

## Interpretation and limitations

Outputs are computational fixed-portion food combinations, not observed participant diets or clinically validated prescriptions. Nutrient feasibility refers only to the active supplied bounds. Meal/category priors describe empirical structure; conformity is not proof of culinary coherence, dietary adequacy or user acceptance. Exact `food_id` exclusion does not guarantee semantically different dishes. Source data, learned priors, generated artifacts and code versions must be recorded separately.

## Generic V2.1 runtime entry point

```bash
PYTHONPATH=. python -m dmp4p_runtime.cli --snapshot examples/runtime_toy --algorithm cpsat_band --max-days 1 --task-seconds 5 --slot-seconds 5 --out toy_result.json
```

The fixture is synthetic and tests integration only. For V2.1 use `--algorithm v2_quality_recovery` with a compatible trusted fixed-pool snapshot. See [runtime usage](dmp4p_runtime/README.md). CP-SAT requires OR-Tools. Named structural-GA and grouped ablations are supplementary algorithms, not changes to the frozen V2.1 default.

### Source-release reconciliation (22 October 2026)

The explicit reconciled prior configuration is config/ga_reconciled_v1.yaml. It retains NHANES 2017–March 2020 and excludes the overlapping standalone 2017–2018 release. New candidate snapshots and evaluation are required when using this configuration. Existing benchmark artifacts retain their recorded provenance; see dmp4p_runtime/README.md.
