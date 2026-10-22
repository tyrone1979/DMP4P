# DMP4P V2.0 — historical Bronze production

V2.0 selects fixed-portion NHANES/FNDDS food rows for a **structure stratum × nutrient-target** unit. The production search uses meal × food-category cells, nutrient guidance and sequential food-identity exclusion to seek up to seven daily alternatives. Category gram bands are soft guidance; prohibited cells and other programmed gates are handled separately.

## Source and data identity

Historical source anchor: `ce00c95` (2026-10-19). This is a historical code anchor, not a reconstructed immutable campaign-launch fingerprint. The historical export is `outputs/figshare_bronze_v2/` in the Linux working environment; that local path is not a public download URL. Its closure snapshot contains 7,981 units, 7,945 with at least one solution and 44,081 distinct generated days. Food-row exports replicate plans for users sharing a unit; remove user copies before counting plans. Later retries are not included in that snapshot.

## Setup and runtime

Use Python 3.11 with NumPy, pandas, PyTorch, SciPy, scikit-learn, PyYAML, tqdm and pyarrow. Install a PyTorch build compatible with your CUDA environment when using a GPU. Source dietary inputs, target tables, optional FPED resources and learned structure priors must be supplied separately.

```bash
PYTHONPATH=. python find_bronze_multiobj.py --help
```

Production class: `ga_gpu_v2.FoodCombinerV2`. The CLI documents input/output paths, GPU selection and search parameters; inspect its options before launching a batch. This branch does not contain the V2-Quality recovery subclass.

## Dataset boundaries

The existing Figshare DOI 30153778 belongs to V1, not this V2.0 export. A separately released V2.0 dataset requires its own explicit version and provenance. No new data DOI is asserted here.
## Versions and branches

- [V1](https://github.com/tyrone1979/DMP4P/tree/V1): the earlier DMP4P generation. Its published data are at [Figshare](https://doi.org/10.6084/m9.figshare.30153778).
- [V2.0](https://github.com/tyrone1979/DMP4P/tree/V2.0): the historical structure-guided Bronze production generation.
- [V2.1](https://github.com/tyrone1979/DMP4P/tree/V2.1): V2-Quality, the current default branch.

Branch names are case-sensitive. Figshare repository release `v7` is a revision of the **V1 deposit**, not algorithm V7. Neither V2.0 nor V2.1 should cite that DOI as its generated output.

## Interpretation and limitations

Outputs are computational fixed-portion food combinations, not observed participant diets or clinically validated prescriptions. Nutrient feasibility refers only to the active supplied bounds. Meal/category priors describe empirical structure; conformity is not proof of culinary coherence, dietary adequacy or user acceptance. Exact `food_id` exclusion does not guarantee semantically different dishes. Source data, learned priors, generated artifacts and code versions must be recorded separately.
