# V2 runtime code

This branch contains the Bronze runtime dependency closure exported from source commit d558b44. Manuscripts, paper experiment launchers, benchmark analysis scripts, datasets, credentials and internal Git history are not included in this update. The existing main-branch application files are retained.

Entry point: `python find_bronze_multiobj.py --help`. Supply NHANES/FNDDS inputs and learned structure priors separately; this update does not distribute those artifacts. See the command-line options for input paths and GPU selection.

Production class: `ga_gpu_v2.FoodCombinerV2`. Quality variants: `algo.v2_quality_search.FoodCombinerV2Quality` and `algo.v2_quality_recovery.FoodCombinerV2QualityRecovery`. These retain the source constructor interfaces; the recovery variant accepts an independent evaluator through its `quality_evaluator` attribute. No benchmark evaluator or experimental runner is included. Production behavior is not automatically replaced by a quality variant.

The two quality modules have been relocated from their development directory; the recovery module import is adjusted accordingly. No search behavior was changed during export.

Use Python 3.11 with numpy, pandas, torch, scipy, scikit-learn, PyYAML, tqdm and pyarrow. CUDA availability depends on the installed PyTorch build. Existing repository LICENSE is retained. Exact food identifiers can be excluded across days; semantic dish uniqueness and clinical suitability are not guaranteed.

## Runtime integration update

The generic dmp4p_runtime package now supplies the independent evaluator, online selector and multi-day/checkpointed entry points. Use python -m dmp4p_runtime.cli; the historical production CLI remains separate. Prior statements that those runtime components were omitted describe the earlier cd89bb1 core export. Exact frozen experiment inputs are still required to reproduce that experiment.
