# DMP4P V2.1 — V2-Quality

Public method version: V2.1. Internal frozen experiment identifier: v2_quality_recovery.
Algorithm source anchor: 0effdf5; frozen experiment protocol: 829a793; public runtime export base: cd89bb1.

This branch contains generic runtime code, not manuscripts or paper experiment/analysis scripts. Independent evaluator injection and a complete online selection/delivery entry point remain integration work; class importability alone is not full reproduction of the paper experiment. The production CLI still uses production V2 unless explicitly integrated with the quality class.

V1: earlier DMP4P Figshare dataset 10.6084/m9.figshare.30153778 (repository release v7 does not mean algorithm V7). V2.0: historical Bronze production method. V2.1: V2-Quality. The current seven-day benchmark is not a full-population V2.1 dataset. Later practical-meal improvements require a separate version decision.

## Runtime integration update

The generic dmp4p_runtime package now supplies the independent evaluator, online selector and multi-day/checkpointed entry points. Use python -m dmp4p_runtime.cli; the historical production CLI remains separate. Prior statements that those runtime components were omitted describe the earlier cd89bb1 core export. Exact frozen experiment inputs are still required to reproduce that experiment.
