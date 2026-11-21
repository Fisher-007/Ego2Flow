# Configuration

| File | Purpose |
| --- | --- |
| `suite.json` | Driving methods, scene directories and applicable scenes for each metric |
| `six_cities_v1.json.gz` | Scene definitions for all 48 scenes, extracted by `python evaluate.py prepare` |
| `runtime.sequential.example.json` | One local GPU, with a map project for each city |
| `runtime.parallel.example.json` | Six local GPUs, one per city |
| `methods/*.json` | Driving method, adapter, parameters and weights |
| `model-environment.example.json` | Example paths for the built-in models and their Python environments |
| `eight_metrics.json`, `reference_bev_energy.json`, `score.json` | Metric thresholds, reference vehicle parameters and score normalization |
| `sensors/av2_front_rig.json` | Camera calibration for OneVL and Alpamayo |
| `sensors/uniad_rig.json` | Camera calibration for UniAD |

Each scene's `scene_dir` points to its routes, requests and signal settings under `configs/scenes/<SCENE>/`. Set `SCENARIO_ROOT` to use a different location.

On one GPU, set `gpu`, `cpus` and `port` in `resource` and choose each city's project in `resource.maps`. For parallel runs, assign these settings per city in `workers`. Use `taskset -pc $$` to list available CPUs. Reserve the base port and offsets +1, +2 and +10 for each worker.

Configuration paths can use environment variables. The runner and scorer use the active interpreter by default. Set `resource.python` for the runner, `resource.metrics_python` for scoring or `python` in a method configuration for that policy.
