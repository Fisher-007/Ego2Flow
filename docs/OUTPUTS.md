# Output files

Use `--output PATH` to change the default output directory shown below.

```text
outputs/benchmark/
├── report/                            # Scores and metrics across the selected runs
├── IDM/runs/
│   ├── PAO-S01_H/                     # Homogeneous deployment
│   └── PAO-S01_M/                     # Mixed deployment
├── logs/PAO/records.log               # Map loading, simulation progress and scoring
├── logs/PAO_launcher.log              # Worker startup and terminal output
├── workers/                           # Per-city status, method queues and process logs
├── carla_servers/PAO/                 # CARLA server log, launcher log and process state
├── cache/                             # Per-city navigation cache for Traffic Manager
├── backup/                            # Previous attempts replaced on resume
└── suite.json                         # Suite, methods and scenes selected for this run
```

## Report

| File under `report/` | Contents |
| --- | --- |
| `RESULTS.md`, `scores.csv` | Eight normalized scores per method, failure fraction, total score and number of observed runs |
| `raw_metrics.csv` | Metrics in their original units, averaged equally over H/M and then across the applicable scenes. A mean is blank if a required value is missing. |
| `conditional_raw.csv` | Means over unflagged H/M runs within each scene, then across scenes that have a value |
| `conditions.csv` | One row per method, scene and deployment, with raw values, normalized scores (`score:` columns), status and failure reasons |
| `results.json` | Per-run results, H/M and scene summaries, method totals and score parameters |

## One run

| File under `<method>/runs/<scene>_<deployment>/` | Contents |
| --- | --- |
| `evaluation.json` | Measured metrics, units and coverage |
| `peak_gain.json` | External leader and follower speed responses in a propagation scene |
| `run.json` | Run outcome and elapsed simulation time |
| `requests.json` | Request arrivals, admissions and completions, including requests still waiting or unfinished at cutoff |
| `config.json`, `submitted_config.json` | Effective run settings and the original settings submitted by the worker |
| `input_receipt.json` | Method and scene inputs used for resume, including the routes used for rescoring |
| `frames.jsonl.gz`, `initial_states.jsonl` | Per-frame vehicle states, controls and signals, and each vehicle's initial state |
| `events.jsonl` | Scenario events, including vehicle admissions and departures |
| `collisions.jsonl` | Collision callbacks whose frame falls within the observation window |
| `collision_callbacks.jsonl`, `collision_window.json` | Raw callback stream and observation frame boundaries |
| `progress.json`, `public_status.json`, `scoring.log` | Latest simulation progress, execution status and scoring output |
| `initialization.json`, `native_signals.json` | Initial vehicle preparation and native traffic signals when recorded |
| `route_outcomes.json`, `run_diagnostics.json` | Route departures and run-level traffic diagnostics |

## Rebuild a report

Use the same `--output` directory for every city batch. After the last batch, run `report` for the methods you evaluated. Omit `--cities` and `--scenes` to include all six cities. For a custom suite, pass the same `--suite` file.

```bash
python evaluate.py report --output outputs/benchmark --methods TM IDM
```

The combined table is saved to `report/RESULTS.md`. A service-only selection such as `PAO-S01` has no propagation score, so its total is blank even when both deployments finish.

`rescore` recalculates the metrics from the recorded observations and saved routes, then rebuilds the report.

```bash
python evaluate.py rescore --output outputs/benchmark --methods TM IDM
```
