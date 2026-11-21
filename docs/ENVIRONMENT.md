# Environment

## Simulator and maps

Extract the six v1.0 map projects under `Ego2Flow_Maps/v1_0/`, preserving each `native/` directory.

```text
Ego2Flow_Maps/
└── v1_0/
    ├── Ego2Flow_PAO_v1_0/native/Unreal/CarlaUE4/
    │   ├── CarlaUE4.uproject
    │   └── Content/Ego2Flow_PAO_v1_0/Maps/Ego2Flow_PAO_v1_0/Ego2Flow_PAO_v1_0.umap
    ├── Ego2Flow_DTW_v1_0/native/
    ├── Ego2Flow_ATX_v1_0/native/
    ├── Ego2Flow_MIA_v1_0/native/
    ├── Ego2Flow_PIT_v1_0/native/
    └── Ego2Flow_WDC_v1_0/native/
```

Set the paths to your maps, Unreal Engine 4.26.2 build and CARLA 0.9.16 Python API in the shell used to start the evaluator.

```bash
export EGO2FLOW_MAPS=/path/to/Ego2Flow_Maps
export UE4_ROOT=/path/to/UnrealEngine_4.26
export CARLA_PYTHON_API=/path/to/CARLA_0.9.16/PythonAPI/carla

ls "$UE4_ROOT/Engine/Binaries/Linux/UE4Editor"
ls "$CARLA_PYTHON_API/agents/navigation/basic_agent.py"
for city in PAO DTW ATX MIA PIT WDC; do
    map_name="Ego2Flow_${city}_v1_0"
    project="$EGO2FLOW_MAPS/v1_0/$map_name/native/Unreal/CarlaUE4"
    ls "$project/CarlaUE4.uproject" "$project/Content/$map_name/Maps/$map_name/$map_name.umap"
done
```

The evaluator launches the native map project with Vulkan. Set `DISPLAY` if your graphics setup requires it.

## Python environments

TM and IDM use the packages in `requirements/runtime.txt` and the navigation agents under `CARLA_PYTHON_API`. If your CARLA build supplies a Python 3.10 client wheel, install it before the runtime requirements. For offline scoring, use `python -m pip install -r requirements/metrics.txt`.
