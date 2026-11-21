# Add a driving policy

A policy factory loads the model for a map and creates a session for each vehicle it controls. Sessions share weights and keep their own navigation, sensor history and controller state. The [route follower](../examples/route_follower.py) shows the interface for a pose-only policy.

## Run the example

Copy `configs/suite.json` to `configs/suite.local.json`. Add `"Example": "configs/methods/Example.json"` to `methods` and append `"Example"` to `method_order`. Run PAO-S01 with your single-GPU configuration, `configs/runtime.local.json`.

```bash
python evaluate.py doctor --suite configs/suite.local.json --runtime configs/runtime.local.json --methods Example --scenes PAO-S01
python evaluate.py run --suite configs/suite.local.json --runtime configs/runtime.local.json --methods Example --scenes PAO-S01
```

Find both deployments under `outputs/benchmark/Example/runs/`.

## Register your method

Save your policy settings in `configs/methods/MyPolicy.local.json`. The `factory` field uses `module:function`. Install your package in the method's Python environment or add the parent of `my_policy/` to `PYTHONPATH`.

```json
{
  "id": "MyPolicy",
  "name": "My Policy",
  "version": "1",
  "visual_policy": {
    "kind": "custom",
    "factory": "my_policy.adapter:create",
    "checkpoint": "/path/to/weights.pth"
  }
}
```

Add `"MyPolicy": "configs/methods/MyPolicy.local.json"` to `methods` and `"MyPolicy"` to `method_order` in your local suite. Run it with `--methods MyPolicy`. The `id` names the output directory and `name` appears in reports. Change `version` when you update the policy or weights so the next run uses them.

The factory receives `visual_policy`, including your model options and, by default, the scene seed. Set `visual_policy.seed_torch` to `true` to seed PyTorch. For a separate model environment, set the top-level `python` field to its interpreter. That interpreter also runs the CARLA client.

## Model and vehicle sessions

`create(config)` returns an object with the following members.

| Member | What to provide |
| --- | --- |
| `sensors` | A list of sensor declarations. Use `[]` for a pose-only policy. |
| `description` | A JSON-serializable dictionary containing `"name"`, saved in `run.json`. |
| `new_vehicle()` | Return a new vehicle session. |
| `close()` | Release model resources when the method finishes on this map. |

Each vehicle session provides the following methods and attributes.

| Method or attribute | Purpose |
| --- | --- |
| `bind_request(request_id)` (optional) | Receive the vehicle's integer request ID before `reset`. |
| `reset(plan)` | Initialize navigation from the ordered list of `(carla.Waypoint, RoadOption)` pairs. |
| `localization_sensor_x_m` | Forward offset of the localization sensor in metres. Use `0.0` at the vehicle origin. |
| `observe_inputs(inputs, preparation=False)` (optional) | Collect sensor history during preparation and driving. Use the CARLA frame number to store each measurement once. |
| `step(inputs, speed, position_xyz, yaw_deg, *, gps_position_xyz)` | Return `(control, diagnostics)` for this vehicle. |
| `close()` | Release the vehicle session when it leaves the simulation or the observation ends. |

### Inputs

`inputs` maps sensor IDs to CARLA measurements from the same frame. Camera images carry BGRA `raw_data`, dimensions, a frame number and a simulation timestamp. Set `sensor_tick` to `0` to receive every world step (0.05 s). For a lower model frame rate, select frames by timestamp in the session.

```python
sensors = [
    dict(id="front", type="sensor.camera.rgb", x=1.0, y=0.0, z=1.6,
         roll=0.0, pitch=0.0, yaw=0.0, width=800, height=450, fov=90, sensor_tick=0),
    dict(id="imu", type="sensor.other.imu", x=0.0, y=0.0, z=0.0,
         roll=0.0, pitch=0.0, yaw=0.0, sensor_tick=0),
]
```

Use your model's sensor poses and calibration. Mount positions are in metres from the vehicle origin (x forward, y right, z up), and rotations are in degrees. Set options such as LiDAR channels and range in `blueprint_attributes`.

`position_xyz` and `gps_position_xyz` are the vehicle origin and localization sensor position in CARLA world coordinates. `yaw_deg` is the vehicle heading in degrees. `speed` is signed forward speed in m/s.

### Controls

Return `steer` in [−1, 1], `throttle` and `brake` in [0, 1], and a JSON-serializable diagnostics dictionary.

```python
control = {"steer": float(steer), "throttle": float(throttle), "brake": float(brake)}
return control, {}
```

For a policy that predicts at least two waypoints at 0.5 s intervals, you can use Ego2Flow's `TrajectoryTracker`. Import it from `benchmark.policies.trajectory_tracker` and create `self.tracker = TrajectoryTracker()` in the vehicle's `reset`. Pass waypoints in ego coordinates as `[right, forward]` in metres and return `self.tracker.step(waypoints, speed)` from `step`.
