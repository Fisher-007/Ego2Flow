"""Run traffic requests with the configured driving methods on a CARLA map."""

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import queue
import sys
import time

from benchmark.runtime.run_diagnostics import summarize
from benchmark.scenes.demand import scheduled_requests, entry_heads
from benchmark.scenes.participants import apply_participant_defaults, record_participant
from benchmark.navigation.route_waypoints import resolve_waypoint, native_endpoint_index
from benchmark.navigation.corridor import Corridor, CorridorTracker
from benchmark.scenes.entry_geometry import entry_is_clear
from benchmark.runtime.control_submission import submit_commands, control_echo
from benchmark.runtime.actor_lifecycle import retire_missing_vehicles
from benchmark.runtime.state_recording import road_fields, initial_state
from benchmark.scenes.receiving_boundary import beyond_receiving_gate
from benchmark.runtime.collision_journal import CollisionJournal
from benchmark.runtime.sensor_history import observe_admitted_history
from benchmark.scenes.leader_pulse import evaluated_ids, external_ids, target_speed_mps, validate as validate_leader


def read(path):
    return json.loads(path.read_text())



def save_observer_image(image, directory, config):
    """Record frames from separate presentation cameras."""
    if config.get("observer_image_format", "png") == "jpg":
        import cv2
        import numpy as np
        pixels = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
        cv2.imwrite(str(directory / f"frame_{image.frame:08d}.jpg"), pixels[:, :, :3],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        image.save_to_disk(str(directory / f"frame_{image.frame:08d}.png"))

def save_policy_image(image, path):
    """Save unchanged native BGR pixels with fast lossless PNG compression."""
    import cv2
    import numpy as np
    pixels = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
    if not cv2.imwrite(str(path), pixels[:, :, :3], [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise IOError("Could not save policy RGB: " + str(path))


def drain_observer(stream, directory, config, first_frame):
    """Record sparse presentation samples by their actual sensor frame IDs."""
    latest=None
    while True:
        try:
            image=stream.get_nowait()
        except queue.Empty:
            return latest
        if image.frame>=first_frame:
            save_observer_image(image,directory,config)
            latest=image.frame


def pulse_target_speed(pulse, base_speed, request_id, elapsed):
    if pulse and request_id == pulse["request_id"] and pulse["start_s"] <= elapsed < pulse["end_s"]:
        return pulse["target_speed_kmh"]
    return base_speed


def incident_stop_reason(collisions, departure_reason, enabled):
    if not enabled:
        return None
    if collisions:
        return 'collision'
    if departure_reason in ('left_supplied_route', 'left_legal_corridor'):
        return departure_reason
    return None


class SignalClock:
    """Apply the supplied phase colors before each tick, restoring on exit."""

    def __init__(self, world, data, carla, supply_event=None):
        self.carla, self.data = carla, data
        self.supply_event = supply_event
        self.programs = {p["id"]: p["phases"] for p in data["source_programs"]}
        if supply_event:
            source=supply_event['source_signal_id']
            cycle=sum(p['duration_s'] for p in self.programs[source])
            controlled=[b for b in data['bindings'] if b['source_signal_id']==source]
            if (not controlled or not all(b['phase_states'][-1]=='Red' for b in controlled)
                    or any(abs(t/cycle-round(t/cycle))>1e-6 for t in (supply_event['start_s'],supply_event['end_s']))):
                raise ValueError('Temporary all-red event must enter/release at cycle boundaries following normal all-red clearance')
        available = {a.get_opendrive_id(): a for a in world.get_actors().filter("traffic.traffic_light*")}
        self.actors = {b["signal_id"]: available[b["signal_id"]] for b in data["bindings"]}
        self.original = {key: (actor.get_state(), actor.is_frozen()) for key, actor in self.actors.items()}
        self.last = {}
        for actor in self.actors.values():
            actor.freeze(True)

    def update(self, elapsed):
        for binding in self.data["bindings"]:
            phases = self.programs[binding["source_signal_id"]]
            position = elapsed % sum(p["duration_s"] for p in phases)
            for index, phase in enumerate(phases):
                if position < phase["duration_s"]:
                    key, state = binding["signal_id"], binding["phase_states"][index]
                    event = self.supply_event
                    if (event and event['start_s'] <= elapsed < event['end_s']
                            and binding['source_signal_id'] == event['source_signal_id']):
                        state = 'Red'
                    if self.last.get(key) != state:
                        self.actors[key].set_state(getattr(self.carla.TrafficLightState, state))
                        self.last[key] = state
                    break
                position -= phase["duration_s"]

    def close(self):
        for key, (state, frozen) in self.original.items():
            self.actors[key].set_state(state)
            self.actors[key].freeze(frozen)


def static_world_data(world, cache):
    """Only immutable native map/TM geometry is shared within one episode."""
    if cache.get('episode_id') != world.id:
        cache.clear()
        cache.update(episode_id=world.id, native_map=world.get_map())
    return cache


def main(argv=None, shared_policy=None, static_cache=None):
    from benchmark.runtime.progress_log import progress
    wall_started = time.perf_counter()
    last_progress_wall = wall_started
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=int(os.environ.get("E2F_RPC_PORT", 23000)))
    parser.add_argument("--carla-root", type=Path, default=Path(os.environ.get("CARLA_ROOT", ".")))
    parser.add_argument("--carla-python-api", type=Path,
                        help="Directory containing CARLA's agents/ navigation package")
    args = parser.parse_args(argv)
    api = args.carla_python_api or args.carla_root / 'PythonAPI/carla'
    sys.path.insert(0, str(api.resolve()))
    import carla
    import random
    import numpy as np
    from agents.navigation.basic_agent import BasicAgent
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from benchmark.navigation.supplied_route_planner import SuppliedRoutePlanner
    from agents.navigation.local_planner import RoadOption
    from shapely.geometry import LineString, Point

    info = read(args.scene_dir / "map_info.json")
    config = read(args.config)
    seed = config.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    scene = next(a for a in read(args.scene_dir / "scene_index.json") if a["scene_id"] == config["scene_id"])
    supplied_routes = read(args.scene_dir / "candidate_routes.json")
    signals = read(args.scene_dir / "signal_programs.json")
    requests = scheduled_requests(config)
    apply_participant_defaults(config, requests)
    arrivals = [r["arrival_s"] for r in requests]
    route_names = list(dict.fromkeys(r["route"] for r in requests))
    from benchmark.scenes.scene_domain import SceneDomain
    scene_domain = SceneDomain(info.get('known_map_issues', []),
                               scene.get('reviewed_driving_polygon_carla_xy_m'))
    scene_domain.validate_routes(supplied_routes, route_names)
    dt = config["fixed_delta_seconds"]
    if arrivals != sorted(arrivals) or any(t < 0 for t in arrivals) or dt <= 0:
        raise ValueError("Arrival times must be sorted and nonnegative, with a positive simulation step")
    pulse = config.get("speed_pulse")
    leader = config.get("external_leader")
    if leader:
        validate_leader(leader, requests)
    intervention_ids = external_ids(config)
    rule = config.get("rule_controller", {})
    visual = config.get("visual_policy")
    visual_ids = visual.get("request_ids", [0]) if visual else []
    method_ids = list(visual_ids)
    rule_ids = set()
    if config.get("deployment"):
        setting = config["deployment"]
        if setting not in ("H", "M", "R"):
            raise ValueError("Deployment must be H, M or R")
        method_ids = evaluated_ids(config, requests)
        if rule:
            if visual:
                raise ValueError("Select one evaluated method, not rule and visual together")
            rule_ids = set(method_ids) if setting != "R" else {r["id"] for r in requests if r["id"] not in intervention_ids}
            visual_ids = []
        else:
            visual_ids = list(method_ids)
            if visual_ids and not visual:
                raise ValueError("Evaluated requests require a visual or rule policy")
        if setting == "R":
            visual = None
        for request in requests:
            request["deployment_role"] = ("external_leader" if request["id"] in intervention_ids else
                "evaluated" if request["id"] in method_ids else "background")
    if len(set(visual_ids)) != len(visual_ids) or any(i not in range(len(arrivals)) for i in visual_ids):
        raise ValueError("visual_policy.request_ids must be unique scheduled demand IDs")
    if visual and abs(dt - .05) > 1e-9:
        raise ValueError("The supplied visual controllers are evaluated at 20 Hz")
    if pulse and not (0 <= pulse["request_id"] < len(arrivals)
                      and 0 <= pulse["start_s"] < pulse["end_s"] < config["duration_s"]
                      and 0 <= pulse["target_speed_kmh"] < requests[pulse["request_id"]]["target_speed_kmh"]):
        raise ValueError("speed_pulse needs an existing request, an in-run interval and a lower nonnegative target speed")
    output = args.output.resolve()
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {"map_id": info["map_id"], "map_version": info["map_version"], "scene_id": scene["scene_id"],
              "scene_dir": str(args.scene_dir.resolve()), "controller": config["controller"],
              "scope": "Traffic-system observation under the declared request, metric-scope and time-window configuration",
              "passed": False, "seed": seed}
    report['scene_domain'] = dict(known_map_issues=info.get('known_map_issues', []),
        reviewed_driving_polygon_carla_xy_m=scene.get('reviewed_driving_polygon_carla_xy_m'),
        policy='Record individual departures or entry into excluded map regions; continue the scheduled whole-scene observation without control override or individual failure removal.',
        qualification='Restrictions only; their presence does not certify physical or visual road quality.')
    report.update(deployment=config.get("deployment"),
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  evaluated_request_count=len(method_ids),
                  ordinary_request_count=len(requests)-len(intervention_ids),
                  external_intervention_count=len(intervention_ids),
                  requested_evaluated_fraction=len(method_ids)/(len(requests)-len(intervention_ids)) if len(requests)>len(intervention_ids) else None)
    signal_diagnostics = config.get("signal_diagnostics", False)
    if signal_diagnostics:
        from benchmark.navigation.signal_diagnostics import inspect_native_signals, observe_agent, vehicle_signal_state
    policy = None
    traffic_manager = None
    model_started = time.perf_counter()
    if visual:
        kind = visual.get("kind", "custom")
        # Seed model construction and stochastic inference.
        from benchmark.policies.adapters import configure_inference, load_policy
        report['inference_randomness'] = configure_inference(visual, seed)
        progress('Policy','Reusing weights' if shared_policy else 'Loading weights',method=config['controller'])
        policy = shared_policy or load_policy(visual)
        progress('Policy','Weights ready',method=config['controller'],wall_s=round(time.perf_counter()-model_started,1))
        report["visual_policy"] = {**policy.description, "request_ids": visual_ids,
                                  "background_controller": "CARLA BasicAgent",
                                  "configured_target_speed_applies_to_background_only": True}
        if policy.description.get('reference_vehicle_blueprint'):
            report['visual_policy']['matches_reference_vehicle'] = (
                all(requests[i]['vehicle_blueprint'] ==
                    policy.description['reference_vehicle_blueprint'] for i in visual_ids))
    report["model_initialization_wall_s"] = time.perf_counter()-model_started
    if leader:
        report["external_leader"] = dict(leader, interpretation="External intervention, not an evaluated/background request; metric uses actual speed.")
    if pulse:
        report["speed_pulse"] = {**pulse, "base_target_speed_kmh": requests[pulse["request_id"]]["target_speed_kmh"],
            "clock": "Step-start simulation seconds; controls affect the following frame-end state."}
    pulse_events, pulse_applied = set(), False
    world = original = original_weather = clock = camera = overview_camera = None
    active, pending, frames, collisions = {}, [], [], []
    parked_actors = []
    cleanup_errors, elapsed, next_request, probe_id = [], 0.0, 0, None
    peak_active = peak_evaluated = peak_sensors = 0
    preparation, preparation_collisions = None, []
    first_observation_frame=None
    measurement_loop_started=None
    preparation_wall_s=0.
    renderer_priming_wall_s=0.
    sparse_observers=config.get('sparse_observer_capture',True)
    report['observer_sampling']=dict(sparse_capture=sparse_observers,
        requested_period_s=config.get('observer_image_period_s',5),
        scope='Presentation cameras only; policy sensors retain their native 20 Hz frames. Sparse diagnostic images are matched offline by their actual sensor frame ID.')
    camera_images, collision_events = queue.Queue(), queue.Queue()
    collision_journal=CollisionJournal(output/'collision_callbacks.jsonl')
    def record_collision(event):
        collision_journal.append(event)
        collision_events.put(event)
    overview_images = queue.Queue()
    termination = "exception"
    recorder_started=False
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(float(config.get("native_map_timeout_s", 600)))
    try:
        world = client.get_world()
        if config.get('fresh_server_process', False):
            # A new synchronous client has no episode snapshot until a tick.
            world.tick(float(config.get('native_map_timeout_s',600)))
            if world.get_actors().filter('vehicle.*') or world.get_actors().filter('sensor.*'):
                raise RuntimeError('Expected an empty dedicated CARLA process')
            receipt = json.loads(Path(config['server_start_record']).read_text())
            report['world_initialization'] = dict(**receipt, reloaded=False,
                initialization_api='dedicated server process; no runtime map reload',
                first_snapshot_frame=world.get_snapshot().frame,
                repeatability_note='Initial map loading is asynchronous; observation and physical preparation are synchronous. A fresh process is not a guarantee of identical pixels or paired trajectories.')
        if config.get("reuse_empty_world"):
            world.tick(float(config.get("native_map_timeout_s",600)))
            if world.get_actors().filter("vehicle.*") or world.get_actors().filter("sensor.*"):
                raise RuntimeError("Reused world still contains vehicles/sensors")
            report["world_initialization"] = dict(reloaded=False, reset="empty world, explicit full signal phase reset, fresh per-vehicle policy/controller state",
                first_snapshot_frame=world.get_snapshot().frame,
                repeatability_note="Same-world reset is not claimed equivalent to a physical world reload.")
        if config.get('reload_world_before_run', False):
            if world.get_actors().filter('vehicle.*') or world.get_actors().filter('sensor.*'):
                raise RuntimeError('Refusing to reload a world with existing vehicles or sensors')
            original = world.get_settings()
            initialization_source_map = world.get_map().name
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = dt
            settings.no_rendering_mode = False
            world.apply_settings(settings)
            world = client.reload_world(False)
            # A full custom map may need minutes for its first rendered frame.
            world.tick(float(config.get('native_map_timeout_s', 600)))
            report['world_initialization'] = dict(reloaded=True,
                synchronous_before_reload=True, retained_settings=True,
                initialization_api='reload_world(reset_settings=False)',
                source_map=initialization_source_map,
                first_snapshot_frame=world.get_snapshot().frame)
        geometry_started=time.perf_counter()
        geometry=static_world_data(world, static_cache if static_cache is not None else {})
        carla_map = geometry['native_map']
        progress('CARLA','Native map available',scene=config['scene_id'],wall_s=round(time.perf_counter()-geometry_started,1))
        # A newly connected client's actor inventory may precede its first
        # episode snapshot. Read a real asynchronous frame before exporting
        # native signals or constructing the signal clock.
        if not world.get_settings().synchronous_mode:
            world.wait_for_tick(30.)
        if carla_map.name.removeprefix("/Game/") != info["carla_map_name"].removeprefix("/Game/"):
            raise ValueError("Loaded map differs from map_info.json")
        if client.get_server_version() != info["carla_version"]:
            raise ValueError("CARLA server version differs from the supplied map environment")
        report.update(map=carla_map.name, server_version=client.get_server_version())
        original_weather = world.get_weather()
        weather = world.get_weather()
        requested_weather = config.get("weather", info.get("default_weather", {}))
        for name, value in requested_weather.items():
            setattr(weather, name, float(value))
        if requested_weather:
            world.set_weather(weather)
        report["weather"] = {name: float(getattr(weather, name)) for name in (
            "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
            "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance",
            "wetness", "fog_falloff", "scattering_intensity", "mie_scattering_scale",
            "rayleigh_scattering_scale")}
        if (config.get('fresh_server_process') or config.get('prime_policy_cameras')) and visual:
            from benchmark.runtime.native_startup import prime_cameras
            priming_started=time.perf_counter()
            report['renderer_priming']=prime_cameras(world,carla_map,carla,policy.sensors,
                supplied_routes[route_names[0]]['validated_waypoints'][0]['position_carla_m'])
            renderer_priming_wall_s=time.perf_counter()-priming_started
        from collections import Counter
        routes, navigation = {}, {}
        shared_navigation_plans = {}
        exact_endpoints = (native_endpoint_index(carla_map)
                           if config.get("exact_native_waypoints", False) else None)
        report["waypoint_resolution"] = {
            "mode": "exact_native_lane_samples" if exact_endpoints is not None else "compatible_station",
            "indexed_waypoints": len(exact_endpoints) if exact_endpoints is not None else 0,
        }
        def supplied_waypoint(point):
            return resolve_waypoint(carla_map, point, exact_endpoints=exact_endpoints)
        for name in route_names:
            supplied = supplied_routes[name]
            plan = []
            for item in supplied["validated_waypoints"]:
                waypoint = supplied_waypoint(item)
                plan.append((waypoint, getattr(RoadOption, item["road_option"])))
            navigation[name] = {
                "position_input": "CARLA local world coordinates; built-in GNSS is not used",
                "route_options": dict(Counter(option.name for _, option in plan)),
                "route_waypoint_lane_changes": dict(Counter(str(wp.lane_change) for wp, _ in plan)),
                "same_direction_driving_neighbor_counts": {
                    side: sum(neighbor is not None and neighbor.lane_type == carla.LaneType.Driving
                              and wp.transform.get_forward_vector().dot(neighbor.transform.get_forward_vector()) > 0
                              for wp, neighbor in ((wp, getattr(wp, f"get_{side}_lane")()) for wp, _ in plan))
                    for side in ("left", "right")},
                "scope": "Supplied validated route only. No general legal lane-change capability is inferred from successor connectivity.",
            }
            line = LineString([item["position_carla_m"][:2] for item in supplied["validated_waypoints"]])
            routes[name] = {"plan": plan, "line": line, "start": plan[0][0], "goal": plan[-1][0]}
            context_plan = plan
            if "navigation_waypoints" in supplied:
                context_plan = []
                for point in supplied["navigation_waypoints"]:
                    wp = supplied_waypoint(point)
                    context_plan.append((wp, getattr(RoadOption, point["road_option"])))
                if context_plan[0][0].transform.location.distance(plan[0][0].transform.location) > .01:
                    raise ValueError("Navigation context changes the supplied entrance")
            routes[name]["navigation_plan"] = context_plan
            if supplied.get("removal_gate"):
                routes[name]["removal_gate"] = supplied["removal_gate"]
                routes[name]["removal_gates"] = [supplied_routes[n]["removal_gate"]
                    for n in supplied["evaluation"]["legal_route_names"] if supplied_routes[n].get("removal_gate")]
            if supplied.get('shared_navigation'):
                shared=supplied['shared_navigation'];reference=shared['route_name']
                if reference not in shared_navigation_plans:
                    common=[]
                    source_route=supplied_routes[reference]
                    for point in source_route.get('navigation_waypoints',source_route['validated_waypoints']):
                        wp=supplied_waypoint(point)
                        common.append((wp,getattr(RoadOption,point['road_option'])))
                    shared_navigation_plans[reference]=common
                routes[name]['shared_navigation']={**shared,'plan':shared_navigation_plans[reference]}
                navigation[name]['shared_navigation']=shared
            navigation[name]["navigation_context_waypoints"] = len(context_plan)
            navigation[name]["navigation_context"] = supplied.get("navigation_context")
        if config.get("navigation_task") == "road_corridor_and_exit":
            for name in route_names:
                data = supplied_routes[name]["corridor"]
                for path in data["paths"]:
                    for point in path["validated_waypoints"]:
                        wp = supplied_waypoint(point)
                routes[name]["corridor"] = Corridor(data)
                navigation[name]["task"] = data["task"]
                navigation[name]["allowed_paths"] = [p["id"] for p in data["paths"]]
        report["navigation_by_route"] = navigation
        report["routes"] = route_names
        if len(route_names) == 1:
            report["navigation"] = navigation[route_names[0]]
        # Configurations may share an entrance only when the route starts match.
        starts = {}
        for request in requests:
            location = routes[request["route"]]["start"].transform.location
            prior = starts.setdefault(request["entry"], location)
            if prior.distance(location) > .01:
                raise ValueError("Routes sharing an entry must have the same start position")
        if original is None:
            original = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode = True, dt, False
        world.apply_settings(settings)
        if config.get('record_server',False):
            recording=str((output/'server_recording.log').resolve())
            report['server_recording']=dict(path=recording,start_frame=world.get_snapshot().frame,
                start_world_time_s=world.get_snapshot().timestamp.elapsed_seconds,
                result=client.start_recorder(recording,True))
            recorder_started=True
        applied_settings = world.get_settings()
        if rule.get('kind') == 'traffic_manager':
            traffic_manager = client.get_trafficmanager(args.port+10)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_random_device_seed(seed)
            traffic_manager.set_hybrid_physics_mode(False)
            traffic_manager.set_respawn_dormant_vehicles(False)
            report['traffic_manager'] = dict(port=args.port+10,synchronous=True,seed=seed,
                hybrid_physics=False,respawn=False,path_api='set_path',auto_lane_change=True,
                ignore_lights_percentage=0.,ignore_signs_percentage=0.,ignore_vehicles_percentage=0.)
        native_tm_paths = None
        if traffic_manager is not None and rule.get('native_tm_cache'):
            from benchmark.navigation.tm_cache import NativeTMPath
            cache_path=str(Path(rule['native_tm_cache']).resolve())
            if geometry.get('tm_cache_path')!=cache_path:
                geometry['native_tm_paths']=NativeTMPath(Path(cache_path), carla_map)
                geometry['tm_cache_path']=cache_path
            native_tm_paths = geometry['native_tm_paths']
            progress('CARLA','Native TM route graph available',scene=config['scene_id'])
        report['rule_controller'] = rule
        report['world_settings'] = {key: getattr(applied_settings, key) for key in (
            'synchronous_mode', 'fixed_delta_seconds', 'substepping',
            'max_substep_delta_time', 'max_substeps', 'no_rendering_mode')}
        report['control_submission'] = dict(method='checked_batch_sync', batch_ticks_world=False,
            checked_vehicle_frames=0, mismatched_vehicle_frames=0)
        clock = SignalClock(world, signals, carla, config.get('supply_event'))
        clock.update(0.)
        world.tick()  # Apply frozen signal state before admitting the first vehicle.
        report["world_activity_settings"] = {k:getattr(world.get_settings(),k) for k in ("tile_stream_distance","actor_active_distance")}
        if signal_diagnostics:
            native = inspect_native_signals(world, carla_map)
            expected = {b['signal_id'] for b in signals['bindings']}
            if expected - {s['signal_id'] for s in native['signals']}:
                raise RuntimeError('Native signal export is missing configured signal actors')
            (output / 'native_signals.json').write_text(json.dumps(native,indent=2)+'\n')
        blueprints = world.get_blueprint_library()
        shared_global_planner = None
        if config.get("parked_context", False):
            from benchmark.scenes.background_actors import spawn_background
            context = read(args.scene_dir / "runtime_context.json")
            parked_actors, parked_records = spawn_background(world, context, carla)
            report["parked_context"] = {"actors": parked_records}
        if config.get("initial_vehicles"):
            report["initial_population"] = {"count": len(config["initial_vehicles"]),
                "initialization": "initialization.json records the requested and measured states at the start of observation."}
        if config.get("overview_camera"):
            view = config["overview_camera"]
            bp = blueprints.find("sensor.camera.rgb")
            for key, value in {"image_size_x": view.get("width", 960), "image_size_y": view.get("height", 540),
                               "fov": view.get("fov", 90), "sensor_tick": config.get('observer_image_period_s',5) if sparse_observers else 0}.items():
                bp.set_attribute(key, str(value))
            overview_camera = world.spawn_actor(bp, carla.Transform(carla.Location(*view["position_m"]),
                carla.Rotation(pitch=view["pitch_deg"], yaw=view["yaw_deg"])))
            overview_camera.listen(overview_images.put)
            (output / "overview_rgb").mkdir(exist_ok=True)
            report["overview_camera"] = {**view, "input_to_policy": False,
                "image_period_s": config.get("observer_image_period_s", 5)}
        frame_path = output / ("frames.jsonl.gz" if config.get("compress_frames") else "frames.jsonl")
        frame_open = gzip.open(frame_path, "wt", compresslevel=1) if config.get("compress_frames") else frame_path.open("w")
        with frame_open as frame_log, (output / "events.jsonl").open("w") as events, (output / 'initial_states.jsonl').open('w') as initial_log:
            def emit(kind, request=None, **extra):
                events.write(json.dumps({"event": kind, "time_s": elapsed,
                    "request_id": request["id"] if request else None,
                    "route": request["route"] if request else None, **extra}) + "\n")

            def retire_actors(snapshot, time_s, preparation_time_s=None):
                def record(kind, request, **extra):
                    emit(kind, request, phase='preparation' if preparation_time_s is not None else 'observation',
                         preparation_time_s=preparation_time_s, **extra)
                removed_camera = False
                missing = retire_missing_vehicles(active, snapshot, time_s, record)
                for item in missing:
                    from benchmark.policies.adapters import close_policy
                    close_policy(item.get('policy'))
                    sensors = [item['collision_sensor'], *item.get('policy_sensors', [])]
                    if item['request']['id'] == probe_id and camera is not None:
                        sensors.append(camera)
                        removed_camera = True
                    for sensor in sensors:
                        if snapshot.find(sensor.id) is not None and sensor.is_alive:
                            sensor.stop()
                            if not sensor.destroy():
                                cleanup_errors.append(f'Destruction not acknowledged for orphan sensor {sensor.id}')
                return removed_camera

            def after_preparation_tick(snapshot, relative_time_s):
                nonlocal camera
                if retire_actors(snapshot, 0., relative_time_s):
                    camera = None

            measurement_loop_started=time.perf_counter()
            observation_wall_started=None
            for step in range(math.ceil(config["duration_s"] / dt)):
                elapsed = step * dt
                while next_request < len(arrivals) and arrivals[next_request] <= elapsed + 1e-9:
                    request = requests[next_request]
                    pending.append(request)
                    next_request += 1
                    emit("requested", request, scheduled_arrival_s=request["arrival_s"])
                for request in entry_heads(pending):
                    route = routes[request["route"]]
                    plan, start = route["navigation_plan"], route["start"]
                    # Try each entrance independently in FIFO order, checking spawn clearance.
                    entry_clear = entry_is_clear(start.transform,active,'initial_speed_mps' in request)
                    if entry_clear:
                        request["spawn_attempts"] += 1
                        blueprint = blueprints.find(request["vehicle_blueprint"])
                        blueprint.set_attribute("role_name", "hero" if probe_id is None else f"benchmark_{request['id']}")
                        spawn = start.transform
                        spawn.location.z += 0.5
                        vehicle = world.try_spawn_actor(blueprint, spawn)
                        if vehicle is not None:
                            request["admitted_s"], request["actor_id"] = elapsed, vehicle.id
                            record_participant(request, vehicle)
                            pending.remove(request)
                            item = {"vehicle": vehicle, "request": request, "route": route,
                                    "length_m": 2 * vehicle.bounding_box.extent.x, "off_route_ticks": 0,
                                    'birth_transform': spawn, 'needs_initial_record': True}
                            active[request["id"]] = item
                            if 'vehicle_physics' not in report:
                                physics=vehicle.get_physics_control()
                                report['vehicle_physics']=dict(blueprint=vehicle.type_id,
                                    mass_kg=physics.mass,drag_coefficient=physics.drag_coefficient,
                                    torque_curve_rpm_nm=[[p.x,p.y] for p in physics.torque_curve],
                                    final_ratio=physics.final_ratio,forward_gear_ratios=[g.ratio for g in physics.forward_gears],
                                    wheels=[dict(tire_friction=w.tire_friction,radius_cm=w.radius,
                                        max_brake_torque_nm=w.max_brake_torque) for w in physics.wheels],
                                    scope='CARLA physical parameters for provenance; energy uses the separately declared common BEV.')
                            item["is_visual"] = request["id"] in visual_ids
                            request["controller"] = policy.description["name"] if item["is_visual"] else "CARLA BasicAgent"
                            if item["is_visual"]:
                                item["policy"] = policy.new_vehicle() if hasattr(policy, "new_vehicle") else policy
                                bind = getattr(item["policy"], "bind_request", None)
                                if bind is not None: bind(request["id"])
                                if kind=='lead' and route.get('shared_navigation'):
                                    item['policy'].reset(plan,shared_navigation=route['shared_navigation'])
                                    navigation[request['route']]['sampled_common_navigation']=item['policy'].navigation
                                else:
                                    item["policy"].reset(plan)
                                item["policy_sensors"], item["policy_queues"] = [], {}
                                for specification in policy.sensors:
                                    bp = blueprints.find(specification["type"])
                                    if bp.has_attribute("noise_seed"):
                                        bp.set_attribute("noise_seed", str(request.get("behavior_seed", seed)))
                                    attributes = {"sensor_tick": specification["sensor_tick"],
                                                  **specification.get("blueprint_attributes", {})}
                                    if specification["type"] == "sensor.camera.rgb":
                                        attributes.update(image_size_x=specification["width"],
                                                          image_size_y=specification["height"], fov=specification["fov"])
                                    for name, value in attributes.items():
                                        bp.set_attribute(name, str(value))
                                    pose = carla.Transform(carla.Location(x=specification["x"], y=specification["y"], z=specification["z"]),
                                                           carla.Rotation(pitch=specification["pitch"], yaw=specification["yaw"], roll=specification["roll"]))
                                    sensor = world.spawn_actor(bp, pose, attach_to=vehicle)
                                    stream = queue.Queue()
                                    sensor.listen(stream.put)
                                    item["policy_sensors"].append(sensor)
                                    item["policy_queues"][specification["id"]] = stream
                                item["policy_diagnostic"] = None
                                (output / "policy_rgb").mkdir(exist_ok=True)
                            elif rule.get('kind') == 'traffic_manager' and request['id'] in rule_ids:
                                from benchmark.navigation.tm_navigation import RouteTrafficManagerAgent as TrafficManagerAgent
                                item['agent'] = TrafficManagerAgent(vehicle,traffic_manager,args.port+10,
                                    request['target_speed_kmh'],{**rule, 'native_tm_paths': native_tm_paths,
                                        'tm_milestone_positions_m': supplied_routes[request['route']].get('tm_milestone_positions_m')})
                                item['agent'].set_global_plan(plan)
                                item['is_tm'] = True
                                request['controller'] = 'CARLA Traffic Manager'
                            else:
                                if config.get("precomputed_route_planner"):
                                    if 'background_planner' not in route:
                                        route['background_planner'] = SuppliedRoutePlanner(plan)
                                    background_planner=route['background_planner']
                                else:
                                    if shared_global_planner is None:
                                        shared_global_planner = GlobalRoutePlanner(carla_map, 1.0)
                                    background_planner=shared_global_planner
                                basic_options = {"dt": dt, "sampling_resolution": 1.0,
                                                 **config.get("basic_agent_options", {})}
                                if request['id'] in intervention_ids and leader.get('longitudinal_control_dict'):
                                    # Apply the shared intervention to the external leader.
                                    basic_options['longitudinal_control_dict'] = leader['longitudinal_control_dict']
                                if rule.get('kind') == 'idm' and request['id'] in rule_ids:
                                    basic_options['longitudinal_control_dict'] = rule['longitudinal_control_dict']
                                agent = BasicAgent(vehicle, target_speed=request["target_speed_kmh"], map_inst=carla_map,
                                                   grp_inst=background_planner,
                                                   opt_dict=basic_options)
                                agent.set_global_plan(list(plan), stop_waypoint_creation=True, clean_queue=True)
                                signal_mode = config.get("basic_agent_signal_mode", "official_legacy_trigger")
                                if signal_mode in ("opendrive_route_references", "opendrive_route_references_yellow_stop"):
                                    from benchmark.navigation.signal_control import install_route_signal_detection
                                    yellow_stop = (config["basic_agent_yellow_stop"]
                                                   if signal_mode.endswith("_yellow_stop") else None)
                                    report.setdefault("basic_agent_route_signals_by_route", {})[request["route"]] = install_route_signal_detection(agent, plan, yellow_stop,
                                        candidate_cache=route.setdefault('signal_candidate_cache',{}))
                                    if yellow_stop is not None:
                                        report["basic_agent_yellow_stop"] = yellow_stop
                                elif signal_mode != "official_legacy_trigger":
                                    raise ValueError(f"Unknown BasicAgent signal mode: {signal_mode}")
                                report["basic_agent_signal_mode"] = signal_mode
                                if signal_diagnostics:
                                    observe_agent(agent)
                                if rule.get('kind') == 'idm' and request['id'] in rule_ids:
                                    from benchmark.policies.rule_control import install_idm
                                    report['idm_parameters'] = install_idm(agent,dt,rule)
                                    request['controller'] = 'IDM-based'
                                item["agent"] = agent
                            item["target_speed_kmh"] = request["target_speed_kmh"]
                            sensor = world.spawn_actor(blueprints.find("sensor.other.collision"), carla.Transform(), attach_to=vehicle)
                            sensor.listen(lambda event, key=request["id"], actor_id=vehicle.id:
                                record_collision({"frame": event.frame, "world_time_s":event.timestamp,
                                    "request_id": key, "actor_id": actor_id,
                                    "other_actor_id": event.other_actor.id, "other_actor_type": event.other_actor.type_id,
                                    "sensor_position_m": [event.transform.location.x, event.transform.location.y, event.transform.location.z],
                                    "impulse": [event.normal_impulse.x, event.normal_impulse.y, event.normal_impulse.z]}))
                            item["collision_sensor"] = sensor
                            if probe_id is None:
                                probe_id = request["id"]
                            if camera is None and request['id'] == probe_id and config.get('observer_camera_enabled', True):
                                camera_bp = blueprints.find("sensor.camera.rgb")
                                for key, value in {"image_size_x": str(config.get("observer_width", 640)), "image_size_y": str(config.get("observer_height", 360)), "sensor_tick": str(config.get('observer_image_period_s',5) if sparse_observers else 0)}.items():
                                    camera_bp.set_attribute(key, value)
                                camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2)), attach_to=vehicle)
                                camera.listen(camera_images.put)
                            emit("admitted", request, actor_id=vehicle.id, external_wait_s=elapsed - request["arrival_s"])
                peak_active = max(peak_active, len(active))
                peak_evaluated = max(peak_evaluated, sum(i["is_visual"] for i in active.values()))
                peak_sensors = max(peak_sensors, sum(len(i.get("policy_sensors", []))+1 for i in active.values()))
                if step == 0 and config.get("initial_vehicles"):
                    if config.get('require_complete_initial_population') and any('initial_speed_mps' in r for r in pending):
                        raise ValueError('The declared joint initial population cannot be placed without overlap; revise the scene before evaluation')
                    clock.update(0.0)
                    progress('CARLA','Preparing physical initial state and sensor history',vehicles=len(active),scene=config['scene_id'])
                    preparation_started=time.perf_counter()
                    if config.get('initialization', {}).get('method') != 'native_policy_history':
                        raise ValueError('Initial populations require native_policy_history preparation')
                    from benchmark.runtime.sensor_history import prepare_physical_history
                    preparation=prepare_physical_history(client,world,active,dt,carla,
                        settling_s=config['initialization'].get('settling_s',1.5),
                        rolling_s=config['initialization']['rolling_s'],
                        native_policy=True,
                        after_tick=after_preparation_tick,
                        defer_policy_until_motion=config['initialization'].get('defer_policy_until_motion',False),
                        history_start_s=config['initialization'].get('history_start_s',0.),
                        require_ready_before_motion=config['initialization'].get('require_ready_before_motion',False))
                    preparation_wall_s += time.perf_counter()-preparation_started
                    progress('CARLA','Initial preparation complete',wall_s=round(preparation_wall_s,1),vehicles=len(active))
                    preparation["pending_initial_request_ids"] = [r["id"] for r in pending if "initial_speed_mps" in r]
                    preparation["simulator_removed_initial_request_ids"] = [r["id"] for r in requests if r.get("removed_s") == 0. and r.get("removal_reason") == "simulator_actor_missing"]
                    report["initial_population"]["prepared_count"] = len(preparation["states_at_time_zero"])
                    report["initial_population"]["pending_count_at_time_zero"] = len(preparation["pending_initial_request_ids"])
                    (output / "initialization.json").write_text(json.dumps(preparation, indent=2) + "\n")
                    emit("initial_population_prepared", prepared_count=len(preparation["states_at_time_zero"]),
                         pending_initial_request_ids=preparation["pending_initial_request_ids"],
                         preparation_seconds=preparation["settling_s"] + preparation["rolling_initialization_s"])
                if observation_wall_started is None:
                    observation_wall_started=time.perf_counter()
                for item in active.values():
                    if item.pop('needs_initial_record',False):
                        initial_log.write(json.dumps(initial_state(item,carla_map,elapsed,
                            prepared=bool(item.get('initial_prepared'))))+'\n')
                initial_log.flush()
                if pulse:
                    for phase, boundary in (("started", pulse["start_s"]), ("ended", pulse["end_s"])):
                        if elapsed >= boundary and phase not in pulse_events:
                            target = active.get(pulse["request_id"])
                            emit("speed_pulse_" + phase, target["request"] if target else None,
                                 target_request_id=pulse["request_id"], configured_time_s=boundary,
                                 actor_id=target["vehicle"].id if target else None,
                                 target_present=target is not None,
                                 target_speed_kmh=pulse["target_speed_kmh"] if phase == "started" else requests[pulse["request_id"]]["target_speed_kmh"])
                            pulse_events.add(phase)
                            if phase == "started":
                                pulse_applied = target is not None
                if leader:
                    boundaries=[('start',leader['start_s']),
                        ('minimum_setpoint',leader['start_s']+leader['deceleration_s']),
                        ('recovery_start',leader['start_s']+leader['deceleration_s']+leader['hold_s']),
                        ('end',leader['start_s']+leader['deceleration_s']+leader['hold_s']+leader['recovery_s'])]
                    for label,boundary in boundaries:
                        if abs(elapsed-boundary)<dt/2:
                            emit('external_leader_'+label,active.get(leader['request_id'],{}).get('request'),
                                configured_time_s=boundary,target_request_id=leader['request_id'],
                                setpoint_mps=target_speed_mps(leader,elapsed))
                decision_started=time.perf_counter()
                commands = []
                clock.update(elapsed)
                event = config.get('supply_event')
                if event:
                    for label, boundary in [('started', event['start_s']), ('ended', event['end_s'])]:
                        if abs(elapsed-boundary) < dt/2:
                            emit('supply_event_'+label, source_signal_id=event['source_signal_id'],
                                 configured_time_s=boundary, affected_states={b['signal_id']:clock.last[b['signal_id']]
                                 for b in signals['bindings'] if b['source_signal_id']==event['source_signal_id']})
                for item in active.values():
                    vehicle_policy = item.get("policy")
                    target_speed = pulse_target_speed(pulse, item["request"]["target_speed_kmh"], item["request"]["id"], elapsed)
                    if leader and item["request"]["id"] == leader["request_id"]:
                        target_speed = 3.6 * target_speed_mps(leader, elapsed)
                    request = item["request"]
                    if "initial_speed_mps" in request and "initial_velocity_applied_s" not in request and elapsed-request["admitted_s"] >= 1:
                        forward = item["vehicle"].get_transform().get_forward_vector()
                        commands.append(carla.command.ApplyTargetVelocity(item['vehicle'].id,
                            forward * float(request['initial_speed_mps'])))
                        request["initial_velocity_applied_s"] = elapsed
                        emit("initial_velocity_applied", request, initial_speed_mps=request["initial_speed_mps"])
                    if target_speed != item["target_speed_kmh"]:
                        if item["is_visual"]:
                            raise ValueError("BasicAgent target-speed pulses are not an intervention interface for visual policies")
                        item["agent"].set_target_speed(target_speed)
                        item["target_speed_kmh"] = target_speed
                    if not item.get("initial_prepared") and elapsed - item["request"]["admitted_s"] < 1:
                        control = carla.VehicleControl(brake=1)
                    elif item.get('is_tm'):
                        count=len(item['agent'].navigation_updates)
                        item['agent'].run_step()
                        if len(item['agent'].navigation_updates)>count:
                            emit('tm_route_tail_reimport',request,**item['agent'].navigation_updates[-1])
                        request['tm_navigation_updates']=item['agent'].navigation_updates
                        item['control'] = None
                        continue
                    elif item["is_visual"]:
                        transform, velocity = item["vehicle"].get_transform(), item["vehicle"].get_velocity()
                        forward = transform.get_forward_vector()
                        gps_position = transform.transform(carla.Location(x=getattr(vehicle_policy, "localization_sensor_x_m", -1.4)))
                        observe_admitted_history(item,item["policy_images"],elapsed,
                            exclude_settling=config.get("exclude_spawn_settling_from_history",False))
                        command, diagnostic = vehicle_policy.step(item["policy_images"], velocity.dot(forward),
                                                          [transform.location.x, transform.location.y, transform.location.z], transform.rotation.yaw,
                                                          gps_position_xyz=[gps_position.x, gps_position.y, gps_position.z])
                        diagnostic["input_frame"] = item["policy_input_frame"]
                        diagnostic["control_time_s"] = elapsed
                        item["policy_diagnostic"] = diagnostic
                        control = carla.VehicleControl(**command)
                    else:
                        if config.get('synchronize_background_initial_control') and not item.get('background_control_synchronized'):
                            controller = item['agent'].get_local_planner()._vehicle_controller
                            previous = repr(controller.past_steering)
                            controller.past_steering = item['vehicle'].get_control().steer
                            item['background_control_synchronized'] = True
                            emit('background_control_synchronized', request,
                                 previous_steering_repr=previous, actual_steering=controller.past_steering)
                        control = item["agent"].run_step()
                    commands.append(carla.command.ApplyVehicleControl(item['vehicle'].id, control))
                    item["control"] = control
                submit_commands(client, commands)
                tick_started=time.perf_counter()
                frame = world.tick()
                sensors_started=time.perf_counter()
                if first_observation_frame is None:
                    first_observation_frame=frame
                elapsed = (step + 1) * dt
                # Kill-Z or world bounds can remove an actor during this tick.
                # Skip its RPCs and sensors; retain the unfinished request.
                snapshot = world.get_snapshot()
                if retire_actors(snapshot, elapsed):
                    camera = None
                dormant=[i["request"]["id"] for i in active.values() if i["vehicle"].is_dormant]
                if dormant:
                    report["dormant_request_ids"] = dormant
                    raise RuntimeError("Traffic actors became dormant: "+str(dormant))
                sensor_wait_s = sensor_save_s = 0.
                sensor_wait_by_type = {}
                for item in active.values():
                    if item["is_visual"]:
                        images = {}
                        for sensor_id, stream in item["policy_queues"].items():
                            waiting_started=time.perf_counter()
                            try:
                                value = stream.get(timeout=60)
                                while value.frame < frame:
                                    value = stream.get(timeout=60)
                            except queue.Empty as error:
                                raise RuntimeError(f"Visual policy {sensor_id} did not deliver world frame {frame} at {elapsed}s") from error
                            if value.frame != frame:
                                raise RuntimeError(f"Visual policy {sensor_id} image frame does not match world")
                            waited=time.perf_counter()-waiting_started
                            sensor_wait_s += waited
                            sensor_wait_by_type[sensor_id]=sensor_wait_by_type.get(sensor_id,0.)+waited
                            images[sensor_id] = value
                            save_started=time.perf_counter()
                            capture = (config.get('policy_image_period_s', 5) > 0
                                and (config.get('policy_image_request_ids') is None or item['request']['id'] in config['policy_image_request_ids'])
                                and step % max(1, round(config.get('policy_image_period_s', 5) / dt)) == 0)
                            if (isinstance(value, carla.Image) and capture
                                    and (config.get('policy_image_sensor_ids') is None
                                         or sensor_id in config['policy_image_sensor_ids'])):
                                save_policy_image(value, output / "policy_rgb" / f"request_{item['request']['id']}_{sensor_id}_{frame:08d}.png")
                            elif isinstance(value, (carla.LidarMeasurement, carla.RadarMeasurement)) and capture and config.get('record_policy_points', True):
                                (output / "policy_points").mkdir(exist_ok=True)
                                np.savez_compressed(output / "policy_points" / f"request_{item['request']['id']}_{sensor_id}_{frame:08d}.npz",
                                    raw_native=np.frombuffer(value.raw_data, np.float32).reshape(-1, 4))
                            sensor_save_s += time.perf_counter()-save_started
                        item["policy_images"], item["policy_input_frame"] = images, frame
                        observe_admitted_history(item,images,elapsed,
                            exclude_settling=config.get("exclude_spawn_settling_from_history",False))
                camera_frame = None
                if sparse_observers:
                    camera_frame=drain_observer(camera_images,output/'rgb',config,first_observation_frame)
                elif camera is not None:
                    image = camera_images.get(timeout=60)
                    while image.frame < frame:
                        image = camera_images.get(timeout=60)
                    if image.frame != frame:
                        raise RuntimeError("RGB does not match the synchronous world frame")
                    camera_frame = image.frame
                    if step % max(1, round(config.get("observer_image_period_s", 5) / dt)) == 0:
                        save_observer_image(image, output / "rgb", config)
                overview_frame = None
                if sparse_observers:
                    overview_frame=drain_observer(overview_images,output/'overview_rgb',config,first_observation_frame)
                elif overview_camera is not None:
                    overview_image = overview_images.get(timeout=60)
                    while overview_image.frame < frame:
                        overview_image = overview_images.get(timeout=60)
                    if overview_image.frame != frame:
                        raise RuntimeError("Overview RGB does not match the synchronous world frame")
                    overview_frame = overview_image.frame
                    if step % max(1, round(config.get("observer_image_period_s", 5) / dt)) == 0:
                        save_observer_image(overview_image, output / "overview_rgb", config)
                snapshot_started=time.perf_counter()
                row = {"frame": frame, "time_s": elapsed, "world_time_s": world.get_snapshot().timestamp.elapsed_seconds, "control_time_s": step * dt,
                       'wall_timing_s':dict(decisions=tick_started-decision_started,
                           world_tick=sensors_started-tick_started,
                           input_wait_and_export=snapshot_started-sensors_started,
                           sensor_queue_wait=sensor_wait_s, sensor_export=sensor_save_s,
                           sensor_wait_by_id=sensor_wait_by_type),
                       "dt_s": dt, "camera_frame": camera_frame, "overview_camera_frame": overview_frame,
                       "pending_requests": [r["id"] for r in pending], "vehicles": [],
                       "lights": {key: str(a.get_state()) for key, a in clock.actors.items()}}
                probe_snapshot = snapshot.find(active[probe_id]["vehicle"].id) if probe_id in active else None
                probe = probe_snapshot.get_transform().location if probe_snapshot is not None else None
                finished = []
                for key, item in active.items():
                    vehicle = item["vehicle"]
                    line, goal = item["route"]["line"], item["route"]["goal"]
                    actor_snapshot=snapshot.find(vehicle.id)
                    transform, velocity, control = actor_snapshot.get_transform(), actor_snapshot.get_velocity(), item["control"]
                    if item.get('is_tm') and item['agent'].enabled:
                        # TM applies controls during the synchronous tick; record its readback.
                        from benchmark.runtime.control_submission import control_state
                        control = vehicle.get_control()
                        applied_control = control_state(control)
                        control_matches = None
                        report['control_submission']['tm_observed_vehicle_frames'] = report['control_submission'].get('tm_observed_vehicle_frames',0)+1
                    else:
                        applied_control, control_matches = control_echo(vehicle, control)
                        report['control_submission']['checked_vehicle_frames'] += 1
                        report['control_submission']['mismatched_vehicle_frames'] += int(not control_matches)
                    acceleration = actor_snapshot.get_acceleration()
                    location = transform.location
                    forward = transform.get_forward_vector()
                    waypoint = carla_map.get_waypoint(location)
                    corners = vehicle.bounding_box.get_world_vertices(transform)
                    offset = line.distance(Point(location.x, location.y))
                    item["off_route_ticks"] = item["off_route_ticks"] + 1 if offset > 3 else 0
                    xmin, ymin, xmax, ymax = scene["core_bbox_carla_xy_m"]
                    row["vehicles"].append({"request_id": key, "actor_id": vehicle.id,
                        "route": item["request"]["route"], "entry": item["request"]["entry"],
                        "state_frame":snapshot.frame,
                        "deployment_role": item["request"].get("deployment_role"),
                        "scene_id": item["request"].get("scene_id",config["scene_id"]),
                        "is_dormant": vehicle.is_dormant,
                        "is_active": vehicle.is_active,
                        "position_m": [location.x, location.y, location.z], "speed_mps": math.hypot(velocity.x, velocity.y),
                        "velocity_mps": [velocity.x, velocity.y, velocity.z], "route_station_m": line.project(Point(location.x, location.y)),
                        "acceleration_mps2": [acceleration.x, acceleration.y, acceleration.z],
                        "forward_xy": [forward.x, forward.y],
                        "bounding_box_world_vertices_m": [[p.x, p.y, p.z] for p in corners],
                        **road_fields(waypoint,velocity,forward),
                        "road_surface_offset_m": location.z - waypoint.transform.location.z,
                        "seconds_since_admission": elapsed - item["request"]["admitted_s"],
                        "initial_prepared": item.get("initial_prepared", False),
                        "target_speed_kmh": None if item["is_visual"] else item["target_speed_kmh"],
                        "controller": item["request"]["controller"],
                        "visual_policy": item.get("policy_diagnostic"),
                        "rule_diagnostic": getattr(item.get("agent"),"idm_diagnostic",None),
                        **({"signal_diagnostic": vehicle_signal_state(vehicle, item.get("agent"))} if signal_diagnostics else {}),
                        "route_offset_m": offset, "inside_core": xmin <= location.x <= xmax and ymin <= location.y <= ymax,
                        "distance_from_probe_m": location.distance(probe) if probe is not None else None,
                        "control": {"throttle": control.throttle, "brake": control.brake, "steer": control.steer},
                        "applied_control": applied_control, "control_roundtrip_match": control_matches})
                    if config.get('record_startup_diagnostics'):
                        row['vehicles'][-1]['actual_gear'] = vehicle.get_control().gear
                        row['vehicles'][-1]['history_capture_first_frame'] = item.get('history_capture_first_frame')
                    item['last_observed_state'] = dict(time_s=elapsed,
                        position_m=[location.x, location.y, location.z], route_offset_m=offset)
                    corridor = item["route"].get("corridor")
                    reached = location.distance(goal.transform.location) < 2 and line.project(Point(location.x, location.y)) > line.length - 1.5
                    if corridor is not None and item["request"].get("completed_s") is None:
                        if "corridor_tracker" not in item:
                            item["corridor_tracker"] = CorridorTracker(corridor)
                        check = item["corridor_tracker"].update(row["vehicles"][-1], dt)
                        row["vehicles"][-1]["corridor"] = check
                        reached = check["forward_exit_crossing"] is not None
                        if config.get("require_legal_completion") and reached and not check["exit_on_compatible_path"]:
                            reached = False
                            item["request"].setdefault("failure_reason", "terminal_crossed_after_corridor_departure")
                        if reached:
                            item["request"]["exit_path"] = check["forward_exit_crossing"]
                        if check["sustained_departure"] and "first_corridor_departure_s" not in item["request"]:
                            item["request"]["first_corridor_departure_s"] = elapsed
                            item["request"].setdefault("failure_reason", "sustained_corridor_departure")
                            emit("left_legal_corridor", item["request"], detail=check)
                            if config.get("stop_on_incident", True):
                                termination = "left_legal_corridor"
                    if item["request"].get("completed_s") is None and item["off_route_ticks"] * dt >= 1 and "first_off_route_s" not in item["request"]:
                        item["request"]["first_off_route_s"] = elapsed
                        emit("left_supplied_route", item["request"], route_offset_m=offset,
                             diagnostic_only=corridor is not None)
                        if corridor is None and config.get("stop_on_incident", True):
                            termination = "left_supplied_route"
                    if reached and item["request"].get("completed_s") is None:
                        if item["request"].get("retain_at_terminal",config.get('retain_platoon_at_terminal')):
                            if 'observation_terminal_crossed_s' not in item['request']:
                                item['request']['observation_terminal_crossed_s']=elapsed
                                emit('platoon_observation_terminal_crossed',item['request'],
                                     retained_under_own_policy=True)
                            continue
                        if config.get("require_legal_completion") and corridor is None and (
                            "first_off_route_s" in item["request"] or forward.dot(goal.transform.get_forward_vector()) <= 0):
                            item["request"].setdefault("failure_reason", "invalid_route_terminal")
                            continue
                        item["request"]["completed_s"] = elapsed
                        if "removal_gate" not in item["route"]:
                            finished.append(key)
                        emit("completed", item["request"])
                    gate=item["route"].get("removal_gate")
                    if gate and item["request"].get("completed_s") is not None:
                        if beyond_receiving_gate([location.x,location.y],[forward.x,forward.y],
                                [[p.x,p.y] for p in corners],item['route']['removal_gates']):
                            finished.append(key)
                            item["request"]["removed_s"]=elapsed
                            emit("receiving_boundary_exit",item["request"],measured_exit_s=item["request"]["completed_s"])
                row['wall_timing_s']['state_recording']=time.perf_counter()-snapshot_started
                frame_log.write(json.dumps(row) + "\n")
                frames.append(row)
                while not collision_events.empty():
                    event = collision_events.get()
                    if preparation and event["frame"] <= preparation["last_preparation_frame"]:
                        preparation_collisions.append(event)
                        continue
                    event["time_s"] = elapsed - (frame - event["frame"]) * dt
                    collisions.append(event)
                    emit("collision", **event)
                for key in finished:
                    if key == probe_id and camera is not None:
                        camera.stop()
                        camera.destroy()
                        camera = None
                    item = active.pop(key)
                    from benchmark.policies.adapters import close_policy
                    close_policy(item.get("policy"))
                    if item.get('is_tm'):
                        item['agent'].disable()
                    for sensor in item.get("policy_sensors", []):
                        sensor.stop()
                        if not sensor.destroy():
                            cleanup_errors.append(f"Destruction not acknowledged for policy sensor {sensor.id}")
                    item["collision_sensor"].stop()
                    destroyed = [item["collision_sensor"].destroy(), item["vehicle"].destroy()]
                    if not all(destroyed):
                        cleanup_errors.append(f"Destruction not acknowledged for completed request {key}")
                if step % 100 == 0 or time.perf_counter()-last_progress_wall>=60:
                    last_progress_wall=time.perf_counter()
                    loop_wall=last_progress_wall-observation_wall_started
                    remaining=(config['duration_s']-elapsed)*loop_wall/elapsed if elapsed>=1 else None
                    progress('Evaluate','Simulation progress',method=config['controller'],scene=config['scene_id'],deployment=config.get('deployment'),sim_s=f"{elapsed:.2f}/{config['duration_s']}",active=len(active),waiting=len(pending),completed=sum(r['completed_s'] is not None for r in requests),wall_s=round(last_progress_wall-wall_started),remaining_wall_s=round(remaining) if remaining is not None else None)
                    frame_log.flush()
                    events.flush()
                    (output / "progress.json").write_text(json.dumps(dict(time_s=elapsed,
                        requested=next_request, active=len(active), waiting=len(pending),
                        finished=sum(r["completed_s"] is not None for r in requests),
                        evaluated_active=sum(i["is_visual"] for i in active.values()),
                        wall_s=time.perf_counter()-wall_started)) + "\n")
                # Continue serving requests after individual exits or departures.
                for scene_violation in scene_domain.new_vehicle_violations(
                        [v for v in row['vehicles'] if v['request_id'] not in finished]):
                    report.setdefault('scene_violations', []).append(dict(time_s=elapsed, frame=frame, **scene_violation))
                    emit('scene_domain_departure', **scene_violation)
                incident = incident_stop_reason(collisions, termination, config.get("stop_on_incident", True))
                if incident:
                    termination = incident
                    break
            else:
                termination = "observation_complete"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        import traceback
        (output / 'exception.txt').write_text(traceback.format_exc())
    finally:
        measurement_loop_wall_s=(time.perf_counter()-measurement_loop_started-preparation_wall_s) if measurement_loop_started is not None else 0.
        # Collision callbacks use a separate stream from world snapshots.
        # Allow the final frame's local callbacks to arrive without advancing
        # the simulated observation window before unsubscribing the sensors.
        (output/'requests.json').write_text(json.dumps(requests,indent=2)+'\n')
        delivery=collision_journal.settle()
        # Close contact subscriptions and the journal before actor cleanup.
        for item in active.values():
            sensor=item.get('collision_sensor')
            if sensor is not None:
                try:sensor.stop()
                except RuntimeError as error:cleanup_errors.append(str(error))
        delivery['after_stop']=collision_journal.settle(quiet_s=.1,maximum_s=.5)
        if frames:
            collisions,outside=collision_journal.select(first_observation_frame,frames[-1]['frame'],dt,
                {r['actor_id']:r['id'] for r in requests if r.get('actor_id') is not None})
            (output/'collisions.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in collisions))
            (output/'collision_window.json').write_text(json.dumps(dict(first_frame=first_observation_frame,
                last_frame=frames[-1]['frame'],dt_s=dt,delivery=delivery,observed_callbacks=len(collisions),
                outside_observation_callbacks=len(outside),physics_advanced_during_drain=False),indent=2)+'\n')
        collision_journal.close()
        if recorder_started:
            client.stop_recorder()
            report['server_recording']['end_frame']=world.get_snapshot().frame
            (output/'server_recording.json').write_text(json.dumps(report['server_recording'],indent=2)+'\n')
            (output/'server_recording_collisions.txt').write_text(client.show_recorder_collisions(recording,'a','a'))
        # Save the observation before native controller and actor teardown.
        report.update(termination=termination,elapsed_s=elapsed,ticks=len(frames),
            execution_completed=False,diagnostics_status='before_actor_cleanup',
            collision_event_stream_complete=delivery['quiet_observed'] and delivery['after_stop']['quiet_observed'],
            checks={'observation_completed':termination=='observation_complete'},
            resources={'wall_s':time.perf_counter()-wall_started})
        (output/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for item in active.values():
            from benchmark.policies.adapters import close_policy
            close_policy(item.get('policy'))
            if item.get('is_tm'):
                item['agent'].disable()
        if sparse_observers and first_observation_frame is not None:
            drain_observer(camera_images,output/'rgb',config,first_observation_frame)
            if overview_camera is not None:
                drain_observer(overview_images,output/'overview_rgb',config,first_observation_frame)
        for actor in parked_actors + [v for v in (camera, overview_camera) if v is not None] + [a for item in active.values() for a in (item.get("collision_sensor"), *item.get("policy_sensors", []), item["vehicle"]) if a is not None]:
            try:
                if world.get_snapshot().find(actor.id) is None or not actor.is_alive:
                    continue
                if actor.type_id.startswith("sensor.") and actor.type_id!='sensor.other.collision':
                    actor.stop()
                if not actor.destroy():
                    cleanup_errors.append(f"Destruction not acknowledged for actor {actor.id}")
            except RuntimeError as error:
                cleanup_errors.append(str(error))
        while not collision_events.empty():
            event = collision_events.get()
            if preparation and event["frame"] <= preparation["last_preparation_frame"]:
                preparation_collisions.append(event)
                continue
            event["time_s"] = elapsed - (frames[-1]["frame"] - event["frame"]) * dt
        if clock is not None:
            try:
                clock.close()
            except RuntimeError as error:
                cleanup_errors.append("restore signals: " + str(error))
        if original_weather is not None:
            try:
                world.set_weather(original_weather)
            except Exception as error:
                cleanup_errors.append("weather: " + str(error))
        if original is not None:
            try:
                if traffic_manager is not None:
                    traffic_manager.set_synchronous_mode(False)
                world.apply_settings(original)
            except RuntimeError as error:
                cleanup_errors.append("restore world settings: " + str(error))
        if preparation is not None:
            preparation["collision_events"] = preparation_collisions
            (output / "initialization.json").write_text(json.dumps(preparation, indent=2) + "\n")
        # Save the observation and callback tail before auxiliary diagnostics.
        report.update(termination=termination, elapsed_s=elapsed, ticks=len(frames),
            cleanup_errors=cleanup_errors, scheduled_duration_s=config["duration_s"],
            diagnostics_status="pending", collision_event_stream_complete=delivery['quiet_observed'] and delivery['after_stop']['quiet_observed'],
            checks={"observation_completed": termination == "observation_complete"},
            resources={"wall_s": time.perf_counter()-wall_started})
        (output / "requests.json").write_text(json.dumps(requests, indent=2, allow_nan=False) + "\n")
        (output / "collisions.jsonl").write_text("".join(json.dumps(event) + "\n" for event in collisions))
        (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        metrics = summarize(requests, frames, collisions, elapsed)
        from benchmark.metrics.route_outcomes import evaluate as evaluate_route_outcomes
        route_outcomes = evaluate_route_outcomes(requests, frames, {name: supplied_routes[name]["validated_waypoints"] for name in route_names}, elapsed, task=config.get("navigation_task"))
        metrics["endpoint_completion_definition"] = "Physical endpoint exits; see route_outcomes.json before interpreting them as successful route execution."
        checks = {"observation_completed": termination == "observation_complete", "request_accounting_balances": metrics["request_accounting_balances"],
                  "rgb_recorded": (any(row["camera_frame"] is not None for row in frames)
                                   if config.get('observer_camera_enabled', True) else True), "cleanup_completed": not cleanup_errors}
        checks['submitted_controls_observed_after_tick'] = (
            report.get('control_submission', {}).get('mismatched_vehicle_frames') == 0)
        if preparation and preparation.get('control_submission'):
            checks['preparation_controls_observed_after_tick'] = (
                preparation['control_submission']['mismatched_vehicle_frames'] == 0)
        report['observer_camera_enabled'] = config.get('observer_camera_enabled', True)
        report["outcomes"] = {"some_requests_completed": metrics["completed"] > 0,
                              "some_endpoint_exits_pass_route_check": route_outcomes["passed_bounded_route_check"] > 0,
                              "endpoint_exits_failing_route_check": route_outcomes["endpoint_exits_failing_route_check"],
                              "entry_wait_recorded": metrics["external_entry_wait_vehicle_seconds"] > 0,
                              "no_collisions": not collisions,
                              "far_vehicle_continuous_motion_observed": metrics["longest_continuous_far_motion_s"] >= 1}
        if pulse:
            checks["speed_pulse_started_with_target_present_and_end_recorded"] = pulse_applied and pulse_events == {"started", "ended"}
        report.update(termination=termination, elapsed_s=elapsed, ticks=len(frames), cleanup_errors=cleanup_errors,
                      checks=checks, passed=all(checks.values()), vehicle_blueprint=config["vehicle_blueprint"])
        report["execution_completed"] = (termination == "observation_complete" and not cleanup_errors
            and checks['submitted_controls_observed_after_tick']
            and checks.get('preparation_controls_observed_after_tick', True))
        report["scheduled_duration_s"] = config["duration_s"]
        report["incidents_stop_observation"] = config.get("stop_on_incident", True)
        report["off_route_requests"] = [r["id"] for r in requests if "first_off_route_s" in r]
        report['dormant_vehicle_frames'] = sum(v.get('is_dormant',False) for row in frames for v in row['vehicles'])
        report['simulator_removed_requests'] = [r['id'] for r in requests
            if r.get('removal_reason') == 'simulator_actor_missing']
        report["resources"] = dict(wall_s=time.perf_counter()-wall_started,
            peak_active_vehicles=peak_active, peak_evaluated_vehicles=peak_evaluated,
            peak_vehicle_sensors=peak_sensors,
            measurement_loop_wall_s=measurement_loop_wall_s,preparation_wall_s=preparation_wall_s,
            renderer_priming_wall_s=renderer_priming_wall_s,
            evaluated_vehicle_seconds=sum(row["dt_s"] for row in frames for v in row["vehicles"] if v.get("deployment_role")=="evaluated"),
            observation_loop_wall_s=sum(sum(row["wall_timing_s"].get(k,0.) for k in ("decisions","world_tick","input_wait_and_export","state_recording")) for row in frames),
            simulated_seconds_per_wall_second=elapsed/(time.perf_counter()-wall_started))
        if report.get('inference_randomness', {}).get('framework') == 'torch':
            import torch
            report["resources"]["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["legal_completion_required"] = config.get("require_legal_completion", False)
        report["diagnostics_status"] = "complete"
        for name, value in (("run.json", report), ("requests.json", requests), ("run_diagnostics.json", metrics),
                            ("route_outcomes.json", route_outcomes)):
            (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        (output / "collisions.jsonl").write_text("".join(json.dumps(event) + "\n" for event in collisions))
        print(json.dumps({"run": report, "metrics": metrics}, indent=2))
    if not report["execution_completed"]:
        raise SystemExit(1)
    return policy


if __name__ == "__main__":
    main()
