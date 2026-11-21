"""Adapt LEAD to benchmark routes and traffic requests."""
import copy
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
from benchmark.navigation.shared_navigation import sample_route_indices, suffix_indices


def add_source_paths(root):
    root = Path(root).resolve()
    for folder in ('', '3rd_party/leaderboard', '3rd_party/scenario_runner'):
        sys.path.insert(0, str(root / folder))


def local_position_to_gps(position):
    """Encode local metres in a zero-reference Mercator pair."""
    radius = 6378137.
    x, y, z = position
    return np.array([math.degrees(2 * math.atan(math.exp(-y / radius))) - 90,
                     math.degrees(x / radius), z], dtype=np.float64)


def decode_radar(raw_data):
    """Decode CARLA radar detections for LEAD."""
    # CARLA stores each detection as [velocity, azimuth, altitude, depth].
    return np.frombuffer(raw_data, dtype=np.float32).reshape(-1, 4)[:, ::-1].copy()


class LEADPolicy:
    inference_hz = 20
    localization_sensor_x_m = 0.
    observe_during_warmup = True

    def __init__(self, source_root, model_dir, device='cuda:0', navigation_sample_factor_m=200.,
                 checkpoint=None, controller_variant='official', longitudinal_mode=None):
        import torch
        import timm
        self.navigation_sample_factor_m = float(navigation_sample_factor_m)
        self.source_root = Path(source_root).resolve()
        self.model_dir = Path(model_dir).resolve()
        selected = ([self.model_dir / checkpoint] if checkpoint
                    else sorted(self.model_dir.glob('model*.pth')))
        if not selected or any(not path.is_file() for path in selected):
            raise FileNotFoundError('Requested LEAD checkpoint is unavailable')
        add_source_paths(self.source_root)
        # Upstream's import configures backend globals. Preserve the evaluator's
        # explicit inference settings rather than inheriting those import effects.
        flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                 torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            from lead.inference.sensor_agent import SensorAgent
        finally:
            (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) = flags
        from lead.common.base_agent import BaseAgent
        from lead.common import ransac
        from lead.common.sensor_setup import av_sensor_setup
        from lead.expert.config_expert import ExpertConfig
        from lead.training.config_training import TrainingConfig
        from lead.inference.config_closed_loop import ClosedLoopConfig
        from lead.inference.closed_loop_inference import ClosedLoopInference

        class SensorProcessor(SensorAgent):
            def __init__(self):
                pass  # Sensor-only state is prepared by reset(), without an upstream world.
            setup = BaseAgent.setup
            def tick(self, input_data):
                with patch.object(ransac, 'fit_ground_parallel', ransac.fit_ground):
                    return super().tick(input_data)

        self.processor_class = SensorProcessor
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.training = TrainingConfig(json.loads((self.model_dir / 'config.json').read_text()),
                                       raise_error_on_missing_key=False)
        self.closed = ClosedLoopConfig(raise_error_on_missing_key=False)
        if longitudinal_mode is not None:
            if longitudinal_mode not in ('target_speed', 'waypoint'):
                raise ValueError('Use a native LEAD longitudinal modality')
            self.closed.throttle_modality = longitudinal_mode
            self.closed.brake_modality = longitudinal_mode
        self.expert = ExpertConfig()
        self.closed.strict_weight_load = True
        # Full strict checkpoints overwrite every backbone parameter. Avoid a
        # redundant ImageNet download during model construction only.
        create_model = timm.create_model
        def construct(*args, **kwargs):
            kwargs['pretrained'] = False
            return create_model(*args, **kwargs)
        with patch.object(timm, 'create_model', side_effect=construct):
            self.inference = ClosedLoopInference(self.training, self.closed, self.expert,
                                                str(self.model_dir), self.device, checkpoint or 'model')
        if len(self.inference.nets) != len(selected):
            raise ValueError('Loaded checkpoint count differs from the requested model')
        for net in self.inference.nets:
            net.requires_grad_(False)
        if controller_variant == 'bounded_lookahead':
            from benchmark.policies.lead_controller import install_bounded_lookahead
            install_bounded_lookahead(self.inference)
        elif controller_variant != 'official':
            raise ValueError('Unknown LEAD controller variant')
        self.initial_controllers = copy.deepcopy({k: v for k, v in vars(self.inference).items()
                                                   if k.endswith('_controller')})
        rig = av_sensor_setup(self.training, 0., 0., True, False, True, radar=True)
        self.sensors = []
        for spec in rig:
            if spec['type'] in ('sensor.other.gnss', 'sensor.speedometer'):
                continue
            spec = {**spec, 'sensor_tick': 0}
            if spec['type'] == 'sensor.lidar.ray_cast':
                spec['blueprint_attributes'] = dict(range=85, rotation_frequency=10, channels=64,
                    upper_fov=10, lower_fov=-30, points_per_second=600000,
                    atmosphere_attenuation_rate=.004, dropoff_general_rate=.45,
                    dropoff_intensity_limit=.8, dropoff_zero_intensity=.4)
            elif spec['type'] == 'sensor.other.radar':
                spec['blueprint_attributes'] = dict(horizontal_fov=spec['horizontal_fov'],
                    vertical_fov=spec['vertical_fov'], points_per_second=1500, range=100)
            elif spec['type'] == 'sensor.other.imu':
                spec['blueprint_attributes'] = dict(noise_accel_stddev_x=.001, noise_accel_stddev_y=.001,
                    noise_accel_stddev_z=.015, noise_gyro_stddev_x=.001,
                    noise_gyro_stddev_y=.001, noise_gyro_stddev_z=.001)
            self.sensors.append(spec)
        self.description = {
            'name': 'LEAD-TFv6-' + self.training.image_architecture,
            'modalities': 'RGB + two LiDARs + four radars + IMU + noiseless local localization',
            'source_root': str(self.source_root), 'model_dir': str(self.model_dir),
            'checkpoints': [str(p) for p in selected],
            'ensemble_size': len(self.inference.nets), 'checkpoint_loading': 'strict',
            'parameters_per_model': sum(p.numel() for p in self.inference.nets[0].parameters()),
            'source': 'https://github.com/kesai-labs/lead/tree/cvpr2026',
            'weights_source': 'https://huggingface.co/ln2697/tfv6',
            'reference_vehicle_blueprint': 'vehicle.lincoln.mkz_2020',
            'vehicle_reference_source': 'LEAD 3rd_party/leaderboard/leaderboard/scenarios/route_scenario.py::_spawn_ego_vehicle',
            'reference_carla_version': '0.9.15',
            'benchmark_initial_brake_policy_frames': max(1, self.training.inital_frames_delay),
            'reference_initial_brake_policy_frames': self.training.inital_frames_delay,
            'initial_brake_scope': 'Supplied SensorAgent first tick initializes history and returns brake without advancing network/PID; later ticks use the checkpoint initial-frame delay. Physical population settling is handled by the common runner.',
            'localization': 'Body-origin world position encoded as a zero-reference Mercator pair; not geographic GNSS',
            'navigation': 'Ego2Flow route sampling preserves LEAD command changes, lane changes and endpoints.',
            'navigation_sample_factor_m': self.navigation_sample_factor_m,
            'temporal_input': 'Native two-LiDAR streams, ground removal and motion-compensated history; existing LEAD UKF/smoother retained',
            'ground_removal': 'lead.common.ransac.fit_ground',
            'lidar_history_queue_steps': self.expert.lidar_stack_size,
            'training_used_lidar_steps_upper_bound': self.training.training_used_lidar_steps,
            'control': 'Published route lateral controller and native '+self.closed.throttle_modality+' longitudinal controller; no PRD or actor/light truth',
            'jpeg_quality': self.closed.jpeg_quality, 'sensors': self.sensors,
            'torch_version': torch.__version__, 'device': str(self.device),
            'closed_loop_settings': {key: getattr(self.closed, key)
                for base in reversed(type(self.closed).__mro__) for key, default in vars(base).items()
                if not key.startswith('_') and isinstance(default, (str, bool, int, float, type(None)))},
            'state_sharing': 'Frozen eval networks called sequentially; network log/context fields are overwritten per forward. Per-vehicle preprocessing, navigation, inference state and PID.',
            'controller_variant': controller_variant,
        }

    def close(self):
        """Release the model references held by this LEAD instance."""
        self.processor = None
        self.inference = None
        self.initial_controllers = {}

    def new_vehicle(self):
        """Create per-vehicle inference state with shared model weights."""
        instance = copy.copy(self)
        instance.inference = copy.deepcopy(self.inference,
            {id(net): net for net in self.inference.nets})
        instance.training = instance.inference.config_training
        instance.closed = instance.inference.config_closed_loop
        instance.expert = instance.inference.config_expert
        instance.initial_controllers = copy.deepcopy(self.initial_controllers)
        return instance

    def reset(self, route, shared_navigation=None):
        import carla
        from lead.common import common_utils
        self.inference.__dict__.update(copy.deepcopy(self.initial_controllers))
        self.inference.step = 4
        self.processor = self.processor_class()
        p = self.processor
        p.training_config, p.config_closed_loop = self.training, self.closed
        native = [(wp.transform, command) for wp, command in route]
        navigation_native=native
        if shared_navigation is not None:
            navigation_native=[(wp.transform,command) for wp,command in shared_navigation['plan']]
            distances=[0.]
            for (a,_),(b,_) in zip(navigation_native,navigation_native[1:]):
                distances.append(distances[-1]+math.hypot(a.location.x-b.location.x,a.location.y-b.location.y))
            ids=suffix_indices(sample_route_indices(navigation_native,self.navigation_sample_factor_m),
                               distances,shared_navigation['start_station_m'])
        else:
            ids = sample_route_indices(native, self.navigation_sample_factor_m)
        p._global_plan_world_coord = [navigation_native[i] for i in ids]
        p._global_plan = []
        for transform, command in p._global_plan_world_coord:
            loc = transform.location
            gps = local_position_to_gps([loc.x, loc.y, loc.z])
            p._global_plan.append((dict(lat=float(gps[0]), lon=float(gps[1]), z=float(gps[2])), command))
        p.step = -1
        # GPS conversion and route planning share a (0, 0) Mercator origin.
        with patch.object(common_utils, 'find_gps_ref', return_value=(0., 0.)):
            p.setup(sensor_agent=True)
        p.noisy_lat_ref = p.noisy_lon_ref = 0.
        for planner in p.gps_waypoint_planners_dict.values():
            planner.route_distances.clear()
            planner.set_route(p._global_plan, True, lat_ref=0., lon_ref=0.)
        p.control = carla.VehicleControl(brake=1.)
        self.navigation = {'input_dense_points': len(native), 'retained_navigation_points': len(ids),
            'sample_factor_m': self.navigation_sample_factor_m, 'sampled_indices': ids,
            'index_scope': 'shared_route' if shared_navigation is not None else 'vehicle_route',
            'shared_route': shared_navigation['route_name'] if shared_navigation is not None else None,
            'sampled_world_xy_m': [[navigation_native[i][0].location.x, navigation_native[i][0].location.y] for i in ids]}
        errors = []
        for transform, _ in native:
            xyz = np.array([transform.location.x, transform.location.y, transform.location.z])
            errors.append(float(np.max(np.abs(common_utils.convert_gps_to_carla(local_position_to_gps(xyz), 0., 0.) - xyz))))
        self.navigation['max_projection_roundtrip_m'] = max(errors)

    def step(self, images, speed_mps, position_xyz, yaw_deg, gps_position_xyz=None):
        import carla
        import torch
        started = time.perf_counter()
        p = self.processor
        p.step += 1
        raw, counts = {}, {}
        for spec in self.sensors:
            key, kind = spec['id'], spec['type']
            value = images[key]
            if kind == 'sensor.camera.rgb':
                array = np.frombuffer(value.raw_data, np.uint8).reshape(value.height, value.width, 4).copy()
            elif kind == 'sensor.lidar.ray_cast':
                array = np.frombuffer(value.raw_data, np.float32).reshape(-1, 4).copy()
                counts[key] = len(array)
            elif kind == 'sensor.other.radar':
                array = decode_radar(value.raw_data)
                counts[key] = len(array)
            else:
                array = np.array([value.accelerometer.x, value.accelerometer.y, value.accelerometer.z,
                    value.gyroscope.x, value.gyroscope.y, value.gyroscope.z, value.compass])
            raw[key] = (value.frame, array)
        frame = images['imu'].frame
        xyz = np.asarray(position_xyz, dtype=float)
        raw['gps'] = (frame, local_position_to_gps(xyz))
        raw['speed'] = (frame, {'speed': np.float64(speed_mps)})
        decoded_at=time.perf_counter()
        data = p.tick(raw)
        preprocessed_at=time.perf_counter()
        if p.step == 0:
            # SensorAgent.run_step initializes and observes its first input,
            # then returns brake without updating network/controller state.
            control = dict(steer=0., throttle=0., brake=1.)
            p.control = carla.VehicleControl(**control)
            return control, dict(model_step=0, initialization_only=True,
                warmup_brake_override=True, input_speed_mps=float(speed_mps),
                navigation_setup=self.navigation, imu_frame=frame,
                sensor_frames={key:value.frame for key,value in images.items()},
                lidar_history_frames=len(p.lidar_pc_queue),
                inference_wall_s=time.perf_counter()-started)
        def tensor(x):
            return torch.as_tensor(np.asarray(x), device=self.device, dtype=torch.float32)
        inputs = {'rgb': tensor(data['rgb'])[None], 'rasterized_lidar': tensor(data['rasterized_lidar']),
            'speed': tensor([speed_mps]), 'command': tensor(data['command']).reshape(1, 6),
            'next_command': tensor(data['next_command']).reshape(1, 6), 'town': np.array(['supplied'])}
        for key in ('target_point_previous', 'target_point', 'target_point_next'):
            inputs[key] = tensor(data[key]).reshape(1, 2)
        if self.training.use_radars:
            inputs['radar'] = tensor(data['radar'])[None]
        transferred_at=time.perf_counter()
        with torch.inference_mode():
            prediction = self.inference.forward(data=inputs)
        # Preserve SensorAgent's raw control, including its waypoint PID output.
        # CARLA's movement component applies the actuator bounds. Pre-clipping
        # here changes negative PID outputs and the processor's previous action.
        control = {key: float(getattr(prediction, key))
                   for key in ('steer', 'throttle', 'brake')}
        controlled_at=time.perf_counter()
        warmup = p.step < self.training.inital_frames_delay
        if warmup:
            control = dict(steer=0., throttle=0., brake=1.)
        p.control = carla.VehicleControl(**control)
        plan = prediction.pred_future_waypoints[0].detach().cpu().float().numpy()
        path = prediction.pred_route[0].detach().cpu().float().numpy()
        target_speed = float(prediction.pred_target_speed_scalar.reshape(-1)[0])
        one_second=int(self.training.carla_fps//self.training.data_save_freq)
        waypoint_speed=float(np.linalg.norm(plan[one_second//2-1]-plan[one_second-1])*2.)
        active_speed=waypoint_speed if self.closed.throttle_modality=='waypoint' else target_speed
        imu_error = (images['imu'].compass - math.radians(yaw_deg) - math.pi / 2 + math.pi) % (2*math.pi) - math.pi
        diagnostic = {'target_forward_right_m': data['target_point'].tolist(),
            'navigation_points_forward_right_m': {key: data[key].tolist() for key in ('target_point_previous', 'target_point', 'target_point_next')},
            'navigation_setup': self.navigation if p.step == 0 else None,
            'command': int(np.argmax(data['command'])) + 1,
            'predicted_waypoints_forward_right_m': plan.tolist(),
            'predicted_route_forward_right_m': path.tolist(),
            'predicted_waypoint_spacing_s': self.training.waypoints_spacing / self.training.carla_fps,
            'pid': {'desired_speed': active_speed, 'longitudinal_mode': self.closed.throttle_modality,
                    'target_speed_head_mps': target_speed, 'waypoint_speed_mps': waypoint_speed,
                    'steer': float(prediction.steer),
                    'throttle': float(prediction.throttle), 'brake': float(prediction.brake)},
            'target_speed_class_probabilities': prediction.pred_target_speed_distribution[0].detach().cpu().float().tolist(),
            'warmup_brake_override': warmup, 'input_speed_mps': float(speed_mps),
            'inference_wall_s': time.perf_counter()-started, 'model_step': p.step,
            'wall_timing_s':dict(raw_decode=decoded_at-started,preprocessing=preprocessed_at-decoded_at,
                tensor_transfer_submit=transferred_at-preprocessed_at,network_and_controller=controlled_at-transferred_at,
                output_conversion=time.perf_counter()-controlled_at),
            'imu_frame': frame, 'sensor_frames': {key: value.frame for key,value in images.items()},
            'imu_compass_minus_pose_heading_rad': float(imu_error), 'raw_point_counts': counts,
            'lidar_history_frames': len(p.lidar_pc_queue),
            'localization_world_xyz_m': xyz.tolist(),
            'localization_roundtrip_error_m': float(np.max(np.abs(data['noisy_state']-xyz))),
            'filtered_position_error_m': float(np.linalg.norm(p.filtered_state[:2]-xyz[:2]))}
        diagnostic['lateral_lookahead'] = next((v.lookahead_diagnostic for v in vars(self.inference).values()
            if hasattr(v, 'lookahead_diagnostic')), None)
        return control, diagnostic
