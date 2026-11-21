"""Run UniAD stage-2 planning on Ego2Flow sensor frames."""
from collections import deque
import copy
from pathlib import Path
import time

import numpy as np

from benchmark import ROOT
from benchmark.policies.trajectory_tracker import TrajectoryTracker, WAYPOINT_INTERVAL_S
from benchmark.policies.uniad_inputs import UniADRig, route_target


def create(config):
    return UniADPolicy(config)


def motion_anchor_path(config, reference):
    """Locate the motion anchors for the stage-2 configuration."""
    config_dir = Path(config['config']).resolve().parent
    if config.get('motion_anchor'):
        return (config_dir / config['motion_anchor']).resolve(strict=True)
    reference = Path(reference)
    if reference.is_absolute():
        return reference.resolve(strict=True)
    checkpoint_dir = Path(config['checkpoint']).resolve().parent
    candidates = [config_dir / reference, checkpoint_dir / reference.name]
    if config.get('asset_root'):
        candidates.insert(0, Path(config['asset_root']) / reference)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f'Place {reference.name} beside the UniAD checkpoint or set visual_policy.motion_anchor')


def stage2_settings(config):
    from mmcv import Config
    settings = Config.fromfile(str(Path(config['config']).resolve()))
    head = settings.model['motion_head']
    anchors = motion_anchor_path(config, head['anchor_info_path'])
    head['anchor_info_path'] = str(anchors)
    return settings, anchors


class UniADPolicy:
    def __init__(self, config):
        import torch
        from mmcv.core.bbox import get_box_type
        from mmcv.datasets.pipelines import Compose
        from mmcv.models import build_model
        from mmcv.utils import load_checkpoint

        self.device = torch.device(config.get('device', 'cuda:0'))
        torch.cuda.set_device(self.device)
        settings, anchors = stage2_settings(config)
        self.model = build_model(settings.model, train_cfg=settings.get('train_cfg'),
                                 test_cfg=settings.get('test_cfg'))
        checkpoint = str(Path(config['checkpoint']).resolve())
        load_checkpoint(self.model, checkpoint, map_location='cpu', strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.pipeline = Compose([stage for stage in settings.inference_only_pipeline
                                 if stage['type'] != 'LoadMultiViewImageFromFilesInCeph'])
        self.box_type = get_box_type('LiDAR')[0]
        self.rig = UniADRig(config.get('calibration', ROOT / 'configs/sensors/uniad_rig.json'))
        self.sensors = self.rig.sensors
        self.description = dict(name='UniAD', checkpoint=checkpoint,
                                config=str(Path(config['config']).resolve()), motion_anchor=str(anchors),
                                source='https://github.com/Thinklab-SJTU/Bench2DriveZoo',
                                sensors=self.sensors, localization='World pose at the GNSS mount',
                                control='Trajectory PID with a navigation target',
                                state='Shared frozen weights and buffers, per-vehicle model history, route and PID')

    def new_vehicle(self):
        return UniADSession(self)

    def close(self):
        self.model = self.pipeline = None


class UniADSession:
    def __init__(self, owner):
        self.owner = owner
        self.sensors = owner.sensors
        self.localization_sensor_x_m = owner.rig.imu_x_m

    def reset(self, route):
        template = self.owner.model
        frozen = {id(t): t for t in (*template.parameters(), *template.buffers())}
        self.model = copy.deepcopy(template, frozen)
        self.pipeline = copy.deepcopy(self.owner.pipeline)
        self.tracker = TrajectoryTracker()
        self.route = deque((np.array([wp.transform.location.x, wp.transform.location.y]), command.value)
                           for wp, command in route)
        self.frame_index = -1

    def predict(self, sample):
        import torch
        from mmcv.parallel.collate import collate

        with torch.no_grad():
            batch = collate([self.pipeline(sample)], samples_per_gpu=1)
            for name, group in batch.items():
                if name != 'img_metas' and torch.is_tensor(group[0]):
                    group[0] = group[0].to(self.owner.device)
            result = self.model(batch, return_loss=False, rescale=True)
            return result[0]['planning']['result_planning']['sdc_traj'][0].cpu().numpy()

    def step(self, images, speed_mps, position_xyz, yaw_deg, gps_position_xyz=None):
        started = time.perf_counter()
        self.frame_index += 1
        position = np.asarray(position_xyz if gps_position_xyz is None else gps_position_xyz)
        target, command = route_target(self.route, position)
        sample, pid_target, diagnostic = self.owner.rig.frame(
            images, speed_mps, position, yaw_deg, target, command, self.frame_index, self.owner.box_type)
        plan = self.predict(sample)
        control, pid = self.tracker.step(plan, speed_mps, navigation_target=pid_target)
        if speed_mps > 5.0:
            control['throttle'] = 0.0
        control = {k: float(np.float32(v)) for k, v in control.items()}
        diagnostic.update(predicted_waypoints_forward_right_m=plan[:, [1, 0]].tolist(),
                          predicted_waypoint_spacing_s=WAYPOINT_INTERVAL_S, input_speed_mps=float(speed_mps),
                          model_step=self.frame_index, pid=pid, inference_wall_s=time.perf_counter() - started)
        return control, diagnostic

    def close(self):
        self.model = self.pipeline = self.tracker = None
        self.route = deque()
