import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark import ROOT
from benchmark.policies.trajectory_tracker import TrajectoryTracker
from benchmark.policies.vlm import VLMSession
from benchmark.policies.vlm_inputs import ONEVL_CAMERAS
from benchmark.runtime.execution import method_spec


def test_speed_thresholds_and_steering_history():
    tracker = TrajectoryTracker()
    stopped, _ = tracker.step(np.zeros((8, 2)), 0.0)
    assert stopped['brake'] == 1 and stopped['throttle'] == 0

    points = np.column_stack((np.zeros(8), 3.5 * np.arange(1, 9)))
    limit = 7.0 * 1.1
    below, info = tracker.step(points, np.nextafter(limit, 0.0))
    above, _ = tracker.step(points, np.nextafter(limit, np.inf))
    assert info['desired_speed'] == 7.0
    assert below['brake'] == 0 and above['brake'] == 1

    diagonal = np.column_stack((np.arange(1, 9), np.arange(1, 9)))
    tracker = TrajectoryTracker()
    first, _ = tracker.step(diagonal, 0.0)
    for _ in range(40):
        steady, _ = tracker.step(diagonal, 0.0)
    assert first['steer'] == pytest.approx(0.534375)
    assert steady['steer'] == pytest.approx(0.75)
    assert first['throttle'] == 0.75 and first['brake'] == 0


def test_navigation_target_steering():
    ahead = np.column_stack((np.zeros(6), np.arange(1, 7))).astype(np.float32)
    right = ahead + np.array([2, 0], dtype=np.float32)
    _, straight = TrajectoryTracker().step(right, 1., navigation_target=[0., 20.])
    assert straight['angle'] > 0 and straight['angle_final'] == 0.0
    _, close = TrajectoryTracker().step(ahead, 1., navigation_target=[4., 2.])
    assert close['angle_final'] == close['angle_target'] > 0.3
    _, distant = TrajectoryTracker().step(ahead, 1., navigation_target=[40., 20.])
    assert distant['angle_target'] > 0.3 and distant['angle_final'] == distant['angle']


@pytest.mark.parametrize('kind, dt, count', [('onevl', 0.5, 10), ('alpamayo', 0.1, 64)])
def test_vehicle_session_tracks_native_predictions(kind, dt, count):
    times = np.arange(1, count + 1) * dt
    prediction = dict(valid=True, dt_s=dt, trajectory_xy_m=np.column_stack((3 * times, -4 * times)))
    rig = SimpleNamespace(pose=lambda inputs: np.eye(4),
                          image=lambda name, image: np.zeros((1, 1, 3), dtype=np.uint8))
    owner = SimpleNamespace(kind=kind, sensors=[], rig=rig, infer=lambda request: prediction)
    session = VLMSession(owner, 0)
    location = SimpleNamespace(x=10.0, y=0.0)
    route = [(SimpleNamespace(transform=SimpleNamespace(location=location)), SimpleNamespace(value=4))]
    session.bind_request(0)
    session.reset(route)
    for frame in range(51):
        camera = SimpleNamespace(frame=frame, timestamp=frame * 0.05)
        images = dict.fromkeys(ONEVL_CAMERAS, camera)
        session.observe_inputs(images, preparation=True)
    control, log = session.step(images, 2.0, np.zeros(3), 0.0)
    assert log['pid']['desired_speed'] == pytest.approx(5.0)
    assert control['steer'] > 0 and control['throttle'] == 0.75 and control['brake'] == 0
    assert log['pid']['wp_1'] == pytest.approx([2.0, 1.5])


@pytest.mark.parametrize('method', ['OneVL', 'Alpamayo'])
def test_method_resolves_without_external_tracker(tmp_path, monkeypatch, method):
    monkeypatch.delenv('EXTERNAL_TRACKER', raising=False)
    for name in ('ONEVL_WEIGHTS', 'ALPAMAYO_WEIGHTS', 'VLM_CARLA_PYTHON', 'VLM_INFERENCE_PYTHON'):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setenv('EGO2FLOW_ROOT', str(ROOT))
    path = ROOT / 'configs/methods' / (method + '.json')
    current = method_spec({'methods': {method: path}}, method)
    assert 'pid_source' not in current['visual_policy']

    legacy = json.loads(path.read_text())
    legacy['visual_policy']['pid_source'] = '${EXTERNAL_TRACKER}/pid_controller.py'
    previous = tmp_path / 'previous.json'
    previous.write_text(json.dumps(legacy))
    assert method_spec({'methods': {method: previous}}, method) == current
