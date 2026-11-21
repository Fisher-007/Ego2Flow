"""Check plugin lifecycle and independent recurrent/controller ownership."""
import json
from types import SimpleNamespace
import sys
import numpy as np
from benchmark import ROOT
from benchmark.policies.adapters import configure_inference, load_policy
from benchmark.policies.uniad import UniADPolicy, motion_anchor_path
from benchmark.runtime.execution import method_spec


def plan():
    return [(SimpleNamespace(transform=SimpleNamespace(location=SimpleNamespace(x=float(x),y=0.))),
             SimpleNamespace(value=4)) for x in range(0,100,5)]


def test_custom_policy_runs_without_a_torch_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',None)
    assert configure_inference(dict(kind='custom'),7)['framework'] == 'adapter_managed'
    owner=load_policy(dict(kind='custom',factory='examples.route_follower:create'))
    a,b=owner.new_vehicle(),owner.new_vehicle()
    a.reset(plan());b.reset(plan())
    control,_=a.step({},0.,[0.,0.,0.],0.)
    assert control['throttle'] > 0 and a.index == 1 and b.index == 0
    a.close()
    assert b.points and owner.sensors == []
    owner.close()


def test_pose_only_session_can_observe_an_empty_sensor_frame():
    from benchmark.runtime.sensor_history import observe_admitted_history
    observed = []
    session = SimpleNamespace(observe_inputs=lambda inputs, preparation: observed.append(inputs))
    item = {'policy': session, 'request': {'admitted_s': 0.}}
    assert observe_admitted_history(item, {}, 1.)
    assert observed == [{}]


class Model:
    def __init__(self):
        self.weight=np.ones(1)
        self.buffer=np.zeros(1)
        self.previous=dict(bev=[])
    def parameters(self):return [self.weight]
    def buffers(self):return [self.buffer]


def test_uniad_sessions_share_only_frozen_storage(monkeypatch):
    owner = UniADPolicy.__new__(UniADPolicy)
    owner.model, owner.pipeline, owner.sensors, owner.box_type = Model(), [dict(cache=[])], [], object()
    owner.rig = SimpleNamespace(imu_x_m=-1.4, frame=lambda inputs, speed, *args: (speed, [0., 10.], {}))
    a, b = owner.new_vehicle(), owner.new_vehicle()
    a.reset(plan())
    b.reset(plan())
    assert a.model.weight is b.model.weight and a.model.buffer is b.model.buffer
    assert a.model.previous is not b.model.previous

    def prediction(speed):
        a.model.previous['bev'].append(speed)
        return np.array([[0., 3.], [0., 6.], [0., 9.], [0., 12.]])

    monkeypatch.setattr(a, 'predict', prediction)
    control, diagnostic = a.step({}, 2., [0., 0., 0.], 0., gps_position_xyz=[-1.4, 0., 0.])
    assert control == dict(steer=0., throttle=.75, brake=0.)
    assert diagnostic['predicted_waypoints_forward_right_m'][:2] == [[3., 0.], [6., 0.]]
    assert (a.frame_index, b.frame_index) == (0, -1)
    assert a.tracker.speed.errors[-1] == .25 and b.tracker.speed.errors[-1] == 0.
    assert b.model.previous['bev'] == [] and owner.model.previous['bev'] == []
    a.pipeline[0]['cache'].append(7)
    assert b.pipeline[0]['cache'] == []
    assert a.step({}, 5., [0., 0., 0.], 0.)[0]['throttle'] > 0
    assert a.step({}, 5.01, [0., 0., 0.], 0.)[0]['throttle'] == 0
    a.reset(plan())
    assert a.frame_index == -1 and a.model.previous['bev'] == [] and a.tracker.speed.errors[-1] == 0.
    a.close()
    assert b.model.weight[0] == 1.


def test_uniad_resolves_assets_without_a_checkout(tmp_path, monkeypatch):
    model_dir = tmp_path / 'weights'
    model_dir.mkdir()
    anchors = model_dir / 'anchors.pkl'
    anchors.touch()
    for name, value in dict(EGO2FLOW_ROOT=str(ROOT), UNIAD_CONFIG=str(tmp_path / 'stage2.py'),
                            UNIAD_WEIGHTS=str(model_dir / 'model.pth'), UNIAD_PYTHON=sys.executable).items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv('MODEL_SOURCE', raising=False)
    method_path = ROOT / 'configs/methods/UniAD.json'
    visual = method_spec({'methods': {'UniAD': method_path}}, 'UniAD')['visual_policy']
    assert 'source_root' not in visual and motion_anchor_path(visual, 'data/others/anchors.pkl') == anchors
    custom = tmp_path / 'alternate.pkl'
    custom.touch()
    assert motion_anchor_path(dict(visual, motion_anchor='alternate.pkl'), 'anchors.pkl') == custom

    old = json.loads(method_path.read_text())
    old['visual_policy']['source_root'] = '${MODEL_SOURCE}'
    old_path = tmp_path / 'method.json'
    old_path.write_text(json.dumps(old))
    assert method_spec({'methods': {'UniAD': old_path}}, 'UniAD')['visual_policy'] == visual
