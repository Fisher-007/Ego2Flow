import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from benchmark.policies.vlm_inputs import CalibratedRig,CausalHistory
from benchmark.policies.vlm_worker import parse_plan,sampling_seed
from benchmark.policies.vlm import navigation_text

class VLMInputTests(unittest.TestCase):
    def history(self,acceleration=0.,lateral=0.):
        h=CausalHistory()
        for f in range(61):
            t=f*.05;pose=np.eye(4);pose[:3,3]=[4*t+.5*acceleration*t*t,lateral*t,0]
            h.append(f,t,pose,{'frame':f})
        return h

    def test_causal_motion_uses_actual_past_frames_and_left_sign(self):
        h=self.history(acceleration=2.,lateral=-1.)
        rig=SimpleNamespace(image=lambda name,value:np.zeros((1,1,3),np.uint8))
        h.images[-1][2].update(ring_front_center=None,ring_front_left=None,ring_front_right=None)
        data=h.inputs('onevl',rig)
        self.assertEqual(data['history_frames'],[[30,20,10],[40,30,20],[50,40,30],[60,50,40]])
        np.testing.assert_allclose(data['states'][-1]['velocity_xy_mps'],[9.5,1.])
        np.testing.assert_allclose(data['states'][-1]['acceleration_xy_mps2'],[2.,0.])
        np.testing.assert_allclose(data['states'][0]['xy_m'],[-12.75,-1.5])
        self.assertEqual(data['states'][-1]['xy_m'],[0.,0.])

    def test_histories_are_independent_and_not_padded(self):
        a,b=CausalHistory(),CausalHistory()
        a.append(7,1.,np.eye(4),{})
        a.append(7,1.,np.eye(4),{})
        self.assertFalse(a.ready('onevl'));self.assertFalse(a.ready('alpamayo'))
        self.assertEqual(len(a.poses),1);self.assertEqual(len(b.poses),0)
        with self.assertRaises(ValueError):a.pose_at(0.)

    def test_alpamayo_camera_major_history_and_actual_pose(self):
        h=self.history()
        for row in h.images:row[2].update(ring_front_center=1,ring_front_left=2,ring_front_right=3)
        data=h.inputs('alpamayo',SimpleNamespace(image=lambda name,value:np.full((1,1,3),value,np.uint8)))
        self.assertEqual(data['history_frames'],list(range(30,61,2)))
        self.assertEqual(data['camera_frames'],[54,56,58,60]*3)
        self.assertEqual([x['data'][0] for x in data['images']],[2]*4+[1]*4+[3]*4)
        np.testing.assert_allclose(data['ego_history_xyz'][0],[-6.,0.,0.])
        np.testing.assert_allclose(data['ego_history_rot'],np.tile(np.eye(3),(16,1,1)))

    def test_camera_mount_pose_roundtrip_and_pixel_coverage(self):
        cfg=json.loads((Path(__file__).parents[1]/'configs/sensors/av2_front_rig.json').read_text())
        rig=CalibratedRig(cfg['calibration']);body=np.eye(4);body[:3,3]=[100,-20,2]
        for spec in rig.sensors:
            mount=rig.mounts[spec['id']]
            self.assertAlmostEqual(np.linalg.det(mount[:3,:3]),1.)
            mx,my=rig.remaps[spec['id']]
            self.assertGreaterEqual(mx.min(),0);self.assertLess(mx.max(),spec['width'])
            self.assertGreaterEqual(my.min(),0);self.assertLess(my.max(),spec['height'])
        value=SimpleNamespace(transform=SimpleNamespace(get_matrix=lambda:body@rig.mounts['ring_front_center']))
        np.testing.assert_allclose(rig.pose({'ring_front_center':value}),body,atol=1e-12)

    def test_parse_does_not_invent_missing_points(self):
        text=', '.join(f'[{i}, {i/10}]'for i in range(10))
        self.assertEqual(len(parse_plan(text)[0]),10)
        points,repaired=parse_plan(text[1:]);self.assertTrue(repaired);self.assertEqual(points,parse_plan(text)[0])
        self.assertIsNone(parse_plan('[1,2], [3,4]')[0])

    def test_random_stream_is_independent_of_queue_history(self):
        a=dict(request_id=4,session_id=0,inference_index=2)
        b=dict(a,session_id=59)
        self.assertEqual(sampling_seed(7,a),sampling_seed(7,b))
        self.assertNotEqual(sampling_seed(7,a),sampling_seed(7,dict(a,request_id=5)))

    def test_navigation_uses_assigned_upcoming_branch(self):
        route=[(np.array([i*10.,0.]),1 if i==4 else 4)for i in range(10)]
        index,text=navigation_text(route,0,np.array([12.,0.]))
        self.assertEqual(index,1);self.assertIn('Turn left',text)

if __name__=='__main__':unittest.main()
