import copy
import unittest
from benchmark.policies.vlm_inputs import motion_text, onevl_motion_text


class OneVLMotionEncodingTests(unittest.TestCase):
    def setUp(self):
        self.states=[dict(offset_s=t,xy_m=[0.,0.],velocity_xy_mps=[0.,-0.],
            acceleration_xy_mps2=[0.,0.]) for t in [-1.5,-1.,-.5,0.]]

    def test_default_preserves_existing_prompt_and_input(self):
        before=copy.deepcopy(self.states)
        self.assertEqual(onevl_motion_text(self.states),(motion_text(self.states),'numeric_history'))
        self.assertEqual(self.states,before)

    def test_static_summary_retains_observed_duration_without_fabricated_motion(self):
        self.states[0]['xy_m'][0]=.004
        self.states[1]['velocity_xy_mps'][1]=-.004
        before=copy.deepcopy(self.states)
        self.assertEqual(onevl_motion_text(self.states,'stationary_summary'),
            ('Past ego motion: stationary over the last 1.5 seconds.\n','stationary_summary'))
        self.assertEqual(self.states,before)

    def test_nonzero_pose_velocity_or_acceleration_keeps_all_numeric_fields(self):
        for field in ['xy_m','velocity_xy_mps','acceleration_xy_mps2']:
            with self.subTest(field=field):
                states=copy.deepcopy(self.states);states[0][field][1]=.006
                self.assertEqual(onevl_motion_text(states,'stationary_summary'),
                    (motion_text(states),'numeric_history'))


if __name__=='__main__':unittest.main()
