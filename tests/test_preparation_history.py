"""Exercise preparation timing with the actual causal history requirement."""
from queue import Queue
from types import SimpleNamespace as NS
import unittest
import copy
from unittest.mock import patch

import numpy as np

from benchmark.runtime.sensor_history import prepare_physical_history, observe_admitted_history
from benchmark.policies.vlm_inputs import CausalHistory


class Control:
    def __init__(self, throttle=0., steer=0., brake=0.):
        self.throttle, self.steer, self.brake = throttle, steer, brake
        self.hand_brake = self.reverse = self.manual_gear_shift = False
        self.gear = 1


class PreparationHistoryTests(unittest.TestCase):
    def run_preparation(self, settling_s, defer, history_start_s=0., require_ready=False):
        queue = Queue()
        state = NS(frame=0, x=0., speed=0., control=Control(), target_applications=0)
        policy = NS(history=CausalHistory(), calls=[], localization_sensor_x_m=0.)

        def observe(images, preparation=False):
            image = images['camera']
            pose = np.eye(4); pose[0, 3] = state.x
            policy.history.append(image.frame, image.timestamp, pose, {})

        def step(images, *args, **kwargs):
            ready = policy.history.ready('onevl')
            policy.calls.append((state.frame, ready))
            return dict(throttle=.2 if ready else 0., steer=0., brake=0. if ready else 1.), dict(
                model_step=len(policy.calls), model_called=ready,
                status='predicted' if ready else 'waiting_for_actual_history',
                history_duration_s=policy.history.poses[-1][0]-policy.history.poses[0][0])

        policy.observe_inputs, policy.step = observe, step
        def history_status():
            poses = policy.history.poses
            return dict(ready=policy.history.ready('onevl'), required_s=2.5,
                duration_s=poses[-1][0]-poses[0][0] if poses else 0.,
                first_frame=poses[0][1] if poses else None,
                last_frame=poses[-1][1] if poses else None)
        policy.input_history_status = history_status
        transform = lambda: NS(location=NS(x=state.x,y=0.,z=0.), rotation=NS(yaw=0.),
            get_forward_vector=lambda: 1., transform=lambda point: point)
        vehicle = NS(id=1, is_dormant=False, get_transform=transform,
            get_velocity=lambda: NS(dot=lambda forward: state.speed), get_control=lambda: state.control)

        def tick():
            state.frame += 1
            if state.control.brake:
                state.speed = 0.
            state.x += .05 * state.speed
            queue.put(NS(frame=state.frame, timestamp=state.frame*.05))
            return state.frame

        def apply(commands, do_tick):
            self.assertFalse(do_tick)
            for kind, value in commands:
                if kind == 'control': state.control = value
                else:
                    state.speed = value; state.target_applications += 1
            return [NS(has_error=lambda: False) for command in commands]

        carla = NS(VehicleControl=Control, Location=lambda x: NS(x=x,y=0.,z=0.), command=NS(
            ApplyVehicleControl=lambda actor, value: ('control',value),
            ApplyTargetVelocity=lambda actor, value: ('velocity',value)))
        active = {0: dict(vehicle=vehicle, request=dict(id=0,initial_speed_mps=5.,
            time_zero_anchor_position_m=[5.5,0.,0.]), is_visual=True, policy=policy,
            policy_queues={'camera':queue})}
        with patch('benchmark.navigation.prepared_routes.align_prepared_background_routes', return_value=[]):
            report = prepare_physical_history(NS(apply_batch_sync=apply), NS(tick=tick), active,
                .05, carla, settling_s=settling_s, rolling_s=1.1, native_policy=True,
                defer_policy_until_motion=defer, history_start_s=history_start_s,
                require_ready_before_motion=require_ready)
        state.first_capture_frame = active[0].get('history_capture_first_frame')
        return report, state, policy

    def test_original_timing_reproduces_history_wait_during_motion(self):
        report, state, policy = self.run_preparation(1.5,False)
        rolling = [x for x in report['trace'] if x['stage']=='physical_rolling']
        self.assertTrue(any(x['vehicles'][0]['policy_status']=='waiting_for_actual_history' for x in rolling))
        self.assertEqual(report['states_at_time_zero'][0]['measured_speed_mps'],0.)

    def test_real_history_is_ready_before_first_moving_prediction(self):
        report, state, policy = self.run_preparation(2.55,True)
        self.assertTrue(all(ready for frame,ready in policy.calls))
        self.assertGreater(policy.calls[0][0]*.05,report['settling_s'])
        self.assertEqual(state.target_applications,1)
        self.assertEqual(report['states_at_time_zero'][0]['measured_speed_mps'],5.)
        self.assertTrue(all(x['vehicles'][0]['model_step'] is None for x in report['trace'] if x['stage']=='settling'))

    def test_settle_frames_excluded_and_real_history_ready_at_release(self):
        report, state, policy = self.run_preparation(3.55, True, 1., True)
        release = report['input_history_at_motion_release'][0]
        self.assertTrue(release['ready'])
        self.assertEqual(release['first_frame'], 21)
        self.assertEqual(state.first_capture_frame, 21)
        self.assertAlmostEqual(release['duration_s'], 2.5)
        self.assertTrue(all(ready for frame, ready in policy.calls))
        self.assertTrue(all(not row['vehicles'][0]['input_history']['ready']
            for row in report['trace'][:20]))
        self.assertEqual(state.target_applications, 1)
        self.assertEqual(state.speed, 5.)

    def test_short_configuration_rejected_before_motion(self):
        with self.assertRaisesRegex(ValueError, 'real history before motion'):
            self.run_preparation(2.55, True, 1., True)

    def test_cold_admission_uses_post_settle_frames_and_independent_states(self):
        seen = [[], []]
        items = [dict(policy=NS(observe_inputs=lambda images, preparation=False, i=i:
            seen[i].append(images['camera'].frame)), request=dict(admitted_s=8.))
            for i in range(2)]
        image = {'camera': NS(frame=100)}
        self.assertFalse(observe_admitted_history(items[0], image, 8.95, True))
        self.assertTrue(observe_admitted_history(items[0], image, 9., True))
        self.assertEqual(seen, [[100], []])
        self.assertEqual(items[0]['request']['admitted_s'], 8.)
        self.assertEqual(items[0]['history_capture_first_frame'], 100)
        items[1]['initial_prepared'] = True
        self.assertTrue(observe_admitted_history(items[1], image, 8., True))



if __name__ == '__main__':
    unittest.main()
