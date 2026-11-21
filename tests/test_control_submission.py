import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from benchmark.runtime.control_submission import submit_commands, control_echo


class ControlSubmissionTests(unittest.TestCase):
    def test_batch_acknowledges_without_advancing_world(self):
        client = Mock()
        client.apply_batch_sync.return_value = [Mock(has_error=lambda: False)] * 2
        submit_commands(client, ['front', 'rear'])
        client.apply_batch_sync.assert_called_once_with(['front', 'rear'], False)

    def test_failed_submission_is_not_silently_accepted(self):
        client = Mock()
        client.apply_batch_sync.return_value = [Mock(has_error=lambda: True, error='actor missing')]
        with self.assertRaisesRegex(RuntimeError, 'actor missing'):
            submit_commands(client, ['front'])

    def test_previous_tick_control_is_detected(self):
        current = SimpleNamespace(throttle=.4, steer=.2, brake=0., hand_brake=False,
                                  reverse=False, manual_gear_shift=False, gear=0)
        previous = SimpleNamespace(**{**vars(current), 'steer': -.1, 'gear': 2})
        vehicle = Mock(get_control=lambda: previous)
        self.assertFalse(control_echo(vehicle, current)[1])
        previous.steer = current.steer
        # Automatic transmission gear changes are not submitted by the policy.
        self.assertTrue(control_echo(vehicle, current)[1])
