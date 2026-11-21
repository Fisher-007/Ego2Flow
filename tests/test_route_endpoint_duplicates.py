import copy
import unittest

from benchmark.metrics.route_outcomes import evaluate


class RepeatedRouteEndpoints(unittest.TestCase):
    def test_original_terminal_direction_and_unfinished_demand_are_preserved(self):
        points = [{'position_carla_m': p, 's': s} for s, p in enumerate(
            [[0, 0, 0], [10, 0, 0], [10, 10, 0], [10, 10, 0], [10, 10, 0]])]
        original = copy.deepcopy(points)
        requests = [{'id': 0, 'route': 'turn', 'arrival_s': 0, 'completed_s': 5},
                    {'id': 1, 'route': 'turn', 'arrival_s': 0, 'completed_s': 6},
                    {'id': 2, 'route': 'turn', 'arrival_s': 3, 'completed_s': None}]
        frames = [{'time_s': 5, 'vehicles': [{'request_id': 0, 'forward_xy': [0, 1]}]},
                  {'time_s': 6, 'vehicles': [{'request_id': 1, 'forward_xy': [0, -1]}]}]
        result = evaluate(requests, frames, {'turn': points}, 20)
        self.assertEqual(points, original)
        self.assertEqual(result['passed_bounded_route_check'], 1)
        self.assertEqual(result['endpoint_exits_failing_route_check'], 1)
        self.assertEqual(result['unresolved_route_tasks_at_cutoff'], 2)
        self.assertEqual(result['request_to_route_check_pass_or_cutoff_vehicle_seconds'], 42)
        self.assertEqual([r['endpoint_forward_dot'] for r in result['requests']], [1., -1., None])


if __name__ == '__main__':
    unittest.main()
