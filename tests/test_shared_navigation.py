import unittest
from types import SimpleNamespace
from benchmark.navigation.shared_navigation import sample_route_indices, suffix_indices


class SharedNavigationTests(unittest.TestCase):
    def test_sampling_retains_navigation_commands_and_strict_distance_threshold(self):
        class Location(float):
            def distance(self, other):
                return abs(self - other)
        commands = [4, 4, 4, 4, 4, 5, 5, 4, 4, 1, 4, 4]
        route = [(SimpleNamespace(location=Location(i * 2)), SimpleNamespace(value=c))
                 for i, c in enumerate(commands)]
        self.assertEqual(sample_route_indices(route, 4), [0, 4, 5, 6, 9, 10, 11])
        self.assertEqual(sample_route_indices(route[:1], 4), [0])

    def test_cars_keep_identical_downstream_keypoints_without_phase_shift(self):
        stations=list(range(0,1001,2));common=[0,100,200,300,400,500]
        rear=suffix_indices(common,stations,45.6)
        front=suffix_indices(common,stations,225.6)
        self.assertEqual(rear,common)
        self.assertEqual(front,common[1:])
        self.assertEqual(front,rear[1:])

    def test_current_previous_point_and_terminal_pair_are_retained(self):
        self.assertEqual(suffix_indices([0,1,2],[0,200,400],200),[1,2])
        self.assertEqual(suffix_indices([0,1,2],[0,200,400],399),[1,2])
