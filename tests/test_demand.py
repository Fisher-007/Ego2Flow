import unittest
from benchmark.scenes.demand import arrival_times


class SustainedDemand(unittest.TestCase):
    def test_arrivals_continue_until_window_end_without_extra_boundary_request(self):
        self.assertEqual(arrival_times({"duration_s": 10, "demand": {"headway_s": 3}}), [0, 3, 6, 9])
        self.assertEqual(len(arrival_times({"duration_s": 360, "demand": {"headway_s": 5}})), 72)

    def test_legacy_burst_is_preserved(self):
        self.assertEqual(arrival_times({"duration_s": 180, "arrival_times_s": [0, 0, 0, 15, 30, 45]}),
                         [0, 0, 0, 15, 30, 45])

class InitialPopulation(unittest.TestCase):
    def test_initial_population_and_boundary_arrivals_share_a_complete_ledger(self):
        from benchmark.scenes.demand import scheduled_requests, entry_heads
        requests = scheduled_requests({"duration_s": 10,
            "initial_vehicles": [{"route": "inside", "initial_speed_mps": 4.}],
            "demand_streams": [{"route": "entrance", "arrival_times_s": [0, 4, 8]}]})
        self.assertEqual(len(requests), 4)
        self.assertEqual([r["arrival_s"] for r in requests], [0, 0, 4, 8])
        self.assertEqual(requests[0]["origin"], "initial_population")
        self.assertIsNone(requests[0]["admitted_s"])
        self.assertEqual(requests[0]["initial_speed_mps"], 4.)
        self.assertEqual(len(entry_heads(requests[:2])), 2)
        # A failed initial spawn remains pending, independently of the entrance.
        requests[1]["admitted_s"] = .1
        self.assertEqual([r["id"] for r in requests if r["admitted_s"] is None], [0, 2, 3])
