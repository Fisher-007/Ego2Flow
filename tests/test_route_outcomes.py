import unittest
from benchmark.metrics.route_outcomes import evaluate


class RouteOutcomes(unittest.TestCase):
    def test_measured_exit_uses_crossing_before_receiving_area(self):
        points=[{'position_carla_m':[0,0,0]},{'position_carla_m':[10,0,0]}]
        frames=[{'time_s':12,'vehicles':[{'request_id':0,'corridor':{'exit_forward_dot':1}}]},
                {'time_s':15,'vehicles':[{'request_id':0,'forward_xy':[-1,0]}]}]
        r=evaluate([{'id':0,'arrival_s':0,'completed_s':12}],frames,points,20,
                   task='road_corridor_and_exit')
        self.assertEqual(r['passed_bounded_route_check'],1)
        self.assertEqual(r['requests'][0]['endpoint_forward_dot'],1)
        self.assertEqual(r['request_to_route_check_pass_or_cutoff_vehicle_seconds'],12)

    def test_missing_crossing_sample_does_not_invent_success(self):
        points=[{'position_carla_m':[0,0,0]},{'position_carla_m':[10,0,0]}]
        r=evaluate([{'id':0,'arrival_s':0,'completed_s':12}],
                   [{'time_s':13,'vehicles':[{'request_id':0,'forward_xy':[1,0]}]}],
                   points,20,task='road_corridor_and_exit')
        self.assertEqual(r['passed_bounded_route_check'],0)
        self.assertTrue(r['requests'][0]['exit_state_missing'])

    def test_departed_endpoint_exit_keeps_unresolved_task_age(self):
        requests=[{'id':0,'arrival_s':0,'completed_s':60,'first_off_route_s':15},
                  {'id':1,'arrival_s':15,'completed_s':100}, {'id':2,'arrival_s':30}]
        frames=[{'time_s':60,'vehicles':[{'request_id':0,'forward_xy':[1,0]}]},
                {'time_s':100,'vehicles':[{'request_id':1,'forward_xy':[1,0]}]}]
        points=[{'position_carla_m':[0,0,0]},{'position_carla_m':[10,0,0]}]
        r=evaluate(requests,frames,points,180)
        self.assertEqual(r['physical_endpoint_exits'],2)
        self.assertEqual(r['passed_bounded_route_check'],1)
        self.assertEqual(r['unresolved_route_tasks_at_cutoff'],2)
        self.assertEqual(r['request_to_route_check_pass_or_cutoff_vehicle_seconds'],415)

    def test_reverse_endpoint_is_not_forward_route_completion(self):
        r=evaluate([{'id':0,'arrival_s':0,'completed_s':12}],
                   [{'time_s':12,'vehicles':[{'request_id':0,'forward_xy':[-1,0]}]}],
                   [{'position_carla_m':[0,0,0]},{'position_carla_m':[10,0,0]}],45)
        self.assertEqual(r['endpoint_exits_failing_route_check'],1)
        self.assertEqual(r['request_to_route_check_pass_or_cutoff_vehicle_seconds'],45)
