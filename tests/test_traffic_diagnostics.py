import unittest
from benchmark.metrics.traffic_diagnostics import service_totals, edie_totals


class TrafficDiagnosticsTests(unittest.TestCase):
    def test_removed_request_remains_unserved(self):
        a=dict(id=0,arrival_s=0,admitted_s=0,completed_s=10)
        b=dict(id=1,arrival_s=5,admitted_s=None,completed_s=None)
        r=service_totals([a,b],20)
        self.assertEqual(r['service_vehicle_s'],25)
        self.assertEqual(r['service_s_per_request'],12.5)
        self.assertEqual(r['completion_rate'],.5)
        b.update(admitted_s=6,removed_s=8,failure_reason='technical_removal')
        r=service_totals([a,b],20)
        self.assertEqual(r['service_vehicle_s'],25)
        self.assertEqual(r['physical_in_network_vehicle_s'],12)
        self.assertEqual(r['removed_unserved_vehicle_s'],12)
        self.assertTrue(r['accounting_balances'])

    def test_plan_edie_example_and_empty_domain(self):
        r=edie_totals(300,60,200,20)
        self.assertEqual((r['q_veh_h_lane'],r['k_veh_km_lane'],r['v_kmh']),(270,15,18))
        self.assertIsNone(edie_totals(0,0,200,20)['v_kmh'])
        self.assertEqual(edie_totals(0,60,200,20)['v_kmh'],0)

