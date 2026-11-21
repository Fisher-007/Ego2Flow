import json
import unittest
from benchmark.metrics.evaluation import PROFILE
from benchmark.metrics.fixed_energy import estimate_fixed_cycle

class FixedEnergyTests(unittest.TestCase):
    def setUp(self):self.s=json.loads(PROFILE.read_text())['energy']

    def test_stopped_auxiliary(self):
        r=estimate_fixed_cycle([0,10],[0,0],[0,0],self.s)
        self.assertEqual(r['status'],'estimated');self.assertAlmostEqual(r['energy_kwh'],2500/3.6e6)

    def test_contact_sensitivity_is_scene_level_and_requires_complete_events(self):
        from benchmark.metrics.fixed_energy import summarize_scene_energy
        settings={**self.s,'quantify_contact_sensitivity':True}
        item=estimate_fixed_cycle([0,1,101],[4,3.99,3.99],[0,0,0],settings,
                                  collision_intervals=[True,False])
        item.update(request_id=0,start_s=0,end_s=101)
        scene=summarize_scene_energy([item],.4,True,.05,True)
        self.assertTrue(scene['contact_qualified_estimate'])
        self.assertEqual(scene['net_kwh'],item['diagnostic_demand_kwh'])
        self.assertIsNone(summarize_scene_energy([item],.4,True,.05,False)['net_kwh'])
        self.assertIsNone(summarize_scene_energy([item],.4,False,.05,True)['net_kwh'])
        # Report sensitivity above the stored contact-limit threshold.
        item['contact_sensitivity_net_kwh']['positive_work_only']+=1.
        high=summarize_scene_energy([item],.4,True,.05,True)
        self.assertGreater(high['contact_sensitivity']['maximum_relative_change'],.05)
        self.assertEqual(high['net_kwh'],item['diagnostic_demand_kwh'])
        item['invalid_reasons'].append('position_velocity_inconsistent')
        self.assertIsNone(summarize_scene_energy([item],.4,True,.05,True)['net_kwh'])


    def test_collision_is_not_regenerated_and_full_soc_is_limited(self):
        normal=estimate_fixed_cycle([0,1,2],[4,2,0],[0]*3,self.s)
        collision=estimate_fixed_cycle([0,1,2],[4,2,0],[0]*3,self.s,collision_intervals=[True,True])
        self.assertGreater(normal['regenerated_terminal_energy_kwh'],0)
        self.assertEqual(collision['regenerated_terminal_energy_kwh'],0)
        self.assertGreater(collision['diagnostic_demand_kwh'],normal['energy_kwh']);self.assertIsNone(collision['energy_kwh'])
        full=estimate_fixed_cycle([0,1],[4,2],[0,0],self.s,initial_soc=.98)
        self.assertEqual(full['regenerated_terminal_energy_kwh'],0)
        self.assertLessEqual(full['final_soc'],.98)


    def test_motor_endpoint_extension_preserves_full_demand(self):
        r=estimate_fixed_cycle([0,1],[20,30],[0,0],self.s,positions=[[0,0,0],[25,0,0]])
        self.assertEqual(r['status'],'estimated')
        self.assertIn('motor_efficiency_endpoint_extension',r['qualifications'])
        self.assertEqual(r['motor_map_extension_vehicle_s'],1.)
        self.assertGreater(r['motor_map_peak_excess_fraction'],0.)
        self.assertGreater(r['diagnostic_demand_kwh'],r['model']['battery_power_w']/3.6e6)
        self.assertEqual(r['power_exceedance_episodes'][0]['start_position_m'],[0,0,0])
        self.assertGreater(r['power_exceedance_episodes'][0]['battery_excess_kwh'],0)

    def test_motor_endpoint_interval_keeps_unclipped_power(self):
        r=estimate_fixed_cycle([176.,176.05],[10.236396431801769,10.857956587198634],
                               [.006303857386599535]*2,{**self.s,'include_interval_details':True})
        row=r['interval_details'][0]
        mechanical=row['wheel_power_w']/r['model']['transmission_efficiency']
        self.assertAlmostEqual(mechanical,241129.388399897,places=4)
        self.assertAlmostEqual(row['battery_terminal_power_w'],mechanical/.93+250,places=6)
        self.assertAlmostEqual(r['energy_kwh'],.0036045682258057333,places=12)
        self.assertEqual(r['status'],'estimated')

    def test_battery_peak_exceedance_is_reported_without_clipping_demand(self):
        r=estimate_fixed_cycle([0,.05],[7,7.75],[0,0],self.s)
        self.assertEqual(r['status'],'estimated')
        self.assertIn('reference_battery_power_exceeded',r['qualifications'])
        self.assertGreater(r['energy_kwh'],r['model']['battery_power_w']*.05/3.6e6)
        self.assertEqual(r['energy_kwh'],r['diagnostic_demand_kwh'])
        self.assertEqual(r['power_exceedance_episodes'][0]['motor_excess_kwh'],0)

    def test_contact_acceleration_is_not_automatically_motor_work(self):
        contact=estimate_fixed_cycle([0,1],[2,8],[0,0],self.s,collision_intervals=[True])
        normal=estimate_fixed_cycle([0,1],[2,8],[0,0],self.s)
        self.assertIsNone(contact['energy_kwh'])
        self.assertEqual(contact['status'],'energy_contact_attribution_uncertain')
        self.assertLess(contact['traction_terminal_energy_kwh'],normal['traction_terminal_energy_kwh'])
        self.assertEqual(contact['regenerated_terminal_energy_kwh'],0)
        self.assertGreater(contact['unattributed_contact_kinetic_kwh'],0)

    def test_road_load_grade_and_regeneration_energy_balance(self):
        flat=estimate_fixed_cycle([0,10],[8,8],[0,0],self.s)
        uphill=estimate_fixed_cycle([0,10],[8,8],[.02,.02],self.s)
        accel=estimate_fixed_cycle([0,10],[8,10],[0,0],self.s)
        decel=estimate_fixed_cycle([0,10],[8,4],[0,0],self.s)
        self.assertGreater(flat['traction_terminal_energy_kwh'],0)
        self.assertGreater(uphill['energy_kwh'],flat['energy_kwh'])
        self.assertGreater(accel['energy_kwh'],flat['energy_kwh'])
        mechanical=.5*(decel['model']['mass_kg']+decel['model']['rotating_mass_kg'])*(8**2-4**2)/3.6e6
        self.assertLess(decel['regenerated_terminal_energy_kwh'],mechanical)
        self.assertAlmostEqual(decel['energy_kwh'],decel['traction_terminal_energy_kwh']+decel['auxiliary_terminal_energy_kwh']-decel['regenerated_terminal_energy_kwh'])

if __name__=='__main__':unittest.main()
