"""Small complete logs with hand-computable exposure and request accounting."""
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.metrics.evaluation import evaluate, METRIC_COLUMNS
from tests.test_eight_metrics import car, route


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data))


def fixture(root,task='service',duration=2.,differences=(0,0,0)):
    scene_dir=root/'scene';run=root/'run';run.mkdir(parents=True)
    routes=dict(lane0=route())
    routes['lane0']['evaluation']['legal_route_names']=['lane0']
    spec=dict(task=task,window_s=[0,duration],analysis_route_names=['lane0'],
        analysis_polygon_carla_xy_m=[[-5,-5],[105,-5],[105,5],[-5,5]],
        target_chain_request_ids=[0,1,2],target_chain_route='lane0')
    dt=1. if task=='propagation' else .05
    config=dict(scene_id='example',seed=7,deployment='H',vehicle_blueprint='test',fixed_delta_seconds=dt,
        duration_s=duration,evaluation=spec,request_manifest=[dict(id=i,route='lane0',arrival_s=0,initial_speed_mps=3+differences[i]) for i in range(3)])
    from shapely.geometry import Polygon,mapping
    config['metric_scope']=dict(version='test-explicit',ordinary_request_ids=[0,1,2],window_s=[0,duration],
        region_geojson=mapping(Polygon(spec['analysis_polygon_carla_xy_m'])),normal_exit='legal_task_completion')
    write(scene_dir/'candidate_routes.json',routes);write(scene_dir/'config.json',config);write(run/'config.json',config)
    write(run/'input_receipt.json',dict(scene_files={'candidate_routes.json':routes}))
    write(run/'run.json',dict(scene_dir=str(scene_dir),map_version='sample',elapsed_s=duration,checks=dict(observation_completed=True)))
    requests=[dict(id=i,route='lane0',arrival_s=0,admitted_s=0 if i<2 else None,completed_s=None) for i in range(3)]
    write(run/'requests.json',requests)
    vehicles=lambda t:([car(i,60-i*15+.5*t,speed=3+(differences[i] if i==0 or t>0 else 0)) for i in range(3)] if task=='propagation'
                       else [car(0,20+5*t),car(1,30+5*t)])
    with (run/'initial_states.jsonl').open('w') as f:
        for v in vehicles(0):f.write(json.dumps(dict(time_s=0,vehicle=v))+'\n')
    with (run/'frames.jsonl').open('w') as f:
        for n in range(1,int(round(duration/dt))+1):f.write(json.dumps(dict(frame=n,time_s=n*dt,dt_s=dt,vehicles=vehicles(n*dt)))+'\n')
    (run/'collisions.jsonl').write_text('')
    return run


class PipelineTests(unittest.TestCase):
    def test_optional_detectors_follow_the_saved_scene(self):
        from benchmark.runtime.execution import receipt
        with tempfile.TemporaryDirectory() as temp:
            folder = fixture(Path(temp))
            scene = Path(temp) / 'scene'
            for name in ('map_info.json', 'scene_index.json', 'signal_programs.json'):
                write(scene / name, {})
            detectors = scene / 'measurements.json'
            write(detectors, {'lane0': {'controls': [], 'detector_stations_m': [25.]}})
            write(folder / 'input_receipt.json', receipt({'scene_dir': scene}, {}))
            before = evaluate(folder)
            detectors.unlink()
            (scene / 'candidate_routes.json').unlink()
            after = evaluate(folder)
            self.assertTrue(before['diagnostics']['structure']['detectors'])
            self.assertEqual(before['diagnostics']['structure'], after['diagnostics']['structure'])
            self.assertEqual(before['metrics'], after['metrics'])

    def test_explicit_scene_supplies_routes_and_detectors(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = fixture(Path(temp))
            scene = Path(temp) / 'scene'
            write(scene / 'measurements.json', {'lane0': {'controls': [], 'detector_stations_m': [25.]}})
            (folder / 'input_receipt.json').unlink()
            result = evaluate(folder, scene)
            self.assertEqual(result['scene_dir'], str(scene))
            self.assertEqual(result['route_source'], str(scene / 'candidate_routes.json'))
            self.assertTrue(result['diagnostics']['structure']['detectors'])

    def test_optional_inference_profiles_do_not_change_traffic_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=fixture(Path(temp));baseline=evaluate(folder)
            path=folder/'frames.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            profiles={
                'lead_uniad':lambda i:dict(inference_wall_s=.25),
                'vlm':lambda i:dict(model_called=i%10==0,
                    output=dict(inference_wall_s=.5)),
                'not_recorded':lambda i:dict(status='preparing'),
            }
            expected={'lead_uniad':(40,10.),'vlm':(4,2.),'not_recorded':(0,0.)}
            for name,profile in profiles.items():
                with self.subTest(profile=name):
                    for i,row in enumerate(rows):row['vehicles'][0]['visual_policy']=profile(i)
                    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                    result=evaluate(folder)
                    self.assertEqual(result['metrics'],baseline['metrics'])
                    timing=result['diagnostics']['inference']
                    self.assertEqual(timing['count'],expected[name][0])
                    self.assertAlmostEqual(timing['total_s'],expected[name][1])
                    if not timing['count']:self.assertIsNone(timing['mean_s'])

    def test_retained_platoon_exit_is_not_a_pre_exit_driving_departure(self):
        from benchmark.metrics.traffic_diagnostics import request_failure_context
        rows=[dict(id=9,failure_reason='sustained_corridor_departure',
                   observation_terminal_crossed_s=35.45,first_corridor_departure_s=36.9),
              dict(id=10,failure_reason='sustained_corridor_departure',
                   observation_terminal_crossed_s=50.,first_corridor_departure_s=40.),
              dict(id=2,failure_reason='sustained_corridor_departure')]
        d=request_failure_context(rows,True)
        self.assertEqual(d['failed_requests'],[10,2])
        self.assertEqual(d['raw_failed_requests'],[9,10,2])
        self.assertEqual(d['post_terminal_boundary_requests'][0]['request_id'],9)
        self.assertEqual(request_failure_context(rows,False)['failed_requests'],[9,10,2])
        self.assertEqual(rows[0]['failure_reason'],'sustained_corridor_departure')

    def test_explicit_region_not_incidental_analysis_override_and_external_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=fixture(Path(temp))
            config=json.loads((folder/'config.json').read_text())
            config['evaluation']['energy_population']='all_ordinary_physical_vehicles'
            # Both cars remain inside the declared scoring region.
            config['evaluation']['analysis_polygon_carla_xy_m']=[[-5,-5],[30,-5],[30,5],[-5,5]]
            write(folder/'config.json',config)
            all_cars=evaluate(folder)
            self.assertAlmostEqual(all_cars['coverage']['energy_vehicle_km'],.02)
            self.assertAlmostEqual(all_cars['coverage']['vehicle_km'],.02)
            requests=json.loads((folder/'requests.json').read_text())
            requests[1]['external_intervention']=True
            config['metric_scope']['ordinary_request_ids']=[0,2]
            write(folder/'config.json',config)
            write(folder/'requests.json',requests)
            ordinary=evaluate(folder)
            self.assertAlmostEqual(ordinary['coverage']['energy_vehicle_km'],.01)
            self.assertEqual(ordinary['diagnostics']['energy_coverage']['vehicle_ids'],[0])
            self.assertEqual(ordinary['metrics']['delay_s_per_request']['requests'],2)

    def test_receiving_space_cannot_reduce_pre_exit_delay(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=fixture(Path(temp))
            requests=json.loads((folder/'requests.json').read_text())
            requests[0].update(completed_s=1.,exit_path='lane0')
            write(folder/'requests.json',requests)
            scored=evaluate(folder)
            item=next(r for r in scored['diagnostics']['requests'] if r['request_id']==0)
            self.assertAlmostEqual(item['free_flow_progress_credit_s'],.5)

    def test_scene_energy_pools_moving_and_stopped_background_without_external_requests(self):
        from benchmark.metrics.fixed_energy import estimate_fixed_cycle
        with tempfile.TemporaryDirectory() as temp:
            folder=fixture(Path(temp))
            initial=[json.loads(line) for line in (folder/'initial_states.jsonl').read_text().splitlines()]
            initial[1]['vehicle']=dict(car(1,30,speed=0),deployment_role='background')
            (folder/'initial_states.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in initial))
            rows=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
            for row in rows:row['vehicles'][1]=dict(car(1,30,speed=0),deployment_role='background')
            (folder/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            result=evaluate(folder);settings=result['settings']['energy']
            moving=estimate_fixed_cycle([0,2],[5,5],[0,0],settings)['energy_kwh']
            stopped=estimate_fixed_cycle([0,2],[0,0],[0,0],settings)['energy_kwh']
            self.assertAlmostEqual(result['metrics']['estimated_energy_kwh_per_100_vehicle_km']['value'],100*(moving+stopped)/.01)
            self.assertEqual(result['diagnostics']['energy_coverage']['vehicle_ids'],[0,1])
            self.assertAlmostEqual(result['diagnostics']['energy_coverage']['observed_vehicle_s'],4)
            scene=result['diagnostics']['energy_scene']
            self.assertAlmostEqual(scene['net_kwh'],moving+stopped)
            self.assertAlmostEqual(scene['net_kwh'],scene['drive_kwh']+scene['auxiliary_kwh']-scene['regeneration_kwh'])

    def test_large_reference_exceedance_is_disclosed_with_full_population(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=fixture(Path(temp))
            rows=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
            for row in rows:
                t=row['time_s'];row['vehicles'][0]=car(0,20+5*t+9*t*t,speed=5+18*t,acceleration=(18,0,0))
            (folder/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            result=evaluate(folder)
            self.assertIsNotNone(result['metrics']['estimated_energy_kwh_per_100_vehicle_km']['value'])
            self.assertIn('motor_efficiency_endpoint_extension',result['diagnostics']['energy_scene']['qualifications'])
            self.assertIsNotNone(result['diagnostics']['estimated_energy_kwh_observed'])
            self.assertTrue(result['diagnostics']['energy_coverage']['state_data_complete'])
            self.assertIsNotNone(result['metrics']['delay_s_per_request']['value'])
            scene=result['diagnostics']['energy_scene']
            self.assertEqual(scene['net_kwh'],scene['diagnostic_net_kwh']);self.assertGreater(scene['motor_map_peak_excess_fraction'],0)
            self.assertEqual(scene['vehicle_count'],2)
            self.assertGreater(scene['battery_excess_kwh'],0)

    def test_complete_accounting_and_energy(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=fixture(Path(temp));result=evaluate(folder)
            self.assertEqual(list(result['metrics']),[r[0] for r in METRIC_COLUMNS])
            self.assertAlmostEqual(result['metrics']['delay_s_per_request']['value'],4/3)
            self.assertAlmostEqual(result['coverage']['vehicle_km'],.02)
            self.assertAlmostEqual(result['coverage']['vehicle_s'],4)
            self.assertAlmostEqual(result['metrics']['low_ttc_exposure_percent']['following_vehicle_s'],2)
            self.assertEqual(result['metrics']['speed_sd_mps']['value'],0)
            self.assertTrue(result['coverage']['energy_complete'])
            self.assertGreater(result['metrics']['estimated_energy_kwh_per_100_vehicle_km']['value'],0)
            self.assertEqual(result['coverage_issues'],[])
            self.assertEqual(result['diagnostics']['series'][-1]['time_s'],2.)
            summary=json.loads((folder/'run.json').read_text());summary['checks']['observation_completed']=False
            write(folder/'run.json',summary)
            self.assertEqual(evaluate(folder)['metrics']['delay_s_per_request']['status'],'observed_with_coverage_limits')

    def test_completion_after_last_display_sample_and_shared_receiving_lane(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=fixture(Path(temp),duration=1.5)
            requests=json.loads((folder/'requests.json').read_text())
            requests[0].update(completed_s=1.4,exit_path='lane0')
            write(folder/'requests.json',requests)
            result=evaluate(folder)
            self.assertEqual(result['diagnostics']['series'][-2]['completed'],0)
            self.assertEqual(result['diagnostics']['series'][-1]['completed'],1)
            self.assertEqual(result['diagnostics']['series'][-1]['time_s'],1.5)
            self.assertEqual(result['diagnostics']['traffic']['series'][-1]['finished'],1)
            from benchmark.metrics.evaluation_geometry import EvaluationGeometry
            import copy
            first=route();second=copy.deepcopy(first)
            middle=copy.deepcopy(first['validated_waypoints'][0])
            middle.update(s_m=50,position_carla_m=[50,0,0])
            second['validated_waypoints'].insert(1,middle)
            geom=EvaluationGeometry(dict(a=first,b=second),dict(analysis_route_names=['a','b'],
                analysis_polygon_carla_xy_m=[[-5,-5],[105,-5],[105,5],[-5,5]]))
            self.assertAlmostEqual(geom.lane_length_m,100.)



if __name__=='__main__':unittest.main()
