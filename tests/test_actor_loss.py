import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from benchmark.runtime.actor_lifecycle import retire_missing_vehicles
from benchmark.navigation.prepared_routes import align_prepared_background_routes
from benchmark.runtime.run_diagnostics import summarize
from benchmark.metrics.traffic_diagnostics import service_totals
from benchmark.metrics.evaluation import evaluate
from tests.test_evaluation_pipeline import fixture, write
from tests.test_eight_metrics import car


class ActorLossTests(unittest.TestCase):
    def test_simulator_loss_keeps_request_and_other_actor(self):
        request=dict(id=0,arrival_s=0.,admitted_s=0.,completed_s=None,
                     failure_reason='sustained_corridor_departure')
        lost=dict(vehicle=Mock(id=10),request=request)
        survivor=dict(vehicle=Mock(id=11),request={'id':1})
        active={0:lost,1:survivor}
        snapshot=SimpleNamespace(find=lambda actor: object() if actor==11 else None)
        emit=Mock()
        self.assertEqual(retire_missing_vehicles(active,snapshot,165.7,emit),[lost])
        self.assertEqual(active,{1:survivor})
        lost['vehicle'].destroy.assert_not_called()
        self.assertEqual(request['failure_reason'],'sustained_corridor_departure')
        self.assertEqual(request['removed_s'],165.7)
        result=service_totals([request],180.)
        self.assertEqual(result['service_vehicle_s'],180.)
        self.assertEqual(result['removed_unserved'],1)
        self.assertAlmostEqual(result['removed_unserved_vehicle_s'],14.3)
        basic=summarize([request],[],[],180.)
        self.assertTrue(basic['request_accounting_balances'])
        self.assertEqual((basic['terminal_backlog'],basic['unfinished_inside']),(1,0))

    def test_pending_kill_actor_still_in_snapshot_keeps_unserved_request(self):
        request=dict(id=0,arrival_s=0.,admitted_s=0.,completed_s=None)
        lost=dict(vehicle=Mock(id=10,is_alive=False),request=request)
        survivor=dict(vehicle=Mock(id=11,is_alive=True),request={'id':1})
        active={0:lost,1:survivor}
        # CARLA 0.9.16 TM removes dead-end actors after producing this snapshot.
        snapshot=SimpleNamespace(find=lambda actor: object())
        self.assertEqual(retire_missing_vehicles(active,snapshot,26.1,Mock()),[lost])
        self.assertEqual(active,{1:survivor})
        self.assertIsNone(request['completed_s'])
        self.assertEqual(service_totals([request],180.)['service_vehicle_s'],180.)
        lost['vehicle'].destroy.assert_not_called()

    def test_preparation_continues_after_actor_loss_and_keeps_unserved_request(self):
        from benchmark.runtime.sensor_history import prepare_physical_history
        request=dict(id=0,arrival_s=0.,admitted_s=0.,completed_s=None)
        vehicle=Mock(id=10,is_alive=False)
        active={0:dict(vehicle=vehicle,request=request)}
        snapshot=SimpleNamespace(find=lambda _:None)
        world=Mock();world.tick.side_effect=[1,2];world.get_snapshot.return_value=snapshot
        times=[]
        def after_tick(state,t):
            times.append(t)
            retire_missing_vehicles(active,state,0.,Mock())
        with patch('benchmark.runtime.sensor_history.submit_commands'):
            report=prepare_physical_history(Mock(),world,active,.05,Mock(),
                settling_s=.05,rolling_s=.05,after_tick=after_tick)
        self.assertEqual(report['last_preparation_frame'],2)
        self.assertEqual(report['states_at_time_zero'],[])
        self.assertEqual(times,[-.05,0.])
        vehicle.get_transform.assert_not_called()
        self.assertEqual(service_totals([request],180.)['service_vehicle_s'],180.)

    def test_prepared_tm_keeps_its_already_active_navigation_buffer(self):
        agent=Mock()
        self.assertEqual(align_prepared_background_routes({0:dict(is_visual=False,is_tm=True,agent=agent)}),[])
        agent.set_global_plan.assert_not_called()

    def test_falling_and_missing_state_cannot_inflate_road_distance(self):
        with TemporaryDirectory() as directory:
            folder=fixture(Path(directory),duration=5.)
            request=dict(id=0,actor_id=10,route='lane0',entry='in',arrival_s=0.,
                admitted_s=0.,completed_s=None,failure_reason='sustained_corridor_departure',
                removed_s=4.,removal_reason='simulator_actor_missing')
            write(folder/'requests.json',[request])
            cfg=json.loads((folder/'config.json').read_text());cfg['metric_scope']['ordinary_request_ids']=[0]
            write(folder/'config.json',cfg)
            (folder/'initial_states.jsonl').write_text('')
            def vehicle(x,z):
                v=car(0,x,z=z);v['road_surface_offset_m']=z
                return v
            frames=[dict(frame=i,time_s=float(i),dt_s=1.,vehicles=v) for i,v in enumerate([
                [vehicle(1,0)],[vehicle(2,0)],[vehicle(90,-100)],[],[]],1)]
            with patch('benchmark.metrics.evaluation.iter_frames',return_value=iter(frames)):
                result=evaluate(folder)
            self.assertAlmostEqual(result['coverage']['vehicle_km'],.001)
            self.assertEqual(result['coverage']['invalid_road_vehicle_s'],1.)
            self.assertEqual(result['diagnostics']['traffic']['service_vehicle_s'],5.)
            self.assertEqual(result['diagnostics']['traffic']['removed_unserved'],1)
            self.assertEqual(result['diagnostics']['series'][-1]['unserved'],1)
            self.assertEqual(result['diagnostics']['series'][-1]['active'],0)
            self.assertIsNone(result['diagnostics']['series'][2]['mean_speed_mps'])


if __name__=='__main__':unittest.main()
