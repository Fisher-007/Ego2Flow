"""Hand-computable exclusion of exit buffers without losing unfinished demand."""
import json
from pathlib import Path
import pytest
from shapely.geometry import Polygon,mapping
from benchmark.metrics.evaluation import evaluate
from benchmark.metrics.fixed_energy import estimate_fixed_cycle
from tests.test_evaluation_pipeline import fixture,write


def test_normal_exit_excludes_later_energy_braking_and_contacts(tmp_path):
    folder=fixture(tmp_path)
    req=json.loads((folder/'requests.json').read_text());req[0].update(completed_s=1.,exit_path='lane0')
    write(folder/'requests.json',req)
    rows=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()]
    for row in rows:
        if row['time_s']>1:row['vehicles'][0]['acceleration_mps2']=[-5,0,0]
        if .1<=row['time_s']<=.65:row['vehicles'][1]['acceleration_mps2']=[-5,0,0]
    (folder/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    events=[dict(time_s=1.5,request_id=i,actor_id=i+10,other_actor_id=100+i,
        other_actor_type='static.road',sensor_position_m=[27.5 if i==0 else 37.5,0,.1],impulse=[1,0,0]) for i in (0,1)]
    (folder/'collisions.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    e=evaluate(folder)
    for field in ('collision_rate_per_vehicle_km', 'hard_braking_rate_per_vehicle_km'):
        metric = e['metrics'][field]
        assert metric['events'] == 1
        assert metric['vehicle_km'] == pytest.approx(.015)
        assert metric['value'] == pytest.approx(1 / .015)
        assert metric['unit'] == 'events/veh-km'
    assert abs(e['coverage']['safety_vehicle_km']-.015)<1e-9
    assert e['coverage']['energy_vehicle_km']==e['coverage']['safety_vehicle_km']
    assert abs(e['diagnostics']['energy_scene']['vehicle_s']-3)<1e-9
    assert len(e['diagnostics']['recorded_collision_episodes'])==2


def test_spatial_exit_does_not_complete_failed_or_waiting_requests(tmp_path):
    folder=fixture(tmp_path)
    cfg=json.loads((folder/'config.json').read_text());cfg['metric_scope']['region_geojson']=mapping(Polygon([[-5,-5],[25,-5],[25,5],[-5,5]]))
    write(folder/'config.json',cfg)
    req=json.loads((folder/'requests.json').read_text());req[0].update(first_corridor_departure_s=1.,failure_reason='sustained_corridor_departure')
    write(folder/'requests.json',req)
    e=evaluate(folder)
    assert abs(e['coverage']['safety_vehicle_km']-.005)<1e-9
    assert e['diagnostics']['completed']==0
    assert 0 in e['diagnostics']['failed_requests']
    assert next(x for x in e['diagnostics']['requests'] if x['request_id']==2)['delay_s']==2
    assert abs(next(x for x in e['diagnostics']['requests'] if x['request_id']==0)['delay_s']-1.5)<1e-9


def test_unscored_context_preserves_soc_but_not_energy_totals():
    settings=json.loads((Path(__file__).parents[1]/'configs/eight_metrics.json').read_text())['energy']
    settings={**settings,'include_interval_details':True}
    full=estimate_fixed_cycle([0,1,2],[0,10,0],[0,0,0],settings,initial_soc=.97999)
    subset=estimate_fixed_cycle([0,1,2],[0,10,0],[0,0,0],settings,initial_soc=.97999,scoring_intervals=[False,True])
    assert subset['final_soc']==full['final_soc']
    assert abs(subset['diagnostic_demand_kwh']-full['interval_details'][1]['net_terminal_energy_kwh'])<1e-12
    assert subset['scored_elapsed_s']==1
    assert subset['traction_terminal_energy_kwh']==0
