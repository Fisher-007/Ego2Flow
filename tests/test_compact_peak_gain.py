import pytest
from benchmark.scenes.leader_pulse import target_speed_mps, evaluated_ids, validate
from benchmark.metrics.peak_gain import amplitude


def test_external_leader_is_not_a_homogeneous_evaluated_vehicle():
    requests=[dict(id=0,external_intervention=True,mixed_role='background'),
              dict(id=1,mixed_role='evaluated'),dict(id=2,mixed_role='background')]
    config=dict(deployment='H',request_manifest=requests)
    assert evaluated_ids(config,requests)==[1,2]
    config['deployment']='M'
    assert evaluated_ids(config,requests)==[1]


def test_pulse_is_bounded_continuous_and_recovers():
    p=dict(request_id=0,base_speed_mps=4.,speed_drop_fraction=.15,start_s=20.,deceleration_s=3.,hold_s=3.,recovery_s=3.)
    validate(p,[dict(id=0,external_intervention=True)])
    assert target_speed_mps(p,19)==4.
    assert target_speed_mps(p,23)==pytest.approx(3.4)
    assert target_speed_mps(p,26)==pytest.approx(3.4)
    assert target_speed_mps(p,29)==4.
    assert abs(target_speed_mps(p,20.001)-4)<1e-6
    assert 3.4<=target_speed_mps(p,21.5)<=4.


def test_peak_gain_uses_time_integral_and_does_not_substitute_missing_tail():
    a=amplitude([0,1,2,3,4],[4,4,4,3.4,4],[0,2],[2,4])
    assert a['loss_mps']==pytest.approx(.6)
    assert a['pre_sd_mps']==0
    assert amplitude([0,1,2,3],[4,4,4,3.4],[0,2],[2,4]) is None


def test_receiving_removal_uses_actual_legal_lane_and_rear_of_vehicle():
    from benchmark.scenes.receiving_boundary import beyond_receiving_gate
    gates=[dict(position_carla_m=[100,y,0],lane_center_forward_xy=[1,0],lane_width_m=3.5) for y in [0,3.5]]
    # Changing to the other permitted lane does not leave the completed car
    # running indefinitely; the entire car must still pass the receiving gate.
    assert beyond_receiving_gate([104,3.5],[1,0],[[101,2.5],[107,4.5]],gates)
    assert not beyond_receiving_gate([102,3.5],[1,0],[[99,2.5],[105,4.5]],gates)
    assert not beyond_receiving_gate([104,3.5],[-1,0],[[101,2.5],[107,4.5]],gates)


@pytest.mark.parametrize("public_export", [False, True])
def test_complete_peak_metric_tracks_ids_and_marks_real_chain_change(tmp_path, public_export):
    import json
    from tests.test_evaluation_pipeline import fixture, write
    from benchmark.metrics.peak_gain import evaluate_peak_gain
    folder=fixture(tmp_path,'propagation',8.)
    config=json.loads((folder/'config.json').read_text())
    config['peak_gain']=dict(leader_request_id=0,follower_request_ids=[1,2],
        pre_window_s=[1,3],response_window_s=[3,8],minimum_leader_loss_mps=.2,
        minimum_pre_speed_mps=1.,maximum_pre_speed_sd_mps=.6,maximum_chain_break_s=.3,
        minimum_tail_loss_mps=.05,minimum_after_pulse_s=2.)
    if public_export:
        config['peak_gain']={key:value for key,value in config['peak_gain'].items()
            if key in ('leader_request_id','follower_request_ids','pre_window_s','response_window_s')}
        config['peak_gain']['source_request_id']=config['peak_gain'].pop('leader_request_id')
    config['external_leader']=dict(request_id=0,start_s=3.,deceleration_s=1.,hold_s=1.,recovery_s=1.)
    write(folder/'config.json',config)
    rows=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
    for row in rows:
        for vehicle in row['vehicles']:
            loss=([.6,.3,.15][vehicle['request_id']] if row['time_s']==4 else 0.)
            vehicle['speed_mps']=4.-loss
        if row['frame']%2:row['vehicles'].reverse()
    (folder/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'scene/candidate_routes.json').unlink()
    peak=evaluate_peak_gain(folder)
    assert peak['metric']['status']=='measured'
    assert peak['metric']['value']==pytest.approx(.5)
    requests=json.loads((folder/'requests.json').read_text())
    for request in requests:
        request['actor_id']=next(v['actor_id'] for v in rows[0]['vehicles'] if v['request_id']==request['id'])
    write(folder/'requests.json',requests)
    (folder/'collisions.jsonl').write_text(json.dumps(dict(time_s=.5,actor_id=requests[0]['actor_id'],other_actor_id=requests[1]['actor_id']))+'\n')
    contact=evaluate_peak_gain(folder)
    assert contact['metric']['value']==pytest.approx(.5)
    assert 'contact_before_leader_pulse' in contact['qualifications']
    (folder/'collisions.jsonl').write_text('')
    for vehicle in rows[4]['vehicles']:
        if vehicle['request_id']==2:
            vehicle['position_m'][1]=8.
            for corner in vehicle['bounding_box_world_vertices_m']:corner[1]+=8.
    (folder/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    broken=evaluate_peak_gain(folder)
    assert broken['metric']['value']==pytest.approx(.5)
    assert 'following_chain_changed_or_not_established' in broken['qualifications']
    assert len(broken['curves'][2]['time_s'])==8
    # A fully observed zero tail response is damping, not missing data.
    for row in rows:
        for vehicle in row['vehicles']:
            if vehicle['request_id']==2:vehicle['speed_mps']=4.
    (folder/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    zero_tail=evaluate_peak_gain(folder)
    assert zero_tail['per_follower_gain'][2]==0
    assert zero_tail['metric']['value']==pytest.approx(.5)
    # Zero leader speed loss leaves the ratio undefined.
    for row in rows:
        for vehicle in row['vehicles']:
            if vehicle['request_id']==0:vehicle['speed_mps']=4.
    (folder/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert evaluate_peak_gain(folder)['metric']['value'] is None
    # A recorded exit buffer cannot supply a declared target's missing task window.
    requests[1].update(completed_s=6.,exit_path='lane0')
    write(folder/'requests.json',requests)
    exited=evaluate_peak_gain(folder)
    assert exited['curves'][1]['time_s'][-1]==6.
    assert 'fixed_response_window_ends_at_normal_task_exit' in exited['reasons']
