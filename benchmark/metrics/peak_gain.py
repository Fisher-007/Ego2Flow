"""Compute observed peak speed-loss ratios from a single leader-pulse run."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from benchmark.scenes import scoring_scene
from benchmark.metrics.evaluation_geometry import EvaluationGeometry
from benchmark.runtime.log_io import iter_frames
from benchmark.metrics.road_validity import road_sample_valid
from benchmark.metrics.traffic_diagnostics import request_failure_context


# Diagnostic thresholds for interpreting the observed response.
PEAK_DIAGNOSTIC_DEFAULTS = dict(
    minimum_leader_loss_mps=.2, minimum_pre_speed_mps=1.,
    maximum_pre_speed_sd_mps=.6, maximum_chain_break_s=.3,
    minimum_tail_loss_mps=.05, minimum_after_pulse_s=20.)


def resolve_peak_spec(spec):
    resolved = {**PEAK_DIAGNOSTIC_DEFAULTS, **spec}
    if 'leader_request_id' not in resolved:
        resolved['leader_request_id'] = resolved.pop('source_request_id')
    return resolved


def window_samples(times, speeds, window):
    times, speeds = np.asarray(times), np.asarray(speeds)
    lo, hi = window
    if len(times) < 2 or times[0] > lo+1e-7 or times[-1] < hi-1e-7:
        return None
    middle = (times > lo) & (times < hi)
    t = np.r_[lo, times[middle], hi]
    v = np.r_[np.interp(lo,times,speeds), speeds[middle], np.interp(hi,times,speeds)]
    return t, v


def amplitude(times, speeds, pre_window, response_window):
    pre = window_samples(times,speeds,pre_window)
    response = window_samples(times,speeds,response_window)
    if pre is None or response is None:
        return None
    t,v = pre; mean = float(np.trapz(v,t)/(t[-1]-t[0]))
    variance = float(np.trapz((v-mean)**2,t)/(t[-1]-t[0]))
    rt,rv = response; low = int(np.argmin(rv))
    return dict(pre_mean_mps=mean,pre_sd_mps=variance**.5,
        minimum_response_mps=float(rv[low]),minimum_response_time_s=float(rt[low]),
        loss_mps=max(0.,mean-float(rv[low])))


def evaluate_peak_gain(run_dir, scene_dir=None, *, routes=None):
    run_dir=Path(run_dir)
    read=lambda p:json.loads(p.read_text())
    cfg=read(run_dir/'config.json');run=read(run_dir/'run.json')
    spec=resolve_peak_spec(cfg['peak_gain']); pulse=cfg['external_leader']
    if routes is None:
        _,routes,_=scoring_scene(run_dir,scene_dir)
    geometry=EvaluationGeometry(routes,cfg['evaluation'])
    chain=[spec['leader_request_id'],*spec['follower_request_ids']]
    requests=read(run_dir/'requests.json')
    explicit_scope=cfg.get('metric_scope') is not None or (run_dir/'metric_scope.json').exists()
    normal_exits={}
    if explicit_scope:
        for r in requests:
            if r['id'] in chain and r.get('exit_path') is not None:
                stop=r.get('completed_s')
                if stop is None:stop=r.get('observation_terminal_crossed_s')
                if stop is not None:normal_exits[r['id']]=stop
    tracks=defaultdict(lambda:dict(time_s=[],speed_mps=[]))
    breaks=defaultdict(float);longest=defaultdict(float);current_break=defaultdict(float)
    changes=[];last_leaders={};invalid=[]
    begin,end=spec['pre_window_s'][0],spec['response_window_s'][1]
    for row in iter_frames(run_dir):
        t=row['time_s'];dt=row['dt_s']
        if t>end+1e-7:break
        all_vehicles={v['request_id']:v for v in row['vehicles']}
        eligible=[v for v in row['vehicles'] if road_sample_valid(v)]
        associations={v['request_id']:geometry.associations(v) for v in eligible}
        relations=geometry.following(eligible,associations,.1)
        for ident in chain:
            v=all_vehicles.get(ident)
            if v is not None and t<=normal_exits.get(ident,float('inf'))+1e-8:
                tracks[ident]['time_s'].append(t);tracks[ident]['speed_mps'].append(v['speed_mps'])
                if begin<t<=end and (v.get('is_dormant') or not road_sample_valid(v)):
                    invalid.append(dict(time_s=t,request_id=ident,reason='inactive_or_invalid_road_state'))
        if not begin<t<=end:continue
        for predecessor,ident in zip(chain,chain[1:]):
            actual=relations.get(ident,{}).get('leader_request_id')
            if actual!=last_leaders.get(ident):
                changes.append(dict(time_s=t,request_id=ident,actual_leader_id=actual,expected_leader_id=predecessor))
            last_leaders[ident]=actual
            if actual!=predecessor:
                breaks[ident]+=dt;current_break[ident]+=dt
                longest[ident]=max(longest[ident],current_break[ident])
            else:current_break[ident]=0.
    amplitudes={i:amplitude(tracks[i]['time_s'],tracks[i]['speed_mps'],
        spec['pre_window_s'],spec['response_window_s']) for i in chain}
    # Retain observed responses and record collisions, lane changes and oscillations.
    reasons=[];qualifications=[]
    if any(stop<end-1e-8 for stop in normal_exits.values()):reasons.append('fixed_response_window_ends_at_normal_task_exit')
    if not run.get('checks',{}).get('observation_completed') or run['elapsed_s']<end-1e-7:
        reasons.append('response_window_incomplete')
    if any(v is None for v in amplitudes.values()):reasons.append('vehicle_window_incomplete')
    if invalid:qualifications.append('inactive_or_off_road_chain_vehicle_state')
    if any(t>spec['maximum_chain_break_s']+1e-7 for t in longest.values()):
        qualifications.append('following_chain_changed_or_not_established')
    for i,v in amplitudes.items():
        if v and (v['pre_mean_mps']<spec['minimum_pre_speed_mps'] or v['pre_sd_mps']>spec['maximum_pre_speed_sd_mps']):
            qualifications.append('pre_pulse_following_not_settled');break
    leader=amplitudes[chain[0]]
    if leader is not None and leader['loss_mps']<spec['minimum_leader_loss_mps']:
        qualifications.append('actual_leader_excitation_too_small')
    if leader is not None and leader['loss_mps']<=0:
        reasons.append('no_actual_leader_speed_loss')
    tail=amplitudes[chain[-1]]
    if tail and tail['loss_mps']<spec['minimum_tail_loss_mps']:
        qualifications.append('tail_response_below_diagnostic_threshold')
    pulse_end=pulse['start_s']+sum(pulse[k] for k in ('deceleration_s','hold_s','recovery_s'))
    if end-pulse_end<spec['minimum_after_pulse_s']:
        qualifications.append('short_post_pulse_observation')
    actor_ids={r['actor_id'] for r in read(run_dir/'requests.json') if r['id'] in chain and r.get('actor_id') is not None}
    contacts=[json.loads(line) for line in (run_dir/'collisions.jsonl').read_text().splitlines()]
    contacts=[e for e in contacts if 0<=e['time_s']<=end and actor_ids.intersection((e['actor_id'],e['other_actor_id']))]
    if any(e['time_s']<pulse['start_s'] for e in contacts):qualifications.append('contact_before_leader_pulse')
    if contacts:qualifications.append('contact_during_chain_observation')
    if run.get('collision_event_stream_complete') is False:
        qualifications.append('contact_interpretation_incomplete')
    boundary=request_failure_context(read(run_dir/'requests.json'),cfg.get('retain_platoon_at_terminal',False))
    if boundary['post_terminal_boundary_requests']:
        qualifications.append('retained_vehicles_crossed_finite_analysis_terminal')
    gains={i:(v['loss_mps']/leader['loss_mps'] if v and leader and leader['loss_mps']>0 else None)
        for i,v in amplitudes.items() if i!=chain[0]}
    diagnostic=max(gains.values()) if gains and all(v is not None for v in gains.values()) else None
    result=dict(definition='observed_peak_speed_loss_ratio',scene_id=cfg['scene_id'],deployment=cfg['deployment'],
        metric=dict(value=diagnostic if not reasons else None,unit='1',status='measured' if not reasons else 'single_run_response_unavailable',
                    actual_leader_speed_loss_mps=leader['loss_mps'] if leader is not None else None,
                    qualifications=list(dict.fromkeys(qualifications))),
        reasons=list(dict.fromkeys(reasons)),diagnostic_peak_gain=diagnostic,
        qualifications=list(dict.fromkeys(qualifications)),
        post_terminal_boundary_requests=boundary['post_terminal_boundary_requests'],
        amplitudes=amplitudes,per_follower_gain=gains,windows=spec,actual_chain_changes=changes,
        cumulative_chain_break_s=dict(breaks),longest_chain_break_s=dict(longest),contacts=contacts,
        invalid_state_count=len(invalid),curves=dict(tracks),
        normal_exit_times_s=normal_exits,
        scope='Observed fixed-ID speed-loss peak gain within the event and task windows. Normal completion ends scoring. Pre-existing oscillations, contacts and lane changes remain in the response and its qualifications.')
    (run_dir/'peak_gain.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
