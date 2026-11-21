"""Compute eight traffic-system metrics from recorded runs."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from benchmark import ROOT
from benchmark.scenes import scoring_scene
from benchmark.metrics.fixed_energy import estimate_fixed_cycle, summarize_scene_energy
from benchmark.metrics.evaluation_geometry import EvaluationGeometry, interpolated_state
from benchmark.runtime.log_io import iter_frames
from benchmark.metrics.road_validity import road_sample_valid
from benchmark.metrics.metric_scope import MetricScope
from benchmark.metrics.traffic_diagnostics import edie_totals, traffic_summary, request_failure_context

PROFILE = ROOT / 'configs/eight_metrics.json'
METRIC_COLUMNS = [
    ('speed_sd_mps','Speed SD','m/s'), ('amplification_ratio','Observed peak speed-loss ratio','1'),
    ('delay_s_per_request','Delay','s/request'), ('throughput_veh_h','Throughput','veh/h'),
    ('collision_rate_per_vehicle_km','Collision rate','events/veh-km'),
    ('low_ttc_exposure_percent','Low-TTC exposure','%'),
    ('hard_braking_rate_per_vehicle_km','Hard-braking rate','events/veh-km'),
    ('estimated_energy_kwh_per_100_vehicle_km','Est. energy','kWh/100 veh-km')]


def read(path):
    return json.loads(path.read_text())


def value(number, unit, status='measured', **details):
    return dict(value=number, unit=unit, status=status, **details)


class GroupedSpeed:
    def __init__(self):
        self.groups=defaultdict(lambda:[0.,0.,0.])

    def add(self, group, speed0, speed1, dt):
        item=self.groups[group]
        item[0]+=dt
        item[1]+=dt*(speed0+speed1)/2
        item[2]+=dt*(speed0*speed0+speed0*speed1+speed1*speed1)/3

    def result(self):
        groups={key:dict(vehicle_s=t,mean_mps=s/t,variance_mps2=max(0.,ss/t-(s/t)**2))
                for key,(t,s,ss) in self.groups.items() if t>0}
        total=sum(g['vehicle_s'] for g in groups.values())
        sd=math.sqrt(sum(g['vehicle_s']*g['variance_mps2'] for g in groups.values())/total) if total else None
        return sd,groups


def collision_episodes(events, geometry, window, contact_gap_s):
    episodes=[];last={};unlocated=0
    for event in sorted(events,key=lambda e:e['time_s']):
        if not window[0] <= event['time_s'] <= window[1]:continue
        position=event.get('sensor_position_m')
        if position is None and geometry is not None:
            unlocated+=1
            continue
        if geometry is not None and not geometry.domain.covers(Point(position[:2])):continue
        pair=tuple(sorted([event['actor_id'],event['other_actor_id']]))
        old=last.get(pair)
        if old is None or event['time_s']-old['end_s']>contact_gap_s:
            kind=event['other_actor_type']
            classification=('vehicle' if kind.startswith('vehicle.') else
                            'pedestrian' if kind.startswith('walker.') else 'static')
            old=dict(actor_pair=list(pair),start_s=event['time_s'],end_s=event['time_s'],
                position_m=position,kind=classification,raw_callbacks=0,maximum_impulse_ns=0.)
            episodes.append(old)
        old['end_s']=event['time_s'];old['raw_callbacks']+=1
        old['maximum_impulse_ns']=max(old['maximum_impulse_ns'],math.sqrt(sum(x*x for x in event['impulse'])))
        last[pair]=old
    return episodes,unlocated


class HardBraking:
    def __init__(self, settings):
        self.settings=settings;self.active={};self.events=[];self.shock_vehicle_s=0.

    def observe(self, vehicle, begin, end, collision_shock=False):
        # Project horizontal acceleration along the direction of motion.
        speed=math.hypot(*vehicle['velocity_mps'][:2])
        acceleration=(float(np.dot(vehicle['acceleration_mps2'][:2],vehicle['velocity_mps'][:2]))/speed
                      if speed>self.settings['hard_brake_minimum_speed_mps'] else None)
        if collision_shock:
            self.shock_vehicle_s+=end-begin
            return
        if acceleration is None or acceleration>=self.settings['hard_brake_threshold_mps2']:return
        key=vehicle['request_id'];event=self.active.get(key)
        if event is None or begin-event['end_s']>self.settings['hard_brake_merge_gap_s']+1e-8:
            event=dict(request_id=key,start_s=begin,end_s=begin,below_threshold_s=0.,
                continuous_below_s=0.,maximum_continuous_below_s=0.,minimum_acceleration_mps2=acceleration)
            self.active[key]=event;self.events.append(event)
        event['continuous_below_s']=(event['continuous_below_s'] if abs(begin-event['end_s'])<1e-7 else 0.)+end-begin
        event['maximum_continuous_below_s']=max(event['maximum_continuous_below_s'],event['continuous_below_s'])
        event['end_s']=end;event['below_threshold_s']+=end-begin
        event['minimum_acceleration_mps2']=min(event['minimum_acceleration_mps2'],acceleration)

    def result(self):
        return [e for e in self.events if e['maximum_continuous_below_s']+1e-8>=self.settings['hard_brake_minimum_duration_s']]


def request_delay(requests, window, progress_credit):
    start,end=window
    due=[r for r in requests if r['arrival_s']<end and
         (r.get('completed_s') is None or r['completed_s']>start)]
    rows=[]
    for r in due:
        stop=min(end,r['completed_s']) if r.get('completed_s') is not None else end
        elapsed=max(0.,stop-max(start,r['arrival_s']))
        credit=min(elapsed,max(0.,progress_credit.get(r['id'],0.)))
        rows.append(dict(request_id=r['id'],elapsed_s=elapsed,free_flow_progress_credit_s=credit,
            delay_s=elapsed-credit,admitted_s=r.get('admitted_s'),completed_s=r.get('completed_s')))
    return (sum(r['delay_s'] for r in rows)/len(rows) if rows else None),rows


def _grade(vehicle, associations):
    if vehicle.get('road_grade') is not None:return vehicle['road_grade']
    if not associations:return None
    nearest=min(associations.values(),key=lambda a:a['offset_m'])
    if not nearest['inside']:return None
    velocity=np.asarray(vehicle['velocity_mps'][:2]);speed=np.linalg.norm(velocity)
    direction=(velocity/speed if speed>.1 else np.asarray(vehicle['forward_xy']))
    return float(nearest['grade']*np.dot(direction,nearest['direction']))


def evaluate(run_dir, scene_dir=None):
    config=read(run_dir/'config.json');run=read(run_dir/'run.json')
    requests=read(run_dir/'requests.json')
    external={r['id'] for r in requests if r.get('external_intervention')}
    ordinary_actor_ids={r['actor_id'] for r in requests if r['id'] not in external and r.get('actor_id') is not None}
    requests=[r for r in requests if r['id'] not in external]
    request_by_id={r['id']:r for r in requests}
    source,supplied,measurements=scoring_scene(run_dir,scene_dir)
    scene_dir=run.get('scene_dir') if source.name=='input_receipt.json' else str(source.parent.resolve())
    profile=read(PROFILE)
    settings={**profile, **config.get('metric_settings', {}), 'version': profile['version']}
    current_energy=profile['energy']
    for key in ('eligibility','power_limit_policy','contact_sensitivity_interpretation'):
        settings['energy'][key]=current_energy[key]
    settings.pop('minimum_distance_km_for_rates',None)
    settings['energy'].pop('contact_sensitivity_limit_fraction',None)
    spec=config['evaluation']
    expected=spec['window_s'];window=[expected[0],min(expected[1],run['elapsed_s'])]
    if window[1]<=window[0]:raise ValueError('No observed evaluation window')
    duration=window[1]-window[0]
    scope_spec=read(run_dir/'metric_scope.json') if (run_dir/'metric_scope.json').exists() else config.get('metric_scope')
    if scope_spec is None:raise ValueError('Explicit metric_scope is required; recording extent is not scoring extent')
    scope=MetricScope(scope_spec,requests,window)
    requests=[r for r in requests if r['id'] in scope.ids]
    request_by_id={r['id']:r for r in requests}
    geometry=EvaluationGeometry(supplied,{**spec,'analysis_polygon_carla_xy_m':list(scope.domain.exterior.coords)})
    structure=None;queue_series=[]
    signals=read(run_dir/'signals.json') if (run_dir/'signals.json').exists() else None
    if measurements is not None:
        from benchmark.metrics.structure_measurements import StructureMeasurements
        for plane in (signals or {}).get('reference_planes',[]):
            for control in measurements.get(plane['route'],{}).get('controls',[]):
                if (control['signal_id'],control['road_id'],control['lane_id'])==(plane['signal_id'],plane['road_id'],plane['lane_id']):
                    control.update(station_m=plane['route_station_m'],position_carla_m=plane['position_m'])
        structure=StructureMeasurements({n:dict(line=p.line,polygon=p.polygon) for n,p in geometry.paths.items()},
            measurements,*window,config.get('measurement_route_aliases',{}))
    events=[json.loads(line) for line in (run_dir/'collisions.jsonl').read_text().splitlines()]
    if external:
        events=[e for e in events if ordinary_actor_ids.intersection((e['actor_id'],e['other_actor_id']))]
    classified_speed=spec.get('speed_population')=='declared_forward_traffic'
    raw_episodes,_=collision_episodes(events,None,window,settings['collision_contact_gap_s'])
    selected_events=[e for e in events if scope.event_selected(e) is True]
    episodes,_=collision_episodes(selected_events,None,window,settings['collision_contact_gap_s'])
    unlocated=sum(scope.event_selected(e) is None for e in events)
    shock=defaultdict(list)
    for e in raw_episodes:
        for actor in e['actor_pair']:
            shock[actor].append((e['start_s']-settings['collision_shock_margin_s'],e['end_s']+settings['collision_shock_margin_s']))
    energy_contacts=defaultdict(list)
    for e in events:
        for actor in {e['actor_id'],e['other_actor_id']}:
            energy_contacts[actor].append((e['time_s']-settings['collision_shock_margin_s'],e['time_s']+settings['collision_shock_margin_s']))
    births={}
    if (run_dir/'initial_states.jsonl').exists():
        for line in (run_dir/'initial_states.jsonl').read_text().splitlines():
            b=json.loads(line);births[b['vehicle']['request_id']]=b
    speed_stats=GroupedSpeed();braking=HardBraking(settings)
    speed_membership=defaultdict(lambda:dict(vehicle_s=0.,minimum_speed_mps=float('inf'),maximum_speed_mps=0.))
    credit=defaultdict(float);tracks=defaultdict(list);distance=dwell=following_time=low_time=0.
    missing_time=invalid_time=unclassified_time=0.;previous={};previous_t=None;previous_frame=None
    series=[];minimum_ttc=None;relationship_changes=[];last_leaders={}
    grouped_series=[];bins=defaultdict(lambda:[0.,0.]);role_time=defaultdict(float)
    inference_times=[]
    energy_distance=0.
    for row in iter_frames(run_dir):
        t=row['time_s'];dt=row['dt_s'];vehicles=row['vehicles']
        if t>window[1]+1e-8:break
        current={v['request_id']:v for v in vehicles}
        ordinary=[v for v in vehicles if v['request_id'] not in external]
        for v in ordinary:
            role_time[v.get('deployment_role','background')]+=max(0.,min(t,window[1])-max(t-dt,window[0]))
            policy=v.get('visual_policy') or {}
            # VLMs store inference latency under output and reuse plans between calls.
            if policy.get('model_called') is not False:
                inference_s=policy.get('inference_wall_s')
                if inference_s is None:
                    inference_s=(policy.get('output') or {}).get('inference_wall_s')
                if inference_s is not None:inference_times.append(inference_s)
        associations={k:geometry.associations(v) for k,v in current.items() if road_sample_valid(v)}
        eligible=[v for v in vehicles if v['request_id'] in associations and geometry.domain.covers(Point(v['position_m'][:2]))]
        if structure:
            # Queue/PET diagnostics follow the occupied lane after a legal
            # lane change, not the request's original route label.
            occupied=[]
            for v in eligible:
                choices=[x for x in associations[v['request_id']].values() if x['inside']]
                if choices:occupied.append({**v,'route':min(choices,key=lambda x:x['offset_m'])['path']})
            structure.observe({**row,'vehicles':occupied})
            if window[0]<t<=window[1]:queue_series.append(dict(time_s=t,queue_length_m=structure.latest_queue_length_m))
        leaders=geometry.following(eligible,associations,settings['closing_speed_epsilon_mps'])
        current_groups=defaultdict(list)
        for key,v in current.items():
            if key not in scope.ids:continue
            a=previous.get(key);t0=previous_t
            contiguous=previous_frame is not None and row['frame']==previous_frame+1
            if a is None and key in births:
                a=births[key]['vehicle'];t0=births[key]['time_s'];contiguous=True
            if a is None or not contiguous or a['actor_id']!=v['actor_id']:
                if scope.contains(key,t,v['position_m']):missing_time+=dt
                continue
            if t0>=t:continue
            # Split intervals at scoring boundaries; unscored intervals update SOC only.
            eligible_spans=scope.exposure(a,v,t0,t)
            begin=max(t0,window[0]);end=min(t,window[1])
            if end>begin:
                cuts=sorted({begin,end,*[t0+x*(t-t0) for lo,hi,_ in eligible_spans for x in (lo,hi)]})
                for left,right in zip(cuts,cuts[1:]):
                    if right<=left:continue
                    fraction=((left+right)/2-t0)/(t-t0)
                    scored=any(lo-1e-10<=fraction<=hi+1e-10 for lo,hi,_ in eligible_spans)
                    av=interpolated_state(a,v,(left-t0)/(t-t0));bv=interpolated_state(a,v,(right-t0)/(t-t0))
                    g0=_grade(av,geometry.associations(av));g1=_grade(bv,geometry.associations(bv))
                    av['speed_mps']*=math.sqrt(1+(g0 or 0.)**2);bv['speed_mps']*=math.sqrt(1+(g1 or 0.)**2)
                    horizontal=math.dist(av['position_m'][:2],bv['position_m'][:2])
                    moved=horizontal*math.sqrt(1+(((g0 or 0.)+(g1 or 0.))/2)**2)
                    if scored:energy_distance+=moved
                    own=tracks[key]
                    if not own or abs(own[-1][-1]['t']-left)>1e-7:
                        own.append([dict(t=left,v=av['speed_mps'],grade=g0,position_m=av['position_m'])])
                    expected_moved=(av['speed_mps']+bv['speed_mps'])*(right-left)/2
                    contact=any(l<right and h>=left for l,h in energy_contacts[v['actor_id']])
                    if scored:
                        contact_episode=any(l<right and h>=left for l,h in shock[v['actor_id']])
                        braking.observe(interpolated_state(a,v,fraction),left,right,contact_episode)
                    own[-1].append(dict(t=right,v=bv['speed_mps'],grade=g1,scored=scored,
                        collision_shock=contact,position_m=bv['position_m'],
                        position_velocity_inconsistent=not contact and abs(moved-expected_moved)>max(.1,.25*expected_moved)))
            if not road_sample_valid(a) or not road_sample_valid(v):
                invalid_time+=sum((hi-lo)*(t-t0) for lo,hi,_ in eligible_spans);continue
            aa=geometry.associations(a)
            for lo,hi,moved in scope.exposure(a,v,t0,t):
                begin=t0+lo*(t-t0);end=t0+hi*(t-t0);span=end-begin
                av=interpolated_state(a,v,lo);bv=interpolated_state(a,v,hi)
                mid=interpolated_state(a,v,(lo+hi)/2);ma=geometry.associations(mid)
                g0=_grade(av,geometry.associations(av));g1=_grade(bv,geometry.associations(bv))
                if g0 is not None and g1 is not None:
                    # Road grade supplies vertical travel, not body pitch/bounce.
                    av['speed_mps']*=math.sqrt(1+g0*g0);bv['speed_mps']*=math.sqrt(1+g1*g1)
                    moved*=math.sqrt(1+((g0+g1)/2)**2)
                group_candidates=[x for x in ma.values() if x['inside']]
                if classified_speed:
                    member,reason=geometry.speed_membership(mid,ma)
                    count=speed_membership[reason];count['vehicle_s']+=span
                    count['minimum_speed_mps']=min(count['minimum_speed_mps'],mid['speed_mps'])
                    count['maximum_speed_mps']=max(count['maximum_speed_mps'],mid['speed_mps'])
                    if member is not None:
                        g=member['group'];speed_stats.add(g,av['speed_mps'],bv['speed_mps'],span)
                        current_groups[g].append(mid['speed_mps'])
                elif group_candidates:
                    own=supplied[v['route']]['evaluation']['legal_route_names']
                    preferred=[x for x in group_candidates if x['path'] in own]
                    group_candidates=preferred or group_candidates
                    g=min(group_candidates,key=lambda x:x['offset_m'])['group']
                    speed_stats.add(g,av['speed_mps'],bv['speed_mps'],span)
                    current_groups[g].append(mid['speed_mps'])
                else:unclassified_time+=span
                # Accumulate distance and time for valid road samples.
                distance+=moved;dwell+=span
                for index in range(int((begin-window[0])//10),int((end-window[0]-1e-9)//10)+1):
                    share=max(0.,min(end,window[0]+(index+1)*10)-max(begin,window[0]+index*10))
                    bins[index][0]+=moved*share/span;bins[index][1]+=share
                progress,_=geometry.legal_progress(mid,ma)
                completed=request_by_id[key].get('completed_s')
                credit[key]+=max(0.,min(end,completed if completed is not None else end)-begin)*progress
                if key in leaders:
                    following_time+=span
                    relation=leaders[key]
                    if math.isfinite(relation['ttc_s']):
                        minimum_ttc=relation['ttc_s'] if minimum_ttc is None else min(minimum_ttc,relation['ttc_s'])
                        low_time+=span*(relation['ttc_s']<settings['ttc_threshold_s'])
                    new_leader=relation['leader_request_id']
                    if last_leaders.get(key)!=new_leader:
                        relationship_changes.append(dict(time_s=t,request_id=key,leader_request_id=new_leader,path=relation['path']))
                        last_leaders[key]=new_leader
                else:last_leaders.pop(key,None)
        if window[0]<t<=window[1] and (not series or t-series[-1]['time_s']>=1-1e-7 or abs(t-window[1])<1e-7):
            due=[r for r in requests if r['arrival_s']<=t and r['arrival_s']<window[1]]
            complete=sum(r.get('completed_s') is not None and r['completed_s']<=t for r in due)
            series.append(dict(time_s=t,requested=len(due),completed=complete,unserved=len(due)-complete,
                outside=sum(r.get('admitted_s') is None or r['admitted_s']>t for r in due),
                active=len(vehicles),stopped=sum(v['speed_mps']<.5 for v in eligible),
                queue_length_m=structure.latest_queue_length_m if structure else None,
                mean_speed_mps=float(np.mean([v['speed_mps'] for v in eligible])) if eligible else None))
            grouped_series.append(dict(time_s=t,group_mean_speed_mps={g:float(np.mean(s)) for g,s in current_groups.items()}))
        previous=current;previous_t=t;previous_frame=row['frame']
    delay,delay_rows=request_delay(requests,window,credit)
    sd,groups=speed_stats.result();brake_events=braking.result();vehicle_km=distance/1000
    safety_vehicle_km=energy_distance/1000
    enough_distance=safety_vehicle_km>0
    complete=[r for r in requests if r.get('completed_s') is not None and window[0]<r['completed_s']<=window[1]
              and r.get('exit_path') is not None]
    energy_vehicle_km=energy_distance/1000
    energy_rows=[];energy_total=0.;energy_complete=True;energy_sampling_status='checked'
    fixed_energy=settings['energy'].get('method')=='fixed_trajectory_interval_work'
    for key,segments in tracks.items():
        soc=settings['energy']['initial_soc']
        for segment in segments:
            if not any(p.get('scored',False) for p in segment[1:]):continue
            last_scored=max(i for i,p in enumerate(segment) if p.get('scored',False))
            segment=segment[:last_scored+1]  # Later exit context cannot affect earlier battery state.
            if any(p['grade'] is None for p in segment):
                item=dict(status='road_grade_unavailable',energy_kwh=None)
            elif fixed_energy:
                try:
                    item=estimate_fixed_cycle([p['t'] for p in segment],[p['v'] for p in segment],
                        [p['grade'] for p in segment],settings['energy'],initial_soc=soc,
                        collision_intervals=[p.get('collision_shock',False) for p in segment[1:]],
                        anomalies=[p.get('position_velocity_inconsistent',False) for p in segment[1:]],
                        positions=[p['position_m'] for p in segment],scoring_intervals=[p['scored'] for p in segment[1:]])
                except ModuleNotFoundError:item=dict(status='energy_dependency_missing',energy_kwh=None)
            else:
                raise ValueError('The release scorer requires fixed_trajectory_interval_work')
            item.update(request_id=key,start_s=segment[0]['t'],end_s=segment[-1]['t'])
            energy_rows.append(item)
            if item.get('final_soc') is not None:soc=item['final_soc']
            if item['energy_kwh'] is None:energy_complete=False
            else:
                energy_total+=item['energy_kwh']
    energy_complete=(energy_complete and bool(energy_rows) and missing_time<1e-7
                     and run.get('collision_event_stream_complete') is not False)
    energy_status=('estimated' if energy_sampling_status=='checked' else 'estimated_'+energy_sampling_status)
    if energy_vehicle_km<=0:energy_status='zero_distance_report_total_energy'
    elif not energy_complete:
        failures={e['status'] for e in energy_rows if e['energy_kwh'] is None}
        energy_status=(next(iter(failures)) if len(failures)==1
            and missing_time<1e-7 else 'incomplete_or_invalid_energy_calculation')
    throughput_applicable=spec['task']!='propagation'
    metrics={
        'speed_sd_mps':value(sd,'m/s','measured' if sd is not None else 'no_grouped_traffic'),
        'amplification_ratio':value(None,'1','single_run_configuration_required' if spec['task']=='propagation' else 'not_applicable'),
        'delay_s_per_request':value(delay,'s/request','measured' if delay is not None else 'no_due_requests',requests=len(delay_rows)),
        'throughput_veh_h':value(3600*len(complete)/duration if throughput_applicable else None,'veh/h',
            'measured' if throughput_applicable else 'not_applicable_short_platoon',completed=len(complete)),
        'collision_rate_per_vehicle_km':value(len(episodes)/safety_vehicle_km if enough_distance and not unlocated else None,
            'events/veh-km','measured' if enough_distance and not unlocated else 'insufficient_distance_or_event_location',events=len(episodes),vehicle_km=safety_vehicle_km),
        'low_ttc_exposure_percent':value(100*low_time/following_time if following_time else 0.,'%',
            following_vehicle_s=following_time,low_ttc_vehicle_s=low_time,
            qualifications=[] if following_time else ['no_following_exposure_zero_convention']),
        'hard_braking_rate_per_vehicle_km':value(len(brake_events)/safety_vehicle_km if enough_distance else None,
            'events/veh-km','measured' if enough_distance else 'insufficient_distance',events=len(brake_events),vehicle_km=safety_vehicle_km),
        'estimated_energy_kwh_per_100_vehicle_km':value(100*energy_total/energy_vehicle_km if energy_vehicle_km>0 and energy_complete else None,
            'kWh/100 veh-km',energy_status,
            sampling_check='original_intervals_no_resampling' if fixed_energy else energy_sampling_status)}
    if run.get('collision_event_stream_complete') is False:
        # Retain observed event counts with incomplete-coverage status.
        metrics['collision_rate_per_vehicle_km']['status']='collision_event_stream_incomplete'
    if config.get('peak_gain'):
        from benchmark.metrics.peak_gain import evaluate_peak_gain
        peak=evaluate_peak_gain(run_dir,routes=supplied)
        metrics['amplification_ratio']={**peak['metric'], 'definition':peak['definition'], 'reasons':peak['reasons']}
    observation_complete=bool(run.get('checks',{}).get('observation_completed')) and run['elapsed_s']>=expected[1]-1e-8
    coverage_issues=[]
    if not observation_complete:coverage_issues.append('observation_window_incomplete')
    if missing_time>1e-7:coverage_issues.append('missing_state_intervals')
    if invalid_time>1e-7:coverage_issues.append('invalid_road_intervals')
    if unclassified_time>1e-7:coverage_issues.append('ungrouped_vehicle_exposure')
    for metric in metrics.values():
        if metric['value'] is not None and coverage_issues:
            metric['calculation_status']=metric['status']
            metric['coverage_issues']=coverage_issues
            metric['status']='observed_with_coverage_limits'
    result=dict(version=settings['version'],scene_id=config['scene_id'],task=spec['task'],
        deployment=config['deployment'],seed=config['seed'],map_version=run['map_version'],
        run=str(run_dir.resolve()),scene_dir=scene_dir,route_source=str(source.resolve()),window_s=window,expected_window_s=expected,
        observation_complete=observation_complete,coverage_issues=coverage_issues,metrics=metrics,settings=settings,
        coverage=dict(vehicle_km=vehicle_km,safety_vehicle_km=safety_vehicle_km,energy_vehicle_km=energy_vehicle_km,vehicle_s=dwell,missing_vehicle_s=missing_time,
            invalid_road_vehicle_s=invalid_time,unclassified_group_vehicle_s=unclassified_time,
            unlocated_collision_callbacks=unlocated,energy_complete=energy_complete),
        diagnostics=dict(metric_scope=scope_spec,recorded_collision_episodes=raw_episodes,scored_collision_callbacks=len(selected_events),requests=delay_rows,completed=len(complete),scheduled=len(requests),
            completion_fraction=len(complete)/len(delay_rows) if delay_rows else None,
            **request_failure_context(requests,config.get('retain_platoon_at_terminal',False)),
            collision_episodes=episodes,hard_braking_events=brake_events,
            collision_shock_vehicle_s_excluded_from_braking=braking.shock_vehicle_s,
            minimum_ttc_s=minimum_ttc,following_relationship_changes=relationship_changes,
            speed_groups=groups,speed_membership=dict(speed_membership),energy_segments=energy_rows,
            estimated_energy_kwh_observed=energy_total if energy_complete else None,
            energy_valid_segments_kwh=energy_total,
            energy_coverage=dict(vehicle_ids=sorted({e['request_id'] for e in energy_rows}),recorded_ordinary_vehicle_ids=sorted(tracks),segments=len(energy_rows),
                observed_vehicle_s=sum(e.get('scored_elapsed_s',e['end_s']-e['start_s']) for e in energy_rows),
                state_data_complete=bool(energy_rows) and missing_time<1e-7
                    and not any(e['status']=='road_grade_unavailable' for e in energy_rows),
                invalid_segments=sum(e['energy_kwh'] is None for e in energy_rows),
                diagnostic_total_demand_kwh=(sum(e['diagnostic_demand_kwh'] for e in energy_rows)
                    if energy_rows and all('diagnostic_demand_kwh' in e for e in energy_rows) else None)),
            qkv=(dict(q_veh_h_lane=3600*distance/(geometry.lane_length_m*duration),
                k_veh_km_lane=1000*dwell/(geometry.lane_length_m*duration),
                v_kmh=3.6*distance/dwell if dwell else None) if geometry.lane_length_m else None),
            qkv_bins=[dict(start_s=window[0]+i*10,end_s=min(window[1],window[0]+(i+1)*10),
                **edie_totals(*bins[i],geometry.lane_length_m,min(10.,duration-i*10)))
                for i in range(math.ceil(duration/10))] if geometry.lane_length_m else [],
            role_vehicle_s=dict(role_time),resources=run.get('resources',{}),
            inference=dict(count=len(inference_times),total_s=sum(inference_times),
                mean_s=float(np.mean(inference_times)) if inference_times else None),
            series=series,grouped_speed_series=grouped_series,queue_series=queue_series,
            structure=structure.result() if structure else None,signals=signals),
        conventions=dict(spatial_exposure='Vehicle reference point; adjacent XY trajectories clipped to the declared polygon and corrected by road grade for surface travel. Body heave is not road distance. Missing grade retains planar distance with incomplete energy coverage.',
            delay='All due requests, including unadmitted/unfinished; only legal positive progress gives capped free-flow credit.',
            energy='Declared ordinary population, region and scoring clock including stopped auxiliary use. Outside-scope context only carries SOC; normal exit buffer is not scored. No fictitious energy for unadmitted requests.',
            following='Minimum longitudinal TTC over occupied declared lane paths, one denominator per vehicle/time. Crossings and stacked roads are not longitudinal leaders.'))
    result['diagnostics']['traffic']=traffic_summary(result,run,config,requests)
    if fixed_energy:
        result['diagnostics']['energy_scene']=summarize_scene_energy(energy_rows,energy_vehicle_km,
            result['diagnostics']['energy_coverage']['state_data_complete'],
            collision_stream_complete=run.get('collision_event_stream_complete') is not False)
        scene=result['diagnostics']['energy_scene']
        if scene.get('contact_qualified_estimate'):
            result['coverage']['energy_complete']=True
            result['diagnostics']['estimated_energy_kwh_observed']=scene['net_kwh']
            result['metrics']['estimated_energy_kwh_per_100_vehicle_km'].update(
                value=scene['kwh_per_100_vehicle_km'],status='estimated_contact_qualified',
                calculation_status='estimated_contact_qualified',qualifications=scene['qualifications'],
                contact_sensitivity=scene['contact_sensitivity'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--scene-dir',type=Path)
    args=parser.parse_args();result=evaluate(args.run,args.scene_dir)
    (args.run/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(scene=result['scene_id'],complete=result['observation_complete'],
        metrics={k:v['value'] for k,v in result['metrics'].items()})))


if __name__=='__main__':main()
