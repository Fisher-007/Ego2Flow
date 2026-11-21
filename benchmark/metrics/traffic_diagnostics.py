"""Collect auxiliary request accounting and traffic-state diagnostics."""


def request_failure_context(requests, retain_at_terminal=False):
    """Distinguish departures before completion from travel beyond the task exit."""
    after=[];failed=[]
    for r in requests:
        reason=r.get('failure_reason')
        if not reason:continue
        exit_s=r.get('observation_terminal_crossed_s');departure=r.get('first_corridor_departure_s')
        if (r.get('retain_at_terminal',retain_at_terminal) and reason=='sustained_corridor_departure'
                and exit_s is not None and departure is not None and exit_s<departure):
            after.append(dict(request_id=r['id'],legal_terminal_crossed_s=exit_s,
                recorded_departure_s=departure,raw_failure_reason=reason))
        else:failed.append(r['id'])
    return dict(failed_requests=failed,raw_failed_requests=[r['id'] for r in requests if r.get('failure_reason')],
                post_terminal_boundary_requests=after)


def traffic_summary(result, run, config, requests):
    """Auxiliary traffic observations alongside the eight primary columns."""
    start,end=result['window_s'];d=result['diagnostics'];coverage=result['coverage']
    after={r['request_id'] for r in d.get('post_terminal_boundary_requests',[])}
    requests=[{**r,'failure_reason':None} if r['id'] in after else r for r in requests]
    service=service_totals(requests,end)
    series=[dict(row,finished=row['completed']) for row in d['series']]
    return dict(**service,map_id=run.get('map_id'),map_version=run['map_version'],
        scene_id=config['scene_id'],setting=config['deployment'],seed=config['seed'],
        load=config.get('load_factor'),task=config.get('task'),
        status='complete' if result['observation_complete'] else 'incomplete_coverage',
        service_window_s=[0,end],measurement_window_s=[start,end],
        scope='Auxiliary ledger/traffic observations; current primary scores are evaluation.metrics.',
        series=series,queue_series=d['queue_series'],structure=d['structure'] or {},
        qkv=d['qkv'],qkv_bins=d['qkv_bins'],vehicle_km=coverage['vehicle_km'],
        following_exposure_s=result['metrics']['low_ttc_exposure_percent']['following_vehicle_s'],
        TTC_exposure_s=result['metrics']['low_ttc_exposure_percent']['low_ttc_vehicle_s'],
        Qout_veh_h=result['metrics']['throughput_veh_h']['value'],
        unserved_growth_veh_s=linear_slope([(r['time_s'],r['unserved']) for r in series if r['time_s']>=(start+end)/2]),
        service_by_entry={entry:service_totals([r for r in requests if r.get('entry')==entry],end)
            for entry in sorted({r.get('entry','') for r in requests})},
        predefined_background_service=service_totals([r for r in requests if r.get('deployment_role')=='background'],end),
        source_version=result['version'])


def service_totals(requests, horizon):
    due = [r for r in requests if r['arrival_s'] < horizon]
    completed = [r for r in due if r.get('completed_s') is not None
                 and r['completed_s'] <= horizon and not r.get('failure_reason')]
    finished_ids = {r['id'] for r in completed}
    admitted = [r for r in due if r.get('admitted_s') is not None and r['admitted_s'] <= horizon]
    removed = [r for r in admitted if r.get('removed_s') is not None and r['id'] not in finished_ids]
    burden = sum((r['completed_s'] if r['id'] in finished_ids else horizon)-r['arrival_s'] for r in due)
    outside = sum((r['admitted_s'] if r in admitted else horizon)-r['arrival_s'] for r in due)
    physical = sum(min(r.get('removed_s') or horizon,
                       r['completed_s'] if r['id'] in finished_ids else horizon)-r['admitted_s'] for r in admitted)
    return dict(requested=len(due), admitted=len(admitted), finished=len(completed),
        active_end=len(admitted)-len(completed)-len(removed), outside_end=len(due)-len(admitted),
        failed=sum(bool(r.get('failure_reason')) for r in due), removed_unserved=len(removed),
        service_vehicle_s=burden, service_s_per_request=burden/len(due) if due else None,
        completion_rate=len(completed)/len(due) if due else None,
        external_wait_vehicle_s=outside, physical_in_network_vehicle_s=physical,
        removed_unserved_vehicle_s=burden-outside-physical,
        accounting_balances=len(due)==len(completed)+(len(admitted)-len(completed)-len(removed))+(len(due)-len(admitted))+len(removed))


def edie_totals(distance_m, dwell_s, lane_length_m, duration_s):
    q = 3600*distance_m/(lane_length_m*duration_s)
    k = 1000*dwell_s/(lane_length_m*duration_s)
    v = 3.6*distance_m/dwell_s if dwell_s else None
    return dict(q_veh_h_lane=q, k_veh_km_lane=k, v_kmh=v,
                distance_m=distance_m, dwell_vehicle_s=dwell_s,
                lane_length_m=lane_length_m, duration_s=duration_s)


def linear_slope(points):
    if len(points)<2:
        return None
    xs, ys = zip(*points)
    mx,my = sum(xs)/len(xs),sum(ys)/len(ys)
    denominator=sum((x-mx)**2 for x in xs)
    return sum((x-mx)*(y-my) for x,y in points)/denominator if denominator else None
