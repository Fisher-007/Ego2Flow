"""Check directed-route completion from recorded trajectories."""
import argparse
import json
import math
from pathlib import Path


def evaluate(requests, frames, points, end_time, task=None):
    def end_tangent(route_points):
        b = route_points[-1]['position_carla_m'][:2]
        # Adjacent lane sections can share the final waypoint.
        for point in reversed(route_points[:-1]):
            a = point['position_carla_m'][:2]
            dx, dy = b[0]-a[0], b[1]-a[1]
            length = math.hypot(dx, dy)
            if length > 0:
                return (dx/length, dy/length)
        raise ValueError('Route has no nonzero XY segment for its endpoint direction')

    # A vehicle can keep driving beyond the task exit after completion.
    completion_times = {r['id']: r['completed_s'] for r in requests
                        if r.get('completed_s') is not None and r['completed_s'] <= end_time}
    exits = {v['request_id']: v for row in frames for v in row['vehicles']
             if v['request_id'] in completion_times
             and abs(row['time_s'] - completion_times[v['request_id']]) < 1e-7}
    outcomes = []
    for r in requests:
        if r['arrival_s'] >= end_time:
            continue
        tangent = end_tangent(points[r['route']] if isinstance(points, dict) else points)
        reached = r.get('completed_s') is not None and r['completed_s'] <= end_time
        corridor_task = task == 'road_corridor_and_exit'
        departure = r.get('first_corridor_departure_s' if corridor_task else 'first_off_route_s')
        departure = departure if departure is not None and departure <= end_time else None
        exit_state = exits.get(r['id'])
        dot = ((exit_state.get('corridor') or {}).get('exit_forward_dot') if corridor_task else
               sum(x*y for x,y in zip(exit_state['forward_xy'], tangent))) if reached and exit_state else None
        passed = reached and departure is None and dot is not None and dot > 0
        outcomes.append({'request_id':r['id'], 'route':r.get('route'), 'endpoint_reached':reached,
            'first_recorded_departure_s':departure, 'endpoint_forward_dot':dot, 'task':task or 'supplied_exact_lane',
            'exit_path':r.get('exit_path'),
            'exit_state_missing':reached and dot is None,
            'passed_bounded_route_check':passed,
            'request_to_route_check_pass_or_cutoff_s':(r['completed_s'] if passed else end_time)-r['arrival_s']})
    passed = sum(r['passed_bounded_route_check'] for r in outcomes)
    return {'scope': ('Enumerated legal source paths and forward road-exit crossing; sustained center/direction departure is invalid. Collision and signals scored separately.'
                      if task == 'road_corridor_and_exit' else
                      'Check directed-route completion from recorded trajectories.\n\n'
                      'Use sustained departure (>3 m for 1 s) and forward-facing endpoint arrival.\n'),
        'observed_end_s':end_time, 'due_requests':len(outcomes),
        'physical_endpoint_exits':sum(r['endpoint_reached'] for r in outcomes),
        'endpoint_exits_failing_route_check':sum(r['endpoint_reached'] and not r['passed_bounded_route_check'] for r in outcomes),
        'passed_bounded_route_check':passed, 'unresolved_route_tasks_at_cutoff':len(outcomes)-passed,
        'request_to_route_check_pass_or_cutoff_vehicle_seconds':sum(r['request_to_route_check_pass_or_cutoff_s'] for r in outcomes),
        'cost_interpretation':'An invalid endpoint exit does not satisfy the task: its unresolved demand retains age until cutoff. This is task accounting, not simulated vehicle occupancy after removal.',
        'limitations':['Bounded corridor/path geometry checks do not establish all traffic-rule compliance; collisions and signal crossings are scored separately.'],
        'requests':outcomes}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--scene-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    read=lambda p:json.loads(p.read_text())
    config=read(args.run/'config.json');metadata=read(args.run/'run.json')
    frames=[json.loads(s) for s in (args.run/'frames.jsonl').read_text().splitlines()]
    points={name: route['validated_waypoints'] for name,route in read(args.scene_dir/'candidate_routes.json').items()
            if 'validated_waypoints' in route}
    if 'demand_streams' not in config:
        points=points[config['route']]
    report=evaluate(read(args.run/'requests.json'),frames,points,metadata['elapsed_s'], task=config.get('navigation_task'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='requests'},indent=2))


if __name__=='__main__':main()
