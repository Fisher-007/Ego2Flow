"""Measure queues, crossing conflicts and signal discharge."""
from collections import defaultdict
import math

from shapely.geometry import MultiPoint, Point


class StructureMeasurements:
    def __init__(self, routes, measurements, start, end, route_aliases=None):
        self.routes = routes
        self.route_aliases = route_aliases or {}
        self.measurements = measurements
        self.start, self.end = start, end
        self.queue_applicable = any(m['controls'] for m in measurements.values())
        self.previous = {}
        self.crossings = []
        self.queue_bins = defaultdict(lambda: defaultdict(lambda: [0., 0., 0., 0.]))
        self.control_totals = defaultdict(lambda: [0., 0., 0.])
        self.latest_queue_length_m = 0.
        self.conflicts = []
        names = list(routes)
        for i, a in enumerate(names):
            for b in names[i+1:]:
                intersection = routes[a]['line'].intersection(routes[b]['line'])
                points = ([intersection] if intersection.geom_type == 'Point' else
                          list(intersection.geoms) if intersection.geom_type == 'MultiPoint' else [])
                for p in points:
                    # Use the fixed conflict zone assigned to this crossing.
                    self.conflicts.append(dict(routes=(a, b), center=[p.x, p.y],
                        zone=p.buffer(2.5), active={}, visits=[]))

    def observe(self, frame):
        t, dt = frame['time_s'], frame['dt_s']
        measured = self.start < t <= self.end
        self.latest_queue_length_m = None
        on_route = defaultdict(list)
        for v in frame['vehicles']:
            route_name = self.route_aliases.get(v['route'],v['route'])
            route = self.routes[route_name]
            pos = Point(*v['position_m'][:2])
            if not route['polygon'].covers(pos):
                continue
            station = route['line'].project(pos)
            a = route['line'].interpolate(max(0., station-.2))
            b = route['line'].interpolate(min(route['line'].length, station+.2))
            dx, dy = b.x-a.x, b.y-a.y
            norm = math.hypot(dx, dy)
            dx, dy = dx/norm, dy/norm
            if dx*v['forward_xy'][0]+dy*v['forward_xy'][1] <= 0:
                continue
            body = [(p[0]-pos.x)*dx+(p[1]-pos.y)*dy for p in v['bounding_box_world_vertices_m']]
            rear, front = station+min(body), station+max(body)
            item = dict(vehicle=v, station=station, front=front, rear=rear)
            on_route[route_name].append(item)
            prior = self.previous.get(v['request_id'])
            if measured and prior is not None:
                for j, s in enumerate(self.measurements[route_name]['detector_stations_m']):
                    if prior < s <= front:
                        self.crossings.append(dict(time_s=t, route=route_name, detector=j,
                            station_m=s, request_id=v['request_id'], speed_mps=v['speed_mps']))
            self.previous[v['request_id']] = front
        if measured:
            self.latest_queue_length_m = 0. if self.queue_applicable else None
            index = int((t-self.start-1e-8)//10)
            for name, measure in self.measurements.items():
                controls = sorted(measure['controls'], key=lambda c:c['station_m'])
                queues = []
                for control in controls:
                    s = control['station_m']
                    upstream = sorted([v for v in on_route[name]
                        if v['vehicle']['speed_mps'] < .5 and v['front'] <= s+2.],
                        key=lambda v:v['front'], reverse=True)
                    chain = []
                    rear = s
                    for v in upstream:
                        if rear-v['front'] > (15. if not chain else 8.):
                            break
                        chain.append(v)
                        rear = v['rear']
                    queues.append(dict(control=control, rear=rear, count=len(chain),
                        length=max(0., s-rear) if chain else 0.))
                intervals=sorted((q['rear'],q['control']['station_m']) for q in queues if q['count'])
                right=-math.inf
                for left,end in intervals:
                    self.latest_queue_length_m+=max(0.,end-max(left,right))
                    right=max(right,end)
                for j, q in enumerate(queues):
                    key = name+':'+str(q['control']['signal_id'])
                    spillback = j>0 and q['count']>0 and q['rear'] <= controls[j-1]['station_m']
                    green = frame['lights'].get(str(q['control']['signal_id'])) == 'Green'
                    demand=any(-2.<=q['control']['station_m']-v['front']<=20.
                               for v in on_route[name])
                    blocked = (green and q['count']>0 and j+1<len(queues)
                               and queues[j+1]['count']>0
                               and queues[j+1]['rear'] <= q['control']['station_m']+8.)
                    totals = self.control_totals[key]
                    totals[0] += dt*spillback
                    totals[1] += dt*(green and demand)
                    totals[2] += dt*blocked
                    row = self.queue_bins[index][key]
                    row[0] += q['length']*dt
                    row[1] = max(row[1], q['length'])
                    row[2] += dt*(q['count']>0 and q['rear']<=5.)
                    row[3] += dt
        for conflict in self.conflicts:
            present = {}
            for v in frame['vehicles']:
                if v['route'] in conflict['routes']:
                    body = MultiPoint([p[:2] for p in v['bounding_box_world_vertices_m']]).convex_hull
                    if body.intersects(conflict['zone']):
                        present[v['request_id']] = v
            for key, v in present.items():
                if key not in conflict['active']:
                    conflict['active'][key] = dict(request_id=key, route=v['route'], entry_s=t, exit_s=None)
            for key in list(conflict['active']):
                if key not in present:
                    visit = conflict['active'].pop(key)
                    visit['exit_s'] = t
                    conflict['visits'].append(visit)

    def result(self):
        pet_events, visits_count, overlaps = [], 0, 0
        cross_route_exposure=False
        conflicts = []
        for conflict in self.conflicts:
            visits = sorted(conflict['visits']+list(conflict['active'].values()),key=lambda v:v['entry_s'])
            measured = [v for v in visits if self.start < v['entry_s'] <= self.end]
            visits_count += len(measured)
            cross_route_exposure |= len({v['route'] for v in measured})>1
            overlap_pairs=set()
            for v in measured:
                others = [o for o in visits if o['route']!=v['route'] and o['entry_s']<=v['entry_s']]
                occupied = [o for o in others if o['exit_s'] is None or o['exit_s']>v['entry_s']]
                if occupied:
                    overlap_pairs.update(tuple(sorted((v['request_id'],o['request_id']))) for o in occupied)
                    continue
                prior = [o for o in others if o['exit_s'] is not None]
                if prior:
                    last = max(prior,key=lambda o:o['exit_s'])
                    pet_events.append(dict(earlier_request=last['request_id'],later_request=v['request_id'],
                        time_s=v['entry_s'],PET_s=v['entry_s']-last['exit_s'],center_m=conflict['center']))
            overlaps+=len(overlap_pairs)
            conflicts.append(dict(routes=conflict['routes'],center_m=conflict['center'],radius_m=2.5,visits=visits))
        return dict(
            signal_queue_applicable=self.queue_applicable,
            detectors=self.crossings,
            queue_bins=[dict(start_s=self.start+10*i,route_signal=key,
                mean_queue_length_m=v[0]/v[3],max_queue_length_m=v[1],
                queue_reaching_entry_s=v[2]) for i,rows in sorted(self.queue_bins.items()) for key,v in rows.items()],
            signals={key:dict(spillback_s=v[0],green_with_approach_demand_s=v[1],
                blocked_green_s=v[2],blocked_green_fraction=v[2]/v[1] if v[1] else None)
                for key,v in self.control_totals.items()},
            queue_definition='Speed below 0.5 m/s; connected upstream body intervals, first front within 15 m of stop plane, successive net gaps at most 8 m. Spillback crosses the previous stop plane; entry contact flags potentially truncated storage.',
            blocked_green_definition='Green with a stopped upstream queue and downstream queue tail within 8 m of its stop plane. Denominator: green frames with a vehicle front in the upstream 20 m approach, including moving vehicles.',
            controls={name:m['controls'] for name,m in self.measurements.items()},
            PET_status=('measured' if pet_events else 'no_sequential_crossing_passages' if cross_route_exposure else 'insufficient_crossing_exposure') if self.conflicts else 'not_applicable_no_crossing_paths',
            PET_events=pet_events, conflict_entries=visits_count,
            low_PET_events_per_1000_conflict_entries=1000*sum(e['PET_s']<1.5 for e in pet_events)/visits_count if pet_events else None,
            simultaneous_conflict_occupations=overlaps,
            conflicts=conflicts,
            PET_definition='Body occupancy of a 2.5 m radius zone around each declared centerline crossing; previous different-route clearance to entry. Simultaneous zone occupancy is separate and is not classified as a collision.')
