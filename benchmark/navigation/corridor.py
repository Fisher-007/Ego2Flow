"""Track directed routes and forward exit gates."""
from bisect import bisect_left
import math
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union
from benchmark.navigation.route_lane import lane_corridor


class Corridor:
    def __init__(self, data):
        self.multilane = data.get('task') == 'multilane_movement_and_terminal'
        self.entry_path = data.get('entry_path')
        self.rear_terminal = data.get('terminal_reference') == 'rear'
        self.margin = data['center_margin_m']
        self.departure_s = data['sustained_departure_s']
        self.paths = []
        for path in data['paths']:
            points = path['validated_waypoints']
            line, polygon, stations, _, _ = lane_corridor(points)
            self.paths.append({'id': path['id'], 'points': points, 'line': line,
                'polygon': polygon, 'tolerated': polygon.buffer(self.margin), 'stations': stations})
        self.polygon = unary_union([p['polygon'] for p in self.paths])


class CorridorTracker:
    def __init__(self, corridor):
        self.corridor = corridor
        self.candidates = {p['id'] for p in corridor.paths}
        self.outside_s = {p['id']: 0. for p in corridor.paths}
        self.previous_gate = {}
        self.active_path = corridor.entry_path
        self.departed = False
        self.union_outside_s = 0.

    def update(self, vehicle, dt):
        center = Point(*vehicle['position_m'][:2])
        forward = vehicle['forward_xy']
        exit_id = exit_dot = None
        raw = []
        current_points = {}
        for path in self.corridor.paths:
            ident = path['id']
            station = path['line'].project(center)
            i = min(bisect_left(path['stations'], station), len(path['points']) - 1)
            direction = path['points'][i]['lane_center_forward_xy']
            current_points[ident] = path['points'][i]
            norm = math.hypot(*direction)
            dot = sum(a*b for a,b in zip(direction, forward))/norm
            # Keep the terminal cross-section open longitudinally for the
            # frame that crosses it; lateral and direction checks still apply.
            goal = path['points'][-1]
            gx, gy = goal['position_carla_m'][:2]
            fx, fy = goal['lane_center_forward_xy']
            norm = math.hypot(fx, fy); fx /= norm; fy /= norm
            dx, dy = center.x-gx, center.y-gy
            signed = dx*fx+dy*fy
            if self.corridor.rear_terminal:
                signed = min((p[0]-gx)*fx+(p[1]-gy)*fy for p in vehicle['bounding_box_world_vertices_m'])
            lateral = abs(-dx*fy+dy*fx)
            goal_dot = forward[0]*fx+forward[1]*fy
            prior = self.previous_gate.get(ident)
            crossed = prior is not None and prior < 0 <= signed and lateral <= goal['lane_width_m']/2 and goal_dot > 0
            apron = self.corridor.rear_terminal and 0 <= dx*fx+dy*fy <= 8 and lateral <= goal['lane_width_m']/2
            inside = path['tolerated'].covers(center) or crossed or apron
            valid = inside and dot > 0
            self.outside_s[ident] = 0. if valid else self.outside_s[ident]+dt
            if self.outside_s[ident]+1e-8 >= self.corridor.departure_s:
                self.candidates.discard(ident)
            if crossed:
                exit_id, exit_dot = ident, goal_dot
            self.previous_gate[ident] = signed
            raw.append({'path': ident, 'center_inside': inside, 'forward_dot': dot,
                        'station_m': station, 'offset_m': path['line'].distance(center),
                        'outside_or_reverse_s': self.outside_s[ident]})
        if self.corridor.multilane:
            valid = [r for r in raw if r['center_inside'] and r['forward_dot'] > 0]
            self.union_outside_s = 0. if valid else self.union_outside_s + dt
            if self.union_outside_s + 1e-8 >= self.corridor.departure_s:
                self.departed = True
            if valid:
                # Overlapping route polylines can sample the same junction at
                # different road IDs. Keep the current valid path instead of
                # treating a sub-centimetre nearest-path change as a lane change.
                chosen = (self.active_path if any(r['path']==self.active_path for r in valid)
                          else min(valid, key=lambda r: r['offset_m'])['path'])
                if self.active_path is None:
                    self.active_path = chosen
                elif chosen != self.active_path:
                    source = current_points[self.active_path]; target = current_points[chosen]
                    native_key = lambda p: [p.get('road_id'), p.get('section_id'), p.get('lane_id')]
                    same_native = native_key(source) == native_key(target)
                    permitted = native_key(target) in source.get('permitted_lane_change_to', [])
                    if same_native or permitted:
                        self.active_path = chosen
                    else:
                        self.departed = True
            self.candidates = set() if self.departed else {p['id'] for p in self.corridor.paths}
        footprint = MultiPoint([p[:2] for p in vehicle['bounding_box_world_vertices_m']]).convex_hull
        return {'compatible_paths': sorted(self.candidates), 'sustained_departure': not self.candidates or (exit_id is not None and exit_id not in self.candidates),
            'forward_exit_crossing': exit_id, 'exit_forward_dot': exit_dot,
            'exit_on_compatible_path': exit_id in self.candidates if exit_id else None,
            'body_outside_corridor_m2': footprint.difference(self.corridor.polygon).area,
            'paths': raw}
