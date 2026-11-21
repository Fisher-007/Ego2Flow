"""Resolve lane paths and spatial exposure for scoring."""
from bisect import bisect_right
from collections import defaultdict
import math

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import unary_union

from benchmark.navigation.route_lane import lane_corridor


class LanePath:
    def __init__(self, name, record):
        self.name = name
        self.points = record.get('analysis_waypoints',record['validated_waypoints'])
        self.line, self.polygon, self.stations, _, _ = lane_corridor(self.points)
        self.xyz = np.asarray([p['position_carla_m'] for p in self.points],dtype=float)
        self.surface_stations = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.xyz,axis=0),axis=1))]
        self.free_flow = np.asarray([p['free_flow_speed_mps'] for p in self.points])
        if min(self.free_flow) <= 0:
            raise ValueError('Free-flow reference must be positive on '+name)
        self.group = record.get('speed_group', name)

    def location(self, vehicle):
        point = Point(vehicle['position_m'][:2])
        s = self.line.project(point)
        i = min(max(0, bisect_right(self.stations, s)-1), len(self.points)-2)
        length = self.stations[i+1]-self.stations[i]
        fraction = (s-self.stations[i])/length
        tangent = (self.xyz[i+1,:2]-self.xyz[i,:2])/length
        surface_tangent=(self.xyz[i+1]-self.xyz[i])/np.linalg.norm(self.xyz[i+1]-self.xyz[i])
        height = float(self.xyz[i,2]+fraction*(self.xyz[i+1,2]-self.xyz[i,2]))
        grade = float((self.xyz[i+1,2]-self.xyz[i,2])/length)
        terminal_delta=np.asarray(vehicle['position_m'][:2])-self.xyz[-1,:2]
        terminal_tangent=(self.xyz[-1,:2]-self.xyz[-2,:2]);terminal_tangent/=np.linalg.norm(terminal_tangent)
        apron=(0<=float(terminal_delta@terminal_tangent)<=8 and
            abs(float(np.cross(terminal_tangent,terminal_delta)))<=self.points[-1]['lane_width_m']/2)
        entry_delta=np.asarray(vehicle['position_m'][:2])-self.xyz[0,:2]
        entry_tangent=self.xyz[1,:2]-self.xyz[0,:2];entry_tangent/=np.linalg.norm(entry_tangent)
        entry_apron=(-2<=float(entry_delta@entry_tangent)<=0 and
            abs(float(np.cross(entry_tangent,entry_delta)))<=self.points[0]['lane_width_m']/2)
        return dict(station_m=s, direction=tangent, surface_direction=surface_tangent,
            surface_station_m=float(np.interp(s,self.stations,self.surface_stations)),height_m=height, grade=grade,
            offset_m=self.line.distance(point), inside=self.polygon.covers(point) or apron or entry_apron,
            free_flow_mps=float(self.free_flow[i]+fraction*(self.free_flow[i+1]-self.free_flow[i])),
            group=self.points[i].get('speed_group', self.group),
            heading_dot=float(tangent @ np.asarray(vehicle['forward_xy'])))


class EvaluationGeometry:
    def __init__(self, routes, spec):
        self.routes = routes
        self.paths = {name: LanePath(name, routes[name]) for name in spec['analysis_route_names']}
        self.domain = Polygon(spec['analysis_polygon_carla_xy_m'])
        self.road_tolerance_m = spec.get('road_height_tolerance_m', 1.)
        native_intervals = defaultdict(list)
        transitions = defaultdict(list)
        for p in self.paths.values():
            for a,b in zip(p.points,p.points[1:]):
                lane_a=(a['road_id'],a.get('section_id',0),a['lane_id'])
                lane_b=(b['road_id'],b.get('section_id',0),b['lane_id'])
                line=LineString([a['position_carla_m'][:2],b['position_carla_m'][:2]])
                clipped=line.intersection(self.domain)
                if lane_a!=lane_b:
                    transitions[(lane_a,lane_b)].append(clipped)
                    continue
                parts=[clipped] if clipped.geom_type=='LineString' else getattr(clipped,'geoms',[])
                for part in parts:
                    if part.geom_type!='LineString' or not part.length:continue
                    ends=[a['s_m']+(b['s_m']-a['s_m'])*line.project(Point(x))/line.length
                          for x in (part.coords[0],part.coords[-1])]
                    lo,hi=sorted(ends)
                    if hi>lo:native_intervals[lane_a].append((lo,hi,part.length/(hi-lo)))
        # Different OD routes may sample the same receiving lane differently.
        # Union its native station intervals before accumulating physical length.
        length=0.
        for pieces in native_intervals.values():
            cuts=sorted({x for a,b,_ in pieces for x in (a,b)})
            for lo,hi in zip(cuts,cuts[1:]):
                covering=[p for p in pieces if p[0]<(lo+hi)/2<p[1]]
                if covering:length+=(hi-lo)*min(covering,key=lambda p:p[1]-p[0])[2]
        self.lane_length_m=length+sum(unary_union(lines).length for lines in transitions.values())

    def associations(self, vehicle):
        corners=vehicle.get('bounding_box_world_vertices_m')
        footprint=MultiPoint([p[:2] for p in corners]).convex_hull if corners else Point(vehicle['position_m'][:2])
        associations={}
        for name,path in self.paths.items():
            loc=path.location(vehicle)
            if (abs(vehicle['position_m'][2]-loc['height_m']) > self.road_tolerance_m
                    or loc['heading_dot'] < math.cos(math.radians(45))):
                continue
            intersection=footprint.intersection(path.polygon)
            if loc['inside'] or (intersection.area > .05 if corners else not intersection.is_empty):
                loc['path']=name
                # Corners projected onto the same curved longitudinal coordinate.
                stations=[float(np.interp(path.line.project(Point(p[:2])),path.stations,path.surface_stations))
                          for p in corners] if corners else [loc['surface_station_m']]
                loc['rear_m'],loc['front_m']=min(stations),max(stations)
                associations[name]=loc
        return associations

    def legal_progress(self, vehicle, associations):
        if vehicle.get('corridor',{}).get('sustained_departure'):
            return 0., None
        own=self.routes[vehicle['route']]
        allowed=own['evaluation']['legal_route_names']
        valid=[a for n,a in associations.items() if n in allowed and a['inside'] and a['heading_dot']>0]
        if not valid:
            return 0., None
        nearest=min(valid,key=lambda a:a['offset_m'])
        speed=float(np.asarray(vehicle['velocity_mps']) @ nearest['surface_direction'])
        return max(0.,min(1.,speed/nearest['free_flow_mps'])), nearest

    def speed_membership(self, vehicle, associations):
        """Classify forward traffic using the declared route groups."""
        allowed=self.routes[vehicle['route']]['evaluation']['legal_route_names']
        candidates=[a for a in associations.values() if a['inside']]
        reason='declared_forward_lane'
        if not candidates:
            native=vehicle.get('native_lane',{})
            identity=(native.get('road_id'),native.get('section_id',0),native.get('lane_id'))
            for name,path in self.paths.items():
                loc=path.location(vehicle)
                matches=[p for p in path.points if (p['road_id'],p.get('section_id',0),p['lane_id'])==identity]
                if (matches and loc['heading_dot']>=math.cos(math.pi/4)
                        and abs(vehicle['position_m'][2]-loc['height_m'])<=self.road_tolerance_m
                        and loc['offset_m']<=max(p['lane_width_m'] for p in matches)/2+.15
                        and 0<loc['station_m']<path.line.length):
                    candidates.append(dict(loc,path=name));reason='native_lane_seam'
        if candidates:
            preferred=[a for a in candidates if a['path'] in allowed]
            selected=min(preferred or candidates,key=lambda a:a['offset_m'])
            if vehicle['speed_mps']>.5 and np.dot(vehicle['velocity_mps'][:2],selected['direction'])<0:
                return None,'reverse_motion'
            return selected,reason
        nearest=min((p.location(vehicle) for p in self.paths.values()),key=lambda a:a['offset_m'])
        if abs(vehicle['position_m'][2]-nearest['height_m'])>self.road_tolerance_m:
            return None,'outside_road_height'
        return None,('crosswise_or_reverse_heading' if nearest['heading_dot']<math.cos(math.pi/4)
                     else 'outside_declared_flow')

    def following(self, vehicles, associations, closing_epsilon):
        candidates={}
        for name,path in self.paths.items():
            occupied=sorted([(associations[v['request_id']][name]['station_m'],v)
                for v in vehicles if name in associations[v['request_id']]],key=lambda row:row[0])
            for (_,follower),(_,leader) in zip(occupied,occupied[1:]):
                f=associations[follower['request_id']][name]
                l=associations[leader['request_id']][name]
                gap=l['rear_m']-f['front_m']
                if gap <= 0:
                    continue  # contact/overlap is retained by collision and overlap diagnostics
                closing=float(np.asarray(follower['velocity_mps']) @ f['surface_direction']
                              - np.asarray(leader['velocity_mps']) @ l['surface_direction'])
                ttc=gap/closing if closing>closing_epsilon else math.inf
                candidates.setdefault(follower['request_id'],[]).append(dict(
                    follower_request_id=follower['request_id'],leader_request_id=leader['request_id'],
                    path=name,gap_m=gap,closing_mps=closing,ttc_s=ttc))
        # A vehicle straddling two lanes has one exposure denominator per tick.
        return {key:min(items,key=lambda x:(x['ttc_s'],x['gap_m'])) for key,items in candidates.items()}

    def interval_exposure(self, previous, current, t0, t1, window):
        lo=max(0.,(window[0]-t0)/(t1-t0));hi=min(1.,(window[1]-t0)/(t1-t0))
        if hi<=lo:return []
        xy0=np.asarray(previous['position_m'][:2]);xy1=np.asarray(current['position_m'][:2])
        delta=xy1-xy0;length=float(np.linalg.norm(delta))
        if length < 1e-10:
            return [(lo,hi,0.)] if self.domain.covers(Point(xy0)) else []
        line=LineString([xy0,xy1]);clipped=line.intersection(self.domain)
        parts=[clipped] if clipped.geom_type=='LineString' else getattr(clipped,'geoms',[])
        result=[]
        for part in parts:
            if part.geom_type!='LineString' or not part.length:continue
            ends=sorted([line.project(Point(part.coords[0]))/length,
                         line.project(Point(part.coords[-1]))/length])
            begin,end=max(lo,ends[0]),min(hi,ends[1])
            if end>begin:result.append((begin,end,length*(end-begin)))
        return result


def interpolated_state(a,b,fraction):
    """Interpolate boundary states for exposure and energy measurement."""
    result={**b}
    for key in ['position_m','velocity_mps','acceleration_mps2','forward_xy','bounding_box_world_vertices_m']:
        result[key]=((1-fraction)*np.asarray(a[key])+fraction*np.asarray(b[key])).tolist()
    result['speed_mps']=float(np.linalg.norm(result['velocity_mps'][:2]))
    if a.get('road_grade') is not None and b.get('road_grade') is not None:
        result['road_grade']=(1-fraction)*a['road_grade']+fraction*b['road_grade']
    return result
