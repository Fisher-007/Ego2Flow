"""Measure vehicle footprint overlap with supplied native lane corridors."""
import argparse
import json
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon, MultiPoint
from shapely.ops import unary_union


def lane_corridor(points):
    left, right, stations = [], [], [0.]
    for i,p in enumerate(points):
        x,y = p['position_carla_m'][:2]
        fx,fy = p['lane_center_forward_xy']
        norm=(fx*fx+fy*fy)**.5
        rx,ry=-fy/norm,fx/norm
        half=p['lane_width_m']/2
        left.append((x-rx*half,y-ry*half));right.append((x+rx*half,y+ry*half))
        if i:
            a=points[i-1]['position_carla_m']
            stations.append(stations[-1]+((x-a[0])**2+(y-a[1])**2)**.5)
    strips=[Polygon([left[i],left[i+1],right[i+1],right[i]]) for i in range(len(points)-1)]
    # On tight bends the inner lane edges can cross between samples.
    corridor=unary_union([p if p.is_valid else p.convex_hull for p in strips if p.area>0])
    line=LineString([p['position_carla_m'][:2] for p in points])
    return line,corridor,stations,left,right


def evaluate(frames, points, request_id=0):
    line,corridor,stations,left,right=lane_corridor(points)
    junctions=[s for p,s in zip(points,stations) if p['is_junction']]
    rows=[];excluded=0
    for row in frames:
        for v in row['vehicles']:
            if v['request_id']!=request_id:continue
            center=Point(*v['position_m'][:2]);station=line.project(center)
            # Single-lane occupancy is ambiguous at junctions and route endpoints.
            if min(station,line.length-station)<3 or any(abs(station-s)<3 for s in junctions):
                excluded+=1;continue
            footprint=MultiPoint([p[:2] for p in v['bounding_box_world_vertices_m']]).convex_hull
            outside=footprint.difference(corridor).area
            rows.append({'time_s':row['time_s'],'dt_s':row['dt_s'],'frame':row['frame'],
                'route_station_m':station,'route_offset_m':v['route_offset_m'],
                'footprint_outside_lane_m2':outside,'footprint_area_m2':footprint.area,
                'center_outside_lane':not corridor.covers(center),'speed_mps':v['speed_mps']})
    # Apply the area threshold when counting time outside the lane.
    threshold=.05
    intervals=[]
    for r in rows:
        if r['footprint_outside_lane_m2']<=threshold:continue
        if not intervals or abs(r['time_s']-intervals[-1]['end_s']-r['dt_s'])>1e-7:
            intervals.append({'start_s':r['time_s'],'end_s':r['time_s'],'duration_s':r['dt_s']})
        else:
            intervals[-1]['end_s']=r['time_s'];intervals[-1]['duration_s']+=r['dt_s']
    return {'scope': ('Measure vehicle footprint overlap with supplied native lane corridors.\n\n'
                      'Corridors use sampled OpenDRIVE lane widths. Junctions and route endpoints\n'
                      'are excluded from this occupancy diagnostic.\n'),
        'request_id':request_id,'lane_point_spacing':'Supplied native waypoint spacing; widths and heading queried from the same XODR',
        'evaluated_seconds':sum(r['dt_s'] for r in rows),'evaluated_frames':len(rows),'excluded_endpoint_or_junction_frames':excluded,
        'outside_area_threshold_m2':threshold,'footprint_outside_seconds':sum(r['dt_s'] for r in rows if r['footprint_outside_lane_m2']>threshold),
        'center_outside_seconds':sum(r['dt_s'] for r in rows if r['center_outside_lane']),
        'max_outside_area_m2':max((r['footprint_outside_lane_m2'] for r in rows),default=0),
        'max_outside_frame':max(rows,key=lambda r:r['footprint_outside_lane_m2']) if rows else None,
        'intervals':intervals,'samples':rows,
        'corridor_left_xy_m':left,'corridor_right_xy_m':right}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--scene-dir',type=Path,required=True)
    p.add_argument('--xodr',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--end-time',type=float)
    p.add_argument('--request-id',type=int,default=0,help='Score this demand ID independently; defaults to the original first vehicle')
    args=p.parse_args()
    import carla
    native=carla.Map('offline_lane_diagnostic',args.xodr.read_text())
    config=json.loads((args.run/'config.json').read_text())
    requests=json.loads((args.run/'requests.json').read_text())
    request=next(r for r in requests if r['id']==args.request_id)
    route_name=request.get('route',config.get('route'))
    points=json.loads((args.scene_dir/'candidate_routes.json').read_text())[route_name]['validated_waypoints']
    for point in points:
        wp=native.get_waypoint_xodr(point['road_id'],point['lane_id'],point['s_m'])
        if wp.transform.location.distance(carla.Location(*point['position_carla_m']))>.01:
            raise ValueError('Supplied route and scoring XODR do not match')
        f=wp.transform.get_forward_vector()
        point.update(lane_width_m=wp.lane_width,lane_center_forward_xy=[f.x,f.y],is_junction=wp.is_junction)
    frames=[json.loads(s) for s in (args.run/'frames.jsonl').read_text().splitlines()]
    if args.end_time is not None:
        frames=[r for r in frames if r['time_s']<=args.end_time]
    result=evaluate(frames,points,request_id=args.request_id)
    result['observation_time_s']=[frames[0]['time_s'],frames[-1]['time_s']]
    result.update(run=str(args.run.resolve()),scene_dir=str(args.scene_dir.resolve()),xodr=str(args.xodr.resolve()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['evaluated_frames','footprint_outside_seconds','center_outside_seconds','max_outside_area_m2','intervals']},indent=2))


if __name__=='__main__':main()
