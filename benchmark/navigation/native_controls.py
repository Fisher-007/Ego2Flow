"""Inspect all OpenDRIVE control kinds, including fixed stop/yield references."""
from collections import defaultdict
import math
import xml.etree.ElementTree as ET


class NativeControls:
    def __init__(self, xodr_file):
        root=ET.parse(xodr_file).getroot()
        definitions={s.get('id'):s.attrib for s in root.findall('.//signal')}
        self.by_road=defaultdict(list)
        for road in root.findall('road'):
            for item in road.findall('signals/*'):
                definition=definitions.get(item.get('id'),{})
                kind=definition.get('type')
                validities=[(int(v.get('fromLane')),int(v.get('toLane'))) for v in item.findall('validity')]
                self.by_road[int(road.get('id'))].append(dict(signal_id=item.get('id'),
                    type=kind,dynamic=definition.get('dynamic'),name=definition.get('name'),
                    s_m=float(item.get('s')),validities=validities))

    def on_route(self, points):
        stations=[0.]
        for a,b in zip(points,points[1:]):
            stations.append(stations[-1]+math.dist(a['position_carla_m'][:2],b['position_carla_m'][:2]))
        by_lane=defaultdict(list)
        for i,point in enumerate(points):by_lane[(point['road_id'],point['lane_id'])].append((i,point))
        found=[];seen=set()
        for (road,lane),rows in by_lane.items():
            low=min(p['s_m'] for _,p in rows);high=max(p['s_m'] for _,p in rows)
            for control in self.by_road[road]:
                if control['validities'] and not any(min(a,b)<=lane<=max(a,b) for a,b in control['validities']):continue
                # Routes are sampled at 2 m; an incoming reference can lie
                # just before the first sample of a short junction road.
                if not low-2.1<=control['s_m']<=high+2.1:continue
                key=(road,lane,control['signal_id'],control['s_m'])
                if key in seen:continue
                seen.add(key)
                i,point=min(rows,key=lambda row:abs(row[1]['s_m']-control['s_m']))
                found.append(dict(control,road_id=road,lane_id=lane,
                    route_station_m=stations[i],station_scope='Nearest 2 m route sample; native plane used for final scoring'))
        return sorted(found,key=lambda x:x['route_station_m'])
