"""Read route locations from the CARLA Traffic Manager cache."""
import struct
from collections import defaultdict


def read_cache(path,native):
    # CARLA writes this layout in LibCarla/trafficmanager/CachedSimpleWaypoint::Write.
    data=path.read_bytes();offset=0
    def take(fmt):
        nonlocal offset
        values=struct.unpack_from('<'+fmt,data,offset);offset+=struct.calcsize('<'+fmt)
        return values
    count,=take('I');nodes=[]
    for _ in range(count):
        ident,road,section,lane,station=take('QIIif')
        n,=take('H');following=take('Q'*n)
        n,=take('H');previous=take('Q'*n)
        left,right,grid,junction,action=take('QQi?B')
        w=native.get_waypoint_xodr(road,lane,station)
        if w is None:w=native.get_waypoint_xodr(road,lane,station-.001)
        if w is None:raise ValueError(f'Cannot resolve native cache node {road}/{lane}/{station}')
        p=w.transform.location
        nodes.append(dict(id=ident,road_id=road,section_id=section,lane_id=lane,s_m=station,
            next=list(following),previous=list(previous),junction=junction,action=action,
            position_carla_m=[p.x,p.y,p.z]))
    if offset!=len(data):raise ValueError('Native cache format differs from CARLA 0.9.16')
    return nodes


class NativeTMPath:
    def __init__(self,path,native):
        from scipy.spatial import cKDTree
        self.native=native
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            native.cook_in_memory_map(str(path))
        self.nodes=read_cache(path,native)
        self.tree=cKDTree([node['position_carla_m']for node in self.nodes])
        self.lanes=defaultdict(list)
        for node in self.nodes:self.lanes[(node['road_id'],node['section_id'],node['lane_id'])].append(node)

    def node_for(self,waypoint):
        candidates=self.lanes[(waypoint.road_id,waypoint.section_id,waypoint.lane_id)]
        if not candidates:raise ValueError('Declared lane is absent from the native TM graph')
        # At a shared endpoint two native branches can have identical XYZ.
        # Use an interior sample on the same declared lane when available.
        for node in sorted(candidates,key=lambda node:abs(node['s_m']-waypoint.s)):
            if abs(node['s_m']-waypoint.s)>10.:break
            _,nearest=self.tree.query(node['position_carla_m'])
            other=self.nodes[nearest]
            if (other['road_id'],other['section_id'],other['lane_id'])==(waypoint.road_id,waypoint.section_id,waypoint.lane_id):return node
        return None

    def locations(self,plan):
        import carla
        nodes=[]
        for waypoint,_ in plan:
            node=self.node_for(waypoint)
            if node is None:continue  # Shared boundary; adjacent lane interiors carry the same route.
            if not nodes or nodes[-1]['id']!=node['id']:nodes.append(node)
        return [carla.Location(*node['position_carla_m'])for node in nodes]

    def branch_indices(self,plan):
        """Find Traffic Manager forks along the route."""
        lanes=defaultdict(list)
        for i,(w,_) in enumerate(plan):
            lanes[(w.road_id,w.section_id,w.lane_id)].append((i,w.s))
        indices=[]
        for key,samples in lanes.items():
            low=min(s for _,s in samples);high=max(s for _,s in samples)
            for node in self.lanes[key]:
                if len(node['next'])>1 and low-5.<=node['s_m']<=high+5.:
                    indices.append(min(samples,key=lambda pair:abs(pair[1]-node['s_m']))[0])
        return sorted(set(indices))
