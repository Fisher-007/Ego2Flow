"""Select task-route milestones for CARLA Traffic Manager."""
from benchmark.policies.rule_control import TrafficManagerAgent


def junction_milestone_indices(plan, spacing=20.):
    points=[w.transform.location for w,_ in plan]
    stations=[0.]
    for a,b in zip(points,points[1:]):stations.append(stations[-1]+a.distance(b))
    spans=[];start=None
    for i,(w,_) in enumerate(plan):
        if w.is_junction and start is None:start=i
        if not w.is_junction and start is not None:
            if stations[i]-stations[start]>=8.:spans.append((start,i))
            start=None
    selected=[0];last=0.
    for i,s in enumerate(stations[1:],1):
        if s-last>=spacing:
            selected.append(i);last=s
    # TM consumes nearby milestones at squared distance <30 m² (~5.48 m).
    # The common 8 m approach clearance avoids a still-upstream goal at a fork.
    selected=[i for i in selected if i==0 or not any(
        stations[a]-8.<=stations[i]<stations[b] for a,b in spans)]
    for _,end in spans:
        index=next((i for i in range(end,len(plan))
                    if stations[i]>=stations[end]+8. and not plan[i][0].is_junction),end)
        selected.append(index)
    return sorted(set([*selected,len(plan)-1]))


def junction_milestones(plan, spacing=20.):
    return [plan[i][0].transform.location for i in junction_milestone_indices(plan,spacing)]


class RouteTrafficManagerAgent(TrafficManagerAgent):
    """Maintain task routes through native Traffic Manager navigation buffers."""
    def path_locations(self,plan):
        cache=self.options.get('native_tm_paths')
        if cache:
            if self.options.get('native_tm_outlets'):
                anchors=self.options.get('tm_milestone_positions_m')
                indices=([min(range(len(plan)),key=lambda i:sum((a-b)**2 for a,b in zip((plan[i][0].transform.location.x,plan[i][0].transform.location.y,plan[i][0].transform.location.z),xyz))) for xyz in anchors] if anchors else None)
                if indices is None:indices=junction_milestone_indices(plan,self.options['path_spacing_m'])
                plan=[plan[i]for i in indices]
            return cache.locations(plan)
        return junction_milestones(plan,self.options['path_spacing_m'])

    def set_global_plan(self,plan,**kwargs):
        enabled=self.enabled
        self.enabled=False
        try:super().set_global_plan(plan,**kwargs)
        finally:self.enabled=enabled
        cache=self.options.get('native_tm_paths')
        if cache:
            self.branches=sorted(set(self.branches+cache.branch_indices(self.plan)))
        if enabled:
            self.manager.set_path(self.vehicle,self.path_locations(self.plan),True)

    def refresh_branch_navigation(self):
        if not self.branches:return
        position=self.vehicle.get_location()
        current=min(range(len(self.plan)),key=lambda i:self.plan[i][0].transform.location.distance(position))
        near=[i for i in self.branches if i not in self.refreshed_branches
              and 0.<=self.stations[i]-self.stations[current]<=12.]
        if not near:return
        actions=self.manager.get_all_actions(self.vehicle)
        unexpected=[w.road_id for _,w in actions if not w.is_junction and w.road_id not in self.route_roads]
        if not unexpected:return
        # ImportPath retains the front waypoint. New targets must start beyond it.
        front=actions[0][1].transform.location if actions else position
        front_index=min(range(len(self.plan)),key=lambda i:self.plan[i][0].transform.location.distance(front))
        cutoff=max(self.stations[current],self.stations[front_index])+6.
        locations=[];location_stations=[]
        for location in self.path_locations(self.plan):
            index=min(range(len(self.plan)),key=lambda i:self.plan[i][0].transform.location.distance(location))
            if self.stations[index]>=cutoff:
                locations.append(location);location_stations.append(self.stations[index])
        if not locations:return
        self.manager.set_path(self.vehicle,locations,True)
        self.refreshed_branches.add(near[0])
        self.navigation_updates.append(dict(route_index=current,branch_index=near[0],
            position_m=[position.x,position.y,position.z],unexpected_planned_roads=unexpected,
            remaining_road_ids=list(dict.fromkeys(w.road_id for w,_ in self.plan[current:])),
            milestones='native TM nodes + outlet anchors' if self.options.get('native_tm_outlets') else 'native TM nodes' if self.options.get('native_tm_paths') else 'junction outlet anchors',
            buffer_front_station_m=self.stations[front_index],first_import_station_m=location_stations[0],
            uploads_per_refresh=1))
