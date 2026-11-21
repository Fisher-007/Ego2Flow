"""Declared scoring population, region and clock, separate from recording extent."""
from shapely.geometry import Point, shape, mapping
from shapely.ops import unary_union
from benchmark.navigation.route_lane import lane_corridor
from benchmark.metrics.evaluation_geometry import EvaluationGeometry


def task_scope(config, routes):
    """Define the scored vehicles, region and time window for a scene."""
    names=config['evaluation']['analysis_route_names']
    domain=unary_union([lane_corridor(routes[n]['validated_waypoints'])[1] for n in names]).buffer(4.)
    ids=[r['id'] for r in config['request_manifest'] if not r.get('external_intervention')]
    return dict(version='declared-task-scope-v1',ordinary_request_ids=ids,
        window_s=config['evaluation']['window_s'],region_geojson=mapping(domain),
        region_source=dict(route_names=names,waypoints='validated_waypoints',roadside_margin_m=4.),
        normal_exit='legal_task_completion',downstream_scoring='none; physical exit buffer only',
        retained_terminal_field='observation_terminal_crossed_s',
        metrics=dict(speed_sd='ordinary IDs, declared region, before legal task exit, assigned forward groups',
            low_ttc='same follower scope; actual physical front vehicle is context',
            collision_braking_energy='same ordinary IDs, region and per-request scoring intervals; shared distance',
            delay='all declared requests including entry waiting; legal completion or observation end',
            throughput='declared ordinary requests with legal task-exit crossing in window',
            observed_speed_loss_ratio=dict(source_id=0,target_ids=config.get('peak_gain',{}).get('follower_request_ids',list(range(1,8))),
                pre_window_s=config.get('peak_gain',{}).get('pre_window_s'),response_window_s=config.get('peak_gain',{}).get('response_window_s'),
                spatial_policy='fixed-ID task observation; departures remain outcomes; normal exit ends scoring',
                early_normal_exit='required fixed response window unavailable; do not include exit-buffer motion')))


class MetricScope:
    def __init__(self, spec, requests, observed_window):
        self.spec=spec;self.ids=set(spec['ordinary_request_ids']);self.domain=shape(spec['region_geojson'])
        self.window=[max(spec['window_s'][0],observed_window[0]),min(spec['window_s'][1],observed_window[1])]
        self.requests={r['id']:r for r in requests};self.stops={}
        for ident in self.ids:
            r=self.requests[ident];stop=self.window[1]
            # Departed requests remain unfinished until the observation cutoff.
            if spec['normal_exit']=='legal_task_completion' and r.get('exit_path') is not None:
                exit_time=r.get('completed_s')
                if exit_time is None:exit_time=r.get(spec.get('retained_terminal_field','observation_terminal_crossed_s'))
                if exit_time is not None:stop=min(stop,exit_time)
            self.stops[ident]=stop

    def request_window(self, ident):
        if ident not in self.ids:return [0.,0.]
        r=self.requests[ident]
        return [max(self.window[0],r.get('admitted_s') if r.get('admitted_s') is not None else self.window[1]),self.stops[ident]]

    def contains(self, ident, time, position):
        lo,hi=self.request_window(ident)
        return lo<=time<=hi and hi>lo and self.domain.covers(Point(position[:2]))

    def exposure(self, previous,current,t0,t1):
        return EvaluationGeometry.interval_exposure(self,previous,current,t0,t1,self.request_window(current['request_id']))

    def event_selected(self,event):
        """Check the collision callback against the vehicle's scoring window and region."""
        ident=event['request_id']
        if ident not in self.ids:return False
        lo,hi=self.request_window(ident)
        if not lo<=event['time_s']<=hi or hi<=lo:return False
        p=event.get('sensor_position_m')
        return None if p is None else self.domain.covers(Point(p[:2]))
