"""Check route support and record vehicle departures."""
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box, shape


class SceneDomain:
    def __init__(self, map_issues=(), reviewed_polygon=None):
        self.exclusions = [(item, box(*item['exclude_bbox_carla_xy_m']))
                           for item in map_issues if item.get('status') == 'open']
        self.reviewed = (shape(reviewed_polygon) if isinstance(reviewed_polygon,dict) else Polygon(reviewed_polygon)) if reviewed_polygon else None
        self.reported = set()

    def validate_routes(self, routes, names):
        """Check routes against the reviewed driving area and excluded regions."""
        for name in names:
            route = routes[name]
            line = LineString([p['position_carla_m'][:2] for p in route['validated_waypoints']])
            if self.reviewed is not None and not self.reviewed.covers(line):
                raise ValueError(f'Route {name} leaves the reviewed driving area')
            navigation = LineString([p['position_carla_m'][:2]
                                     for p in route.get('navigation_waypoints', route['validated_waypoints'])])
            for issue, region in self.exclusions:
                if line.intersects(region) or navigation.intersects(region):
                    raise ValueError(f'Route {name} intersects excluded map region {issue["id"]}')

    def first_vehicle_violation(self, vehicles):
        if not self.exclusions and self.reviewed is None:
            return None
        for vehicle in vehicles:
            corners = vehicle.get('bounding_box_world_vertices_m')
            footprint = (MultiPoint([p[:2] for p in corners]).convex_hull if corners
                         else Point(vehicle['position_m'][:2]))
            common = dict(request_id=vehicle['request_id'], position_carla_m=vehicle['position_m'],
                          action='record scene-domain departure; continue whole observation and vehicle control')
            for issue, region in self.exclusions:
                if footprint.intersects(region):
                    return dict(common, reason='excluded_map_region', map_issue_id=issue['id'])
            if self.reviewed is not None and not self.reviewed.covers(footprint):
                return dict(common, reason='outside_reviewed_driving_area')
        return None

    def new_vehicle_violations(self, vehicles):
        """Record each vehicle/reason once, retaining actors and all later frames."""
        result=[]
        for vehicle in vehicles:
            violation=self.first_vehicle_violation([vehicle])
            if violation is None:continue
            key=(violation['request_id'],violation['reason'],violation.get('map_issue_id'))
            if key not in self.reported:
                self.reported.add(key);result.append(violation)
        return result
