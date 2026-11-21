"""Use the supplied legal route with BasicAgent without rebuilding a city graph."""
from agents.navigation.global_route_planner import GlobalRoutePlanner


class SuppliedRoutePlanner(GlobalRoutePlanner):
    """The GlobalRoutePlanner interface over one already validated waypoint plan."""

    def __init__(self, plan):
        self.plan=list(plan)

    def trace_route(self, origin, destination):
        start=min(range(len(self.plan)),key=lambda i:self.plan[i][0].transform.location.distance(origin))
        end=min(range(len(self.plan)),key=lambda i:self.plan[i][0].transform.location.distance(destination))
        return self.plan[start:end+1]
