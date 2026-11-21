"""Example driving policy that follows a route using vehicle pose."""
import math


def create(config):
    return Owner(config)


class Owner:
    def __init__(self, config):
        self.config = dict(config)
        self.sensors = []
        self.description = dict(name='Example route follower', inputs='Ego pose, speed and prescribed route')

    def new_vehicle(self):
        return Session(self.config)

    def close(self):
        pass


class Session:
    localization_sensor_x_m = 0.

    def __init__(self, config):
        self.target_speed = config.get('target_speed_mps', 4.)
        self.index = 0

    def bind_request(self, request_id):
        self.request_id = request_id

    def reset(self, plan):
        self.points = [(w.transform.location.x, w.transform.location.y) for w, _ in plan]
        self.index = 0

    def step(self, inputs, speed, position_xyz, yaw_deg, *, gps_position_xyz=None):
        x, y = position_xyz[:2]
        while self.index < len(self.points)-1 and math.hypot(self.points[self.index][0]-x, self.points[self.index][1]-y) < 4.:
            self.index += 1
        dx, dy = self.points[self.index][0]-x, self.points[self.index][1]-y
        error = (math.atan2(dy, dx)-math.radians(yaw_deg)+math.pi) % (2*math.pi)-math.pi
        stop = self.index == len(self.points)-1 and math.hypot(dx,dy) < 2.
        throttle = max(0., min(.5, .25*(self.target_speed-speed))) if not stop else 0.
        brake = 1. if stop else max(0., min(1., .25*(speed-self.target_speed)))
        return dict(steer=max(-1.,min(1.,error)), throttle=throttle, brake=brake), dict(route_index=self.index)

    def close(self):
        self.points = []
