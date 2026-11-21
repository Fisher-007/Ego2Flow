"""LEAD lateral PID with bounded lookahead after curvature correction."""
import numpy as np


# Adapted from LEAD cvpr2026, lead/common/pid_controller.py (LateralPIDController.step).
def bounded_lateral_step(self, route, current_speed, ego_vehicle_location,
                         ego_vehicle_rotation, sensor_agent_steer_correction=False):
    from lead.common import common_utils
    lookahead = np.clip(self.speed_scale * current_speed * 3.6 + self.speed_offset, 24, 105) / 10 - 2
    lookahead = int(min(lookahead, route.shape[0] - 1))
    curvature = common_utils.waypoints_curvature(route)
    if sensor_agent_steer_correction:
        lookahead += np.clip(int(curvature * self.config.sensor_agent_steer_correction_param), 0, 2)
    requested = int(lookahead)
    lookahead = min(requested, len(route) - 1)
    self.lookahead_diagnostic = dict(requested=requested, used=lookahead,
                                    clipped=requested != lookahead)
    desired = route[lookahead] - ego_vehicle_location
    heading = (np.arctan2(desired[1], desired[0]) - ego_vehicle_rotation) % (2 * np.pi)
    heading = heading if heading < np.pi else heading - 2 * np.pi
    heading = heading * 180.0 / np.pi / 90.0
    self._window.append(heading)
    self._window = self._window[-self.n:]
    derivative = 0.0 if len(self._window) == 1 else self._window[-1] - self._window[-2]
    steering = np.clip(self.k_p * heading + self.k_d * derivative + self.k_i * np.mean(self._window), -1., 1.).item()
    return round(float(np.clip(steering, -1., 1.)), 3)


def install_bounded_lookahead(inference):
    from types import MethodType
    count = 0
    for value in vars(inference).values():
        if type(value).__name__ == 'LateralPIDController':
            value.step = MethodType(bounded_lateral_step, value)
            count += 1
    if count != 1:
        raise ValueError('Expected the published LEAD lateral PID instance')
