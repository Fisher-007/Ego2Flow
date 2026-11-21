"""Check whether the whole vehicle has crossed a downstream exit gate."""
import math


def beyond_receiving_gate(position, forward, corners, gates):
    for gate in gates:
        gx,gy=gate['position_carla_m'][:2]
        fx,fy=gate['lane_center_forward_xy'];norm=math.hypot(fx,fy)
        fx/=norm;fy/=norm
        rear=min((p[0]-gx)*fx+(p[1]-gy)*fy for p in corners)
        lateral=abs(-(position[0]-gx)*fy+(position[1]-gy)*fx)
        if rear>=0 and lateral<=gate['lane_width_m']/2 and forward[0]*fx+forward[1]*fy>0:
            return True
    return False
