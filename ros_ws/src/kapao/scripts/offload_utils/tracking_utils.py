#!/usr/bin/env python
import rospy
import numpy as np
from numpy import ma
from pykalman import KalmanFilter
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from std_srvs.srv import Empty

def los_end(start, end, length, safety_length, map, step_size=0.5):
    norm = np.linalg.norm(end - start)
    one_step = (end - start) / (norm + 1e-6)
    step_num = int(length / step_size)
    safety_step_num = int(safety_length / step_size)
    if step_num <= 0:
        return start
    step_lengths = np.arange(step_num + safety_step_num, dtype=np.float32) * step_size
    all_steps = one_step.reshape(1,-1) * step_lengths.reshape(-1,1) + start   # [step_num, 2]
    all_steps = all_steps.astype(np.int32)
    all_steps = np.clip(all_steps, a_min=0, a_max=np.array([map.shape]) - 1)
    visited = map[tuple(all_steps.T)] > 0
    if visited.any():
        idx = np.argmax(visited) - safety_step_num  # backward from the obstacle
        return all_steps[max(0, idx - 1)]
    return all_steps[-1 * safety_step_num]

def assign_around_2d(map, idx, val, footprint_size=4):
    height, width = map.shape
    map[max(int(idx[0])-footprint_size, 0): min(int(idx[0])+footprint_size, height), 
        max(int(idx[1])-footprint_size, 0): min(int(idx[1])+footprint_size, width)
        ] = val

class Kalman2DTracker:
    def __init__(self, init_len=10, buffer_len=100, update_freq=30) -> None:
        self.init_len = init_len
        self.len = buffer_len
        self.update_freq = update_freq
        assert buffer_len > init_len and update_freq > init_len
        self.observations = ma.masked_all([buffer_len, 2], dtype=np.int32)
        self.idx = 0
        self.kf = None
        self.inited = False
        self.x_now = self.P_now = None

    def observed(self, observation):
        if not self.inited and observation is ma.masked:
            return [None, None]
        self.observations.mask[self.idx % self.len] = False
        self.observations[self.idx % self.len] = observation
        self.idx += 1
        if not self.inited and self.idx == self.init_len:
            self.inited = True
            initial_state_mean = [self.observations[0, 0],
                      0,
                      self.observations[0, 1],
                      0]

            transition_matrix = [[1, 1, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 1],
                                [0, 0, 0, 1]]

            observation_matrix = [[1, 0, 0, 0],
                                [0, 0, 1, 0]]
            kf = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean)
            kf = kf.em(self.observations[:self.idx], n_iter=5)
            (filtered_state_means, filtered_state_covariances) = kf.filter(
                self.observations[:self.idx])
            self.x_now = filtered_state_means[-1, :]
            self.P_now = filtered_state_covariances[-1, :]
            self.kf = kf
            if isinstance(self.x_now, ma.MaskedArray):
                return self.x_now[[0,2]].data
            else:
                return self.x_now[[0,2]]
        if self.inited:
            # if self.idx % self.update_freq == 0:
            #     self.kf = self.kf.em(self.observations[:self.idx], n_iter=5)
            (self.x_now, self.P_now) = self.kf.filter_update(
                filtered_state_mean = self.x_now,
                filtered_state_covariance = self.P_now,
                observation = observation)
            if isinstance(self.x_now, ma.MaskedArray):
                return self.x_now[[0,2]].data
            else:
                return self.x_now[[0,2]]
        else:
            if not ma.is_masked(self.observations):
                return self.observations[self.idx-1].data
            # Return the latest valid observation
            slices = ma.flatnotmasked_contiguous(self.observations)
            if len(slices) == 0:
                return [None, None]
            last_slice = slices[-1]
            return self.observations.ravel()[last_slice].reshape(-1,2)[-1].data

class CostmapMaintain:
    def __init__(self, main_name="move_base/global_costmap/costmap",
                 update="move_base/global_costmap/costmap_updates") -> None:
        self.map = None
        self.origin = None
        self.resolution = None
        rospy.Subscriber(main_name, OccupancyGrid, self.get_whole_costmap)
        rospy.Subscriber(update, OccupancyGridUpdate, self.update_costmap)

    def get_whole_costmap(self, msg: OccupancyGrid):
        height, width = msg.info.height, msg.info.width
        self.resolution = msg.info.resolution
        self.origin = np.array([
            msg.info.origin.position.x,
            msg.info.origin.position.y,
            msg.info.origin.position.z,
        ])
        self.map = np.array(msg.data).reshape([height, width])

    def update_costmap(self, msg: OccupancyGridUpdate):
        if self.map is None:
            rospy.logwarn("Main costmap not received.")
            return
        x, y, height, width = msg.x, msg.y, msg.height, msg.width
        partial_map = np.asarray(msg.data).reshape([height, width])
        try:
            self.map[y: y+height, x: x+width] = partial_map
        except:
            rospy.logwarn(f"Costmap update error. Clearing costmap.")
            try:
                rospy.ServiceProxy("/move_base/clear_costmaps", Empty)()
            except:
                pass
