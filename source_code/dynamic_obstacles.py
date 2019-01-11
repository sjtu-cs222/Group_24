import numpy as np
import math


class DynaObs:
    def __init__(self, num_of_dyna_obs):
        self.num_of_obs = num_of_dyna_obs
        self.interval_time = 0
        self.global_time = 0
        self._delta_t = 0.001
        self.ob_states = np.zeros((num_of_dyna_obs, 5))     # each state: [x, y, theta, v, w]
        self.vel_lim = np.zeros((num_of_dyna_obs, 2))       # [v_max, w_max]    both from 0 to 1
        self.center_pts = np.zeros((num_of_dyna_obs, 2))    # center points for every obstacle [x, y]
        self.radius = np.zeros(num_of_dyna_obs)             # limit the range of movement for every obstacle
        # when an obstacle reaches its boundary, it should move directly toward its center
        # flag = 0: move randomly,  flag = 1: move back toward center
        self.back_to_center_flag = np.zeros(num_of_dyna_obs, dtype='int')

    def initial(self, x_lim, y_lim, velocity_lim, method='same'):
        # _x_lim: [left, right], _y_lim: [down, up],  the boundary of obstacles
        # velocity_lim: [v_max, w_max]
        self.set_centers([x_lim[0]+3, x_lim[1]-3], [y_lim[0]+3, y_lim[1]-3])
        self.ob_states[:, :2] = self.center_pts.copy()
        self.set_radius(x_lim, y_lim)
        self.set_vel_lim(velocity_lim[0], velocity_lim[1], method)

    def set_centers(self, center_x_lim, center_y_lim):
        # center_x_lim: [left, right], center_y_lim: [down, up],  the boundary of centers
        xs = np.random.uniform(center_x_lim[0], center_x_lim[1], self.num_of_obs)
        ys = np.random.uniform(center_y_lim[0], center_y_lim[1], self.num_of_obs)
        self.center_pts = np.concatenate((xs, ys)).reshape((2, self.num_of_obs)).transpose()

    def set_radius(self, x_lim, y_lim):
        # x_lim: [left, right], y_lim: [down, up],  the boundary of movement of obstacles
        boundary = np.array([x_lim[0], y_lim[0], x_lim[1], y_lim[1]] *
                            self.num_of_obs).reshape((self.num_of_obs, 4))
        center = np.concatenate((self.center_pts, self.center_pts), axis=1)
        max_radius = np.min(abs(center - boundary), axis=1)
        self.radius = max_radius
        # self.radius = np.array(list(map(lambda x: np.random.randint(1, x+1), max_radius)))

    def set_vel_lim(self, v_max, w_max, method='same'):
        if method == 'same':
            self.vel_lim = np.array([v_max, w_max] * self.num_of_obs).reshape((self.num_of_obs, 2))
        elif method == 'random':
            vs = np.random.uniform(v_max / 2, v_max, self.num_of_obs)
            ws = np.random.uniform(w_max / 2, w_max, self.num_of_obs)
            self.vel_lim = np.concatenate((vs, ws)).reshape((2, self.num_of_obs)).transpose()

    def update(self, current_time=None):
        if current_time is None:
            self.interval_time = 0.1
        else:
            self.interval_time = current_time - self.global_time
            self.global_time = current_time

        # move randomly
        random_curr_vel = np.array(list(map(lambda vel: [np.random.uniform(0.1, vel[0]),
                                                         np.random.uniform(-vel[1], vel[1])], self.vel_lim)))
        random_curr_vel[np.where(self.back_to_center_flag > 0)] = 0

        # move back toward center
        back_to_center_vel = np.array(list(map(lambda vel: [vel[0], 0], self.vel_lim)))
        back_to_center_vel[np.where(self.back_to_center_flag == 0)] = 0

        self.ob_states[:, 3:] = random_curr_vel + back_to_center_vel    # set v and w

        for step in range(int(self.interval_time / self._delta_t)):
            self.ob_states = np.array(list(map(self.change_state, self.ob_states)))

        # check whether an obstacle is far from its center
        diff = self.center_pts - self.ob_states[:, :2]      # x, y difference between each obstacle and its center
        distance_sq = np.array(list(map(lambda pos: pos[0]**2 + pos[1]**2, diff)))
        r_sq = self.radius ** 2

        idx_far_from_center = np.where(distance_sq > 0.9 * r_sq)
        self.back_to_center_flag[idx_far_from_center] = 1
        self.ob_states[:, 2][idx_far_from_center] = np.array(list(map(lambda pos: math.atan2(pos[1], pos[0]),
                                                                      diff[idx_far_from_center])))
        self.back_to_center_flag[np.where(distance_sq < 0.1 * r_sq)] = 0

    # motion equation
    def change_state(self, state):      # state: [x, y, theta, v, w]
        weights = np.array([[math.cos(state[2]), 0], [math.sin(state[2]), 0], [0, 1]])
        return np.concatenate(((np.matmul(weights, state[3:]) * self._delta_t) + state[:3], state[3:]))


# eg.
# dynamic_obs = DynaObs(15)
# dynamic_obs.initial(x_lim=[0, 30], y_lim=[0, 30], velocity_lim=[2, 1], method='random')
#
# current_time = 5
# dynamic_obs.update(current_time)
