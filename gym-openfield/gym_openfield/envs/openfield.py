"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import json
import os
import pandas as pd
import socket
import pickle


class OpenFieldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

    def __init__(self, goal_velocity=0):
        self._read_json_p()
        self.min_position = np.array([self.xmin_position, self.xmax_position])
        self.max_speed = 0.07
        
        # self.goal_velocity = goal_velocity

        # self.force=0.001
        # self.gravity=0.0025

        self.low = np.array([self.xmin_position, self.ymin_position])
        self.high = np.array([self.xmax_position, self.ymax_position])

        self.viewer = None

        self.action_space = spaces.Box(low=np.array([-10.0, -10.0, 0.0]), high=np.array([10.0, 10.0, 10000.0]),
                                       dtype=np.float32)  # Should read from a corresponding parameter file
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.last_action = 0.0
        self.action = 0.0

        #        self.reward_recep_field = 0.1

        self.seed()
        self.read_trials_params()
        
        self.crnt_trial_num = 1
        self.crnt_trial_time = 0
        
        self.trials_update()
        self.cnt = 0
        self.cnt_begin = 0
        self.done = False
        self.cnt_rew_deliver = 5
        self.prev_tr_end = 0.0

        self.f_tr_loc = open(os.path.join(self.data_path, 'locs_time.dat'), 'w')
        self.f_tr_loc.write('trial\ttime\tx\ty\n')

        self.f_tr_time_rew = open(os.path.join(self.data_path, 'trial_time_rew.dat'), 'w')
        self.f_tr_time_rew.write('trial\ttime\treward\n')
        
        self.host = socket.gethostname()
        self.port = 41111
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.settimeout(5)
        try:
            self.socket.connect((self.host, self.port))
            self.connected = True
        except socket.error as e:
            print(f"[CONNECTION NOT POSSIBLE] {e}")
            self.connected = False
            
        
    def read_trials_params(self):
        with open('parameter_sets/current_parameter/sim_params.json', 'r') as fl:
            sim_dict = json.load(fl)

        data_dir = sim_dict['data_path']
        master_rng_seed = str(sim_dict['master_rng_seed'])
        main_data_dir = os.path.join(*data_dir.split('/')[0:-1])
        fig_dir = os.path.join(main_data_dir, 'fig-' + master_rng_seed)
        fl_name = 'parameter_sets/current_parameter/trials_params.dat'
#        self.trials_params = pd.read_csv(os.path.join(fig_dir,fl_name),sep = "\t")
        self.trials_params = pd.read_csv(fl_name,sep = "\t")
        
    def trials_update(self):
        if self.crnt_trial_num <=self.max_num_trs:
            trial_dummy = self.trials_params.loc[self.trials_params['trial_num']==self.crnt_trial_num]    
            self.start_x = trial_dummy['start_x'].values[0]
            self.start_y = trial_dummy['start_y'].values[0]
            goal_x = trial_dummy['goal_x'].values[0]
            goal_y = trial_dummy['goal_y'].values[0]
            self.goal_shape = trial_dummy['goal_shape'].values[0]
            if self.goal_shape == 'round':
                self.reward_recep_field = trial_dummy['goal_size1'].values[0]
            elif self.goal_shape == 'rect':
                self.reward_recep_field = (trial_dummy['goal_size1'].values[0], trial_dummy['goal_size2'].values[0])
            self.max_tr_dur = trial_dummy['max_tr_dur'].values[0] / 1000

        else:
            self.start_x = self.xmax_position+5.
            self.start_y = self.ymax_position+5.
            goal_x = self.xmin_position
            goal_y = self.ymin_position
            self.goal_shape = None
            self.reward_recep_field = 0.0
        self.goal_position = np.array([goal_x, goal_y])        
        
    def seed(self, seed=None): 
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if len(action) == 1:
            # print('openfield-env: ', 'action vector has only one element. Neutral action is chosen!')
            action = np.array([0., 0.])
            nest_running = False
            runtime = 0.0
        else:
            runtime = action[-1]
            action = action[0:-1]
            nest_running = True
            self.cnt_begin += 1

        velocity = np.array(action)
#        print('velocity=',velocity)

        self.action = np.arctan2(action[0], action[1])
        position = self.state[0:2]

        tmp_pos = position + velocity
        
        candidates = []
        # Check is the line segment pos->tmp_pos doesn't cross any line segment defined by the obstacle
        for obstacle in self.obstacle_list:
            for i, point in enumerate(obstacle):
                obs_segment = (obstacle[i], obstacle[i - 1]) # Points need to be listed sequentially!
                if self._intersect(obs_segment[0], obs_segment[1], position, tmp_pos):
                    candidates.append(self._calc_collision_point(obs_segment[0], obs_segment[1], position, tmp_pos))
        
        #This handles cases where multiple segments are crossed
        if len(candidates) > 1:
            best = candidates[0]
            for candidate in candidates:
                if math.dist(tmp_pos, candidate) < math.dist(tmp_pos, best):
                    best = candidate
            tmp_pos = best
        elif len(candidates) == 1:
            tmp_pos = candidates[0]

        # Can maybe use numpy.clip here?
        if tmp_pos[0] > self.xmax_position:
            tmp_pos[0] = self.xmax_position
        if tmp_pos[0] < self.xmin_position:
            tmp_pos[0] = self.xmin_position
        if tmp_pos[1] > self.ymax_position:
            tmp_pos[1] = self.ymax_position
        if tmp_pos[1] < self.ymin_position:
            tmp_pos[1] = self.ymin_position

        position = tmp_pos
        reward = -1

        #field_geom_mean = np.sqrt((self.xmax_position - self.xmin_position) * (self.ymax_position - self.ymin_position))
        field_geom_mean = 1
        rew_territory = field_geom_mean * self.reward_recep_field
        if self.goal_shape == 'round':
            done = bool(np.linalg.norm(position - self.goal_position) <= rew_territory)
        elif self.goal_shape == 'rect':
            done = OpenFieldEnv._inside_rect(position[0], position[1], self.goal_position, self.reward_recep_field)
        elif self.goal_shape is None:
            done = False
            
        # if self.hide_goal == True:
        #     done = False


        self.f_tr_loc.write('{:d}\t{:.5f}\t{:.5f}\t{:.5f}\n'.format(self.crnt_trial_num, runtime, position[0], position[1]))
        
        if self.connected:
            serialized_data = pickle.dumps((self.crnt_trial_num, runtime, position[0], position[1]))
            try:
                self.socket.send(serialized_data)
            except BrokenPipeError:
                self.socket.close()
                self.connected = False

        crnt_trial_time = runtime - self.prev_tr_end
        
        if done:
            reward = 1
            self.f_tr_time_rew.write('{:d}\t{:.5f}\t{:.1f}\n'.format(self.crnt_trial_num, runtime, reward))
            self.crnt_trial_num += 1
            self.trials_update()
            self.prev_tr_end = runtime
        elif crnt_trial_time >= self.max_tr_dur:
            self.f_tr_time_rew.write('{:d}\t{:.5f}\t{:.1f}\n'.format(self.crnt_trial_num, runtime, reward))
            self.prev_tr_end = runtime
            self.crnt_trial_num += 1
            self.trials_update()
            done = True

        self.state = position
        self.cnt += 1
#        print("openfield state = ",np.append(self.state, [self.crnt_trial_num, crnt_trial_time]))

        return np.append(self.state, [self.crnt_trial_num, crnt_trial_time]) ,reward, done, {}
    
    @staticmethod
    def _inside_rect(x, y, r_center, field):
        r_w = field[0]
        r_h = field[1]
        left_x = r_center[0] - r_w
        right_x = r_center[0] + r_w
        top_y = r_center[1] - r_h
        bottom_y = r_center[1] + r_h
        
        return (x >= left_x and x <= right_x and y >= top_y and y <= bottom_y)
            
    '''
    Calculates the intersection point between two line segments, so that corrections
    can be made if the agent attempts to move into an obstacle
    '''
    @staticmethod
    def _calc_collision_point(obs1, obs2, traj1, traj2):
        OFFSET = .05 # Offset value which defines how far to displace the agent from the obstacle border in the case of a collision.
        
        # Get vectors for obstacle line segment and trajectory
        obs_seg = np.subtract(obs2, obs1)
        traj_seg = np.subtract(traj2, traj1)
        
        dist1 = (obs_seg[0] * (obs1[1] - traj1[1]) - obs_seg[1] * (obs1[0] - traj1[0])) / (traj_seg[1] * obs_seg[0] - traj_seg[0] * obs_seg[1])
        dist2 = (traj_seg[0] * (obs1[1] - traj1[1]) - traj_seg[1] * (obs1[0] - traj1[0])) / (traj_seg[1] * obs_seg[0] - traj_seg[0] * obs_seg[1])
        
        if (dist1 >= 0 and dist1 <= 1 and dist2 >= 0 and dist2 <= 1):
            dist1 = max(dist1 - OFFSET, 0)
            x_int = traj1[0] + (dist1 * traj_seg[0])
            y_int = traj1[1] + (dist1 * traj_seg[1])
            return [x_int, y_int]
        else:
            # If this method is called at all, we expect a collision. This branch indicates some sort of bug, since we haven't found one!
            raise NotImplementedError
    
    
    #Checks if three given points are arranged in a counter-clockwise order. This is used for collision detection.
    #https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    @staticmethod
    def _ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    
    @staticmethod
    def _intersect(A, B, C, D):
        return OpenFieldEnv._ccw(A, C, D) != OpenFieldEnv._ccw(B, C, D) and OpenFieldEnv._ccw(A, B, C) != OpenFieldEnv._ccw(A, B, D)

    def reset(self):
        self.state = np.array([self.start_x, self.start_y,self.crnt_trial_time])  
        return np.array(self.state)

    def _height(self, xs):
        return self.ymax_position - self.ymin_position

    # def render(self, mode='human'):
    #     screen_width = 600
    #     screen_height = 600

    #     world_width = self.xmax_position - self.xmin_position
    #     world_height = self.ymax_position - self.ymin_position
    #     scale_w = screen_width / world_width
    #     scale_h = screen_height / world_height
    #     carwidth = 40
    #     carheight = 20

    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering
    #         self.viewer = rendering.Viewer(screen_width, screen_height)
    #         # xs = np.linspace(self.xmin_position, self.xmax_position, 100)
    #         # ys = np.linspace(self.ymin_position, self.ymax_position, 100)
    #         # xys = list(zip((xs-self.min_position)*scale, (ys-self.min_position)*scale))

    #         # self.track = rendering.make_polyline(xys)
    #         # self.track.set_linewidth(4)
    #         # self.viewer.add_geom(self.track)

    #         clearance = 10

    #         l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
    #         rat = rendering.make_circle(carheight)  # rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])#
    #         rat.add_attr(rendering.Transform(translation=(0, clearance)))
    #         self.cartrans = rendering.Transform()
    #         rat.add_attr(self.cartrans)
    #         self.viewer.add_geom(rat)
    #         frontwheel = rendering.make_circle(carheight / 4)
    #         frontwheel.set_color(.5, .5, .5)
    #         frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
    #         frontwheel.add_attr(self.cartrans)
    #         self.viewer.add_geom(frontwheel)
    #         # backwheel = rendering.make_circle(carheight/2.5)
    #         # backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
    #         # backwheel.add_attr(self.cartrans)
    #         # backwheel.set_color(.5, .5, .5)
    #         # self.viewer.add_geom(backwheel)
    #         flagx = (self.goal_position[0] - self.xmin_position) * scale_w
    #         flagy1 = (self.goal_position[1] - self.ymin_position) * scale_h
    #         flagy2 = flagy1 + 50
    #         flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
    #         self.viewer.add_geom(flagpole)
    #         flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
    #         flag.set_color(.8, .8, 0)
    #         self.viewer.add_geom(flag)

    #     pos = self.state
    #     self.cartrans.set_translation((pos[0] - self.xmin_position) * scale_w, (pos[1] - self.ymin_position) * scale_h)
    #     self.cartrans.set_rotation(np.pi / 2 - self.action)
    #     # self.last_action = np.copy(self.action)

    #     return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_keys_to_action(self):
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}  # control with left and right arrow keys

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _read_json_p(self):
        with open('parameter_sets/current_parameter/sim_params.json', 'r') as f:
            sim_dict = json.load(f)
        with open('parameter_sets/current_parameter/env_params.json', 'r') as f:
            env_dict = json.load(f)
        simtime = sim_dict['simtime'] * 1000
        dt = sim_dict['dt']
        self.max_num_trs = sim_dict['max_num_trs']
        self.steps_to_stop = simtime / dt
#        self.max_tr_dur = sim_dict['max_tr_dur'] / 1000.
        self.sim_env = env_dict['sim_env']
        
        # calculate points from obstacle data for collision detection
        obs_dict = env_dict["environment"]["obstacles"]
        self.obstacle_list = []
        if obs_dict["flag"]:
            for center, vert, horiz in zip(obs_dict["centers"], obs_dict["vert_lengths"], obs_dict["horiz_lengths"]):
                delta_y = vert / 2. # Get the length and width 
                delta_x = horiz / 2.  # as distances from the center point
                
                ll = (center[0] - delta_x, center[1] - delta_y) # lower left
                lr = (center[0] + delta_x, center[1] - delta_y) # lower right
                ur = (center[0] + delta_x, center[1] + delta_y) # upper right
                ul = (center[0] - delta_x, center[1] + delta_y) # upper left
                
                # Note that the list of points needs to be given IN ORDER!
                # Otherwise, the diagonals of the rectangle will be treated as the obstacle borders
                self.obstacle_list.append([ll, lr, ur, ul])
                
        self.env_limit_dic = env_dict['environment'][self.sim_env]
        if self.sim_env == 'openfield':
            self.xmin_position = float(self.env_limit_dic['xmin_position'])
            self.xmax_position = float(self.env_limit_dic['xmax_position'])
            self.ymin_position = float(self.env_limit_dic['ymin_position'])
            self.ymax_position = float(self.env_limit_dic['ymax_position'])
        elif self.sim_env == 'tmaze':
            self.xmin_position = float(self.env_limit_dic['xmin_position'])
            self.xmax_position = float(self.env_limit_dic['xmax_position'])
            self.ymin_position = float(self.env_limit_dic['ymin_position'])
            self.ymax_position = float(self.env_limit_dic['ymax_position'])
            
            cw = self.env_limit_dic['corridor_width']
            gaw = self.env_limit_dic['goal_arm_width']
            ov = 0.1  # The distance to 'overshoot' environment boundaries to ensure the agent can't escape the tmaze by sliding along the border
            
            # These points can be added to the obstacle list to check for collision detection
            p1 = (self.xmin_position - ov, self.ymax_position - gaw)
            p2 = (-cw / 2, self.ymax_position - gaw)
            p3 = (-cw / 2, self.ymin_position - ov)
            p4 = (cw / 2, self.ymin_position - ov)
            p5 = (cw / 2, self.ymax_position - gaw)
            p6 = (self.xmax_position + ov, self.ymax_position - gaw)
            
            self.obstacle_list.append([p1, p2])
            self.obstacle_list.append([p2, p3])
            self.obstacle_list.append([p4, p5])
            self.obstacle_list.append([p5, p6])
            
        else:
            print("environment {} undefined".format(self.sim_env))

        # self.start_x = float(sim_dict['start']['x_position'])
        # self.start_y = float(sim_dict['start']['y_position'])

        # self.hide_goal = sim_dict['goal']['hide_goal']
        # self.reward_recep_field = sim_dict['goal']['reward_recep_field']
        # self.goal_x = sim_dict['goal']['x_position']
        # self.goal_y = sim_dict['goal']['y_position']

        self.data_path = sim_dict['data_path']
