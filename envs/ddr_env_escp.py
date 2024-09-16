import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import warnings

import dill
import gym
import numpy as np
# import pygame
from gym import spaces
from scipy.integrate import solve_ivp

from utils.ddr_constants import KIN_ACTION_DICT, DYNAMIC_CONSTANTS, DYN_ACTION_DICT, DYNAMIC_CONSTANTS_RANGE, \
    HARD_DYNAMIC_CONSTANTS_RANGE, DYNAMIC_CONSTANTS_NORM_FACTS
from utils.ddr_utils import limit_angles, right_hand_side, create_lognormal_dist, seed_everything, \
    generate_cone_smartly, calculate_dist_of_points


class KinDiffRobot(gym.Env):
    def __init__(self, env_config, **kwargs):
        super(gym.Env).__init__()
        # Environment variables
        self.dt = env_config.get("dt", 0.1)
        self.max_dist_from_goal = env_config.get("max_dist_from_goal", 3.)
        self.can_go_around_gate = env_config.get("can_go_around_gate", True)
        max_dist = env_config.get("max_travelled_dist", 2 * self.max_dist_from_goal * 2)
        self.max_travelled_dist_of_wheels = max_dist if max_dist else 2 * self.max_dist_from_goal * 2
        self.robot_state = np.zeros((15,))  # [x, y, θ, ϕ_1, ϕ_2, Dx, Dy, Dθ, Dϕ_1, Dϕ_2, c1, c2, c3, i_1, i_2]
        self.robot_x, self.robot_y, self.robot_theta = 0., 0., 0.
        self.prev_ang_between_robot_and_gate = 0.
        self.travelled_dist_of_wheels = 0.
        self.prev_dist_from_goal = self.max_dist_from_goal
        self.standing_still_counter = 0
        self.goal_point = (1., 0., 0.)
        self.gate_width = 0.5
        self.num_obstacles = env_config.get("num_obstacles", 0)
        self.goal_cones = [(1., 0.), ] * 2
        self.obstacles = [(1., 0.), ] * self.num_obstacles
        self.cone_radius = 0.075
        self.current_step = 0
        self.action_dict = KIN_ACTION_DICT
        self.action_multiplier = 1.
        self.action_space_type = env_config.get("action_space_type", "discrete_simple")
        self.observation_space_type = env_config.get("observation_space_type", "goal_robot")
        self.np_random, _ = self._np_random(env_config.get("seed", 42))

        self.use_hard_scenarios = env_config.get("use_hard_scenarios", False)
        self.system_dynamics = list(DYNAMIC_CONSTANTS.values())
        self.rendering = env_config.get("rendering", False)
        self.use_final_error_reward = env_config.get("use_final_error_reward", True)
        self.use_step_reward = env_config.get("use_step_reward", True)

        # Pygame variables
        self.screen = None
        self.robot_img = None
        self.screen_size = 500
        self.scale = self.screen_size / (self.max_dist_from_goal * 2)

        # Creating action and observation space
        self.action_space = self._create_action_space()
        self.observation_space, self.observation_space_type, self.normalizing_factors = self._create_observation_space()

    def _np_random(self, seed):
        # Based on: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/utils/seeding.py
        seed_seq = np.random.SeedSequence(seed)
        np_seed = seed_seq.entropy
        rng = np.random.Generator(np.random.PCG64(seed_seq))
        return rng, np_seed

    def reset(self, seed=None, **kwargs):
        # Reset robot dynamics
        self.robot_state = np.zeros((15,))
        self.robot_x, self.robot_y, self.robot_theta = 0., 0., np.deg2rad(90 - self.np_random.integers(-85, 85))
        # Generating random heading to the robot
        self.robot_state[2] = self.robot_theta
        self.current_step = 0
        self.travelled_dist_of_wheels = 0.
        self.standing_still_counter = 0
        self._robot_reset()
        # Generating the gate and the obstacles
        self.goal_point = self._generate_goal()
        self.goal_cones = self._generate_goal_cones()
        self.obstacles = self._generate_obstacles()
        # Calculating the current observation
        obs = self._create_observation()
        self.prev_ang_between_robot_and_gate, self.prev_dist_from_goal = self._prepare_goal_state_for_termination()
        state = self._normalize_obs(obs)
        info = self._get_info_dict(cause=None)

        assert any(np.isnan(state)) is False, f"State contains NaNs {state}"
        assert any(np.isinf(state)) is False, f"State contains Infs {state}"
        assert (np.all(state <= 1.0) and np.all(state >= -1.0)), \
            f"The observation is not between -1 and 1. {state}"
        assert state.dtype == np.float32, f"The state type is not float32. {state.dtype}"

        return state, info

    def _robot_reset(self):
        pass

    def step(self, action):
        # Checking if the action is in the action space
        if not self.action_space.contains(action):
            if isinstance(self.action_space, spaces.Box):
                # warnings.warn(f"Action {action} is not in the action space {self.action_space}")
                action = np.clip(action, self.action_space.low, self.action_space.high)
            else:
                raise ValueError(f"Action {action} is not in the action space {self.action_space}")

        self.current_step += 1
        action = [element * self.action_multiplier for element in (self.action_dict[action]
                                                                   if "discrete" in self.action_space_type else action)]
        robot_state, t_ = self._robot_step(action)
        for i in range(1, robot_state.shape[-1]):
            self.robot_state = robot_state[:, i]
            # Updating the robot position
            self.robot_x = self.robot_state[0]
            self.robot_y = self.robot_state[1]
            self.robot_theta = self.robot_state[2] = limit_angles(self.robot_state[2])
            self.robot_state[3] = limit_angles(self.robot_state[3])
            self.robot_state[4] = limit_angles(self.robot_state[4])
            self.calculate_travelled_dist_of_wheels(dt=t_[i] - t_[i - 1])
            terminated, cause = self._check_termination()
            if terminated:
                break

        # Checking if the robot is standing still should not be calculated in the inner steps
        cause = "standing_still" if self._check_standing_still() else cause
        terminated = True if cause is not None else False


        obs = self._create_observation()
        reward = self._calculate_reward(cause)
        reward = np.float32(reward)
        # Saving the angle between the robot and the gate for the next step
        self.prev_ang_between_robot_and_gate, self.prev_dist_from_goal = self._prepare_goal_state_for_termination()
        assert cause in ["success", None] if reward >= 0 else True
        state = self._normalize_obs(obs)
        info = self._get_info_dict(cause=cause)

        assert any(np.isnan(state)) is False, f"State contains NaNs {state}"
        assert any(np.isinf(state)) is False, f"State contains Infs {state}"
        assert (np.all(state <= 1.0) and np.all(state >= -1.0)), \
            f"The observation is not between -1 and 1. {state}"
        assert state.dtype == np.float32, f"The state type is not float32. {state.dtype}"
        assert reward.dtype == np.float32, f"The state type is not float32. {reward.dtype}"
        assert (-1.0 <= reward <= 1.0), f"The reward is not between -1 and 1. {reward}"

        return state, reward, terminated, cause in ["slow", "standing_still"], info

    def _robot_step(self, action):
        # Calculating the new position of the robot
        velocity, w = action
        theta = self.robot_theta + w * self.dt
        v_x = velocity * np.cos(theta)
        v_y = velocity * np.sin(theta)
        x = self.robot_x + v_x * self.dt
        y = self.robot_y + v_y * self.dt

        # Calculating the angle and velocity of the wheels
        # Source: https://www.roboticsbook.org/S52_diffdrive_actions.html
        d_phi_l = velocity / self.system_dynamics[3] - (self.system_dynamics[5] / 2) * (w / self.system_dynamics[3])
        d_phi_r = velocity / self.system_dynamics[3] + (self.system_dynamics[5] / 2) * (w / self.system_dynamics[3])
        phi_l = self.robot_state[4] + d_phi_l * self.dt
        phi_r = self.robot_state[5] + d_phi_r * self.dt
        raise NotImplementedError
        robot_state = np.array([x, y, theta, phi_r, phi_l, v_x, v_y, w, d_phi_l, d_phi_r, 0., 0., 0., 0., 0.],
                               dtype=np.float32)

        assert any(np.isnan(robot_state)) is False, f"State contains NaNs {robot_state}"
        assert any(np.isinf(robot_state)) is False, f"State contains Infs {robot_state}"

        return robot_state

    def _create_action_space(self):
        if self.__class__ == KinDiffRobot and "discrete" not in self.action_space_type:
            raise ValueError("KinDiffRobot can only have discrete action space")
        if "discrete_simple" in self.action_space_type:
            # Only left or right wheel has voltage or both have the same voltage
            return spaces.Discrete(3)
        elif "discrete_complex" in self.action_space_type:
            # Wheels can have positive/negative voltage, both can have no voltage or the same positive/negative voltage
            return spaces.Discrete(len(DYN_ACTION_DICT))
        elif "continuous" in self.action_space_type:
            # Wheels can have any voltage between -1 and 1 (later it will be multiplied by the max voltage)
            return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            raise ValueError("action_space_type should be: discrete_simple, discrete_complex, continuous")

    def _create_observation_space(self):
        obs_type = self.observation_space_type.split("_") if isinstance(self.observation_space_type, str) else \
            self.observation_space_type
        # Assert observation formulation
        if 'obstacles' in obs_type and self.num_obstacles == 0:
            warnings.warn("There are no obstacles in the environment, so removed from the observation space")
            obs_type.remove('obstacles')
        if 'goal' not in obs_type:
            warnings.warn("Goal should be in the observation space type, so added silently.")
            obs_type.append('goal')

        # Ensuring the order of the observations
        obs_type = [obs_i for obs_i in ["goal", "gates", "obstacles", "robot", "dynamics"] if obs_i in obs_type]

        # Creating the observation space and the normalizing factors
        normalizing_factors = []
        for obs_i in obs_type:
            if obs_i == "goal":
                normalizing_factors += [self.max_dist_from_goal, np.pi, np.pi]
            elif obs_i == "gates":
                normalizing_factors += 2 * [self.max_dist_from_goal, np.pi]
            elif obs_i == "obstacles":
                normalizing_factors += len(self.obstacles) * [self.max_dist_from_goal, np.pi]
            elif obs_i == "robot":
                robot_norm_factors = [self.max_dist_from_goal / self.dt,
                                      self.max_dist_from_goal / self.dt,
                                      np.pi / self.dt, np.pi / self.dt, np.pi / self.dt,
                                      300., 300.,
                                      ]
                normalizing_factors += robot_norm_factors
                self.robot_dyn_dim = len(robot_norm_factors)
            elif obs_i == "dynamics":
                normalizing_factors += DYNAMIC_CONSTANTS_NORM_FACTS
            else:
                raise ValueError(f"Observation type {obs_i} is not supported")

        observation_space = spaces.Box(low=-1, high=1, shape=(len(normalizing_factors),), dtype=np.float32)
        normalizing_factors = np.array(normalizing_factors, dtype=np.float32)
        # Resetting due to division by zero
        normalizing_factors[normalizing_factors == 0] = 1

        return observation_space, obs_type, normalizing_factors

    def _observe_goal_state(self):
        # Calculating the distance between the robot and the center of the gate
        dist_from_goal = self._calculate_distance_from_goal()
        # Calculating the angle between the robot and the gate
        gate_robot_ang = limit_angles(self._calculate_ang_btw_rotating_obj_and_point(point_x=self.robot_x,
                                                                                     point_y=self.robot_y,
                                                                                     rot_obj_x=self.goal_point[0],
                                                                                     rot_obj_y=self.goal_point[1],
                                                                                     rot_obj_theta=self.goal_point[2]))
        obs = np.array([dist_from_goal, gate_robot_ang, self._get_angle_to_perpendicular_w_gate()], dtype=np.float32)
        return obs

    def _observe_robot_state(self):
        # Extracting the useful information from robot_state (leave out: x, y, θ, ϕ_1, ϕ_2 and c1, c2, c3)
        # robot_state: [x, y, θ, ϕ_1, ϕ_2, Dx, Dy, Dθ, Dϕ_1, Dϕ_2, c1, c2, c3, i_1, i_2]
        robot_state = np.concatenate((self.robot_state[5:10], self.robot_state[-2:]))
        return robot_state

    def _observe_obstacles(self, obstacle_list):
        obs_of_cones = []

        for cone in obstacle_list:
            dist_from_cone = calculate_dist_of_points((self.robot_x, self.robot_y), cone)
            # Transform the point-wise distance to bumper-to-bumper distance
            dist_from_cone -= (self.cone_radius + self.system_dynamics[5]/2)
            dist_from_cone = max(dist_from_cone, 0.)
            angle_btw_robot_and_cone = self._calculate_ang_btw_rotating_obj_and_point(point_x=cone[0],
                                                                                      point_y=cone[1],
                                                                                      rot_obj_x=self.robot_x,
                                                                                      rot_obj_y=self.robot_y,
                                                                                      rot_obj_theta=self.robot_theta)
            angle_btw_robot_and_cone = limit_angles(angle_btw_robot_and_cone)
            obs_of_cones.extend([dist_from_cone, angle_btw_robot_and_cone])

        return np.array(obs_of_cones, dtype=np.float32)

    def _create_observation(self):
        obs = []
        for obs_type in self.observation_space_type:
            if obs_type == "goal":
                obs += [self._observe_goal_state()]
            elif obs_type == "gates":
                obs += [self._observe_obstacles(self.goal_cones)]
            elif obs_type == "obstacles":
                obs += [self._observe_obstacles(self.obstacles)]
            elif obs_type == "robot":
                obs += [self._observe_robot_state()]
            elif obs_type == "dynamics":
                obs += [self.system_dynamics]  # [np.array(list(DYNAMIC_CONSTANTS.values()), dtype=float)]
        return np.concatenate(obs).astype(np.float32)

    def _normalize_obs(self, obs):
        # Normalizing the observation between -1 and 1
        norm_obs = obs / self.normalizing_factors
        norm_obs = np.clip(norm_obs, -1., 1.)
        # Asserting obs is between -1 and 1
        assert (np.all(norm_obs <= 1.0) and np.all(norm_obs >= -1.0)), \
            f"The observation is not between -1 and 1. {obs, norm_obs}"
        return norm_obs

    def _calculate_distance_from_goal(self):
        # Calculating the distance between the robot and the center of the gate
        dist_from_goal = calculate_dist_of_points((self.robot_x, self.robot_y), self.goal_point)
        dist_from_goal = min(dist_from_goal, self.max_dist_from_goal)
        return dist_from_goal

    def _calculate_ang_btw_rotating_obj_and_point(self, point_x, point_y, rot_obj_x, rot_obj_y, rot_obj_theta):
        # Calculating the vector from rotating object to point
        rot_obj_to_point = np.array([point_x - rot_obj_x, point_y - rot_obj_y])
        # Getting the normal vector of the rotating object
        rot_obj_normal = np.array([np.cos(rot_obj_theta - np.pi / 2), np.sin(rot_obj_theta - np.pi / 2)])
        # Calculating the angle between the point and the rotating object
        angle = np.arccos(np.dot(rot_obj_to_point, rot_obj_normal) / (np.linalg.norm(rot_obj_to_point)
                                                                      * np.linalg.norm(rot_obj_normal)))
        # Calculating the sign of the angle
        sign = np.sign(np.cross(rot_obj_normal, rot_obj_to_point))
        angle = sign * angle
        if np.isnan(angle):
            angle = 0.0
        return angle

    def calculate_travelled_dist_of_wheels(self, dt):
        # Travelled distance of one wheel = angle * radius
        right_dist = abs(self.robot_state[8] * dt * self.system_dynamics[3])
        left_dist = abs(self.robot_state[9] * dt * self.system_dynamics[3])

        self.travelled_dist_of_wheels += right_dist + left_dist

    def _check_termination(self):
        angle_between_robot_and_gate, dist_from_goal = self._prepare_goal_state_for_termination()

        assert dist_from_goal <= self.max_dist_from_goal, "The dist from the gate is greater than the maximum possible."

        # Determine whether the robot got too far from the gate
        is_far = dist_from_goal == self.max_dist_from_goal

        # Determine whether robot collided into one of the cones
        is_collision = self._check_collision_with_cones(self.goal_cones + self.obstacles)

        missed_gate, success, wrong_direction = self._check_gate_passing(angle_between_robot_and_gate, dist_from_goal)

        # Defining the cause of termination
        cause = "far" if is_far else "collision" if is_collision else "missed" if missed_gate else "wrong_direction" \
            if wrong_direction else "slow" if self.travelled_dist_of_wheels >= self.max_travelled_dist_of_wheels else \
            "success" if success else None

        terminated = True if cause is not None else False

        return terminated, cause

    def _check_gate_passing(self, angle_between_robot_and_gate, dist_from_goal):
        # Determine whether the robot crossed the gate from the wrong direction
        wrong_direction = dist_from_goal <= self.gate_width and \
                          (abs(angle_between_robot_and_gate) <= np.pi / 2) and (
                                  abs(self.prev_ang_between_robot_and_gate) > np.pi / 2)
        # Determine whether the robot crossed the gate successfully or missed it on its sides
        if (abs(angle_between_robot_and_gate) >= np.pi / 2) and (abs(self.prev_ang_between_robot_and_gate) < np.pi / 2):
            success = dist_from_goal < self.gate_width
            missed_gate = dist_from_goal > self.gate_width and not self.can_go_around_gate
        else:
            success, missed_gate = False, False
        return missed_gate, success, wrong_direction

    def _prepare_goal_state_for_termination(self):
        dist_from_goal, angle_between_robot_and_gate, _ = self._observe_goal_state()
        return angle_between_robot_and_gate, dist_from_goal

    def _calculate_reward(self, cause):
        # Determining the reward of the step
        reward_ = 0 if not self.use_step_reward \
            else (self.prev_dist_from_goal - self._calculate_distance_from_goal()) / self.max_dist_from_goal
        if cause == "success":
            reward_ = 1. if not self.use_final_error_reward else 1. - self._calculate_final_reward()
        elif cause in ["far", "collision", "missed", "wrong_direction", "slow", "standing_still"]:
            reward_ = -1.

        return reward_

    def _calculate_final_reward(self):
        dist_, angle_, _ = self._observe_goal_state()
        misplacement_x = abs(np.cos(angle_ - np.pi / 2) * dist_)
        misplacement_theta = abs(self._get_angle_to_perpendicular_w_gate())
        # Creating some thresholds for the final error
        misplacement_x = 0. if misplacement_x < 0.05 else misplacement_x
        misplacement_theta = 0. if misplacement_theta < np.deg2rad(10.) else misplacement_theta
        reward = min(misplacement_x / (self.gate_width * 2 - self.cone_radius)
                     + misplacement_theta / (np.pi / 3), 1)
        return reward

    def _get_info_dict(self, cause):
        info_ = {"cause": cause,
                 "dynamics": self.system_dynamics,
                 "dist_from_goal": self._calculate_distance_from_goal(),
                 "angle_to_goal": np.rad2deg(self._get_angle_to_perpendicular_w_gate()),
                 "travelled_dist": self.travelled_dist_of_wheels,
                 "steps": self.current_step,
                 "dt": self.dt,
                 "robot_state": self.robot_state,
                 }
        return info_

    def _generate_goal(self):
        # Generating random position and angle for the goal point (for the gate)
        goal_x, goal_y = 0, self.np_random.uniform(low=(max(0.5 * self.max_dist_from_goal, 1.2)),
                                                   high=(0.95 * self.max_dist_from_goal))
        goal_angle = self.np_random.uniform(low=-np.pi / 4, high=np.pi / 4)
        return goal_x, goal_y, goal_angle

    def _generate_goal_cones(self):
        cones = []
        # Calculating the coordinates of the cones defining the gate
        for cone in [(-self.gate_width, 0), (self.gate_width, 0)]:
            # Rotating the cones around the origin
            x = cone[0] * np.cos(self.goal_point[2]) - cone[1] * np.sin(self.goal_point[2])
            y = cone[0] * np.sin(self.goal_point[2]) + cone[1] * np.cos(self.goal_point[2])
            # Shifting the rotated cone to the position of the gate
            x += self.goal_point[0]
            y += self.goal_point[1]
            # Storing the coordinates of the cone
            cones.append((x, y))
        return cones

    def _generate_obstacles(self):
        cones = []
        trials_ = 0
        # Generating obstacle cones
        while len(cones) < self.num_obstacles and trials_ < 100:
            trials_ += 1
            # Generating random position for the cone
            if len(cones) < 1:
                cone_candidates = generate_cone_smartly(y=self.goal_point[1], alpha=self.robot_theta,
                                                        min_dist=self.system_dynamics[5], np_random=self.np_random)[
                                  :self.num_obstacles]
            else:
                cone_candidates = []
            cone_candidates += generate_cone_smartly(y=self.goal_point[1],
                                                     alpha=self.np_random.uniform(-np.pi / 2, np.pi / 2),
                                                     min_dist=self.system_dynamics[5], np_random=self.np_random)[1:]

            # If the generated cone is too close to another cone, it will not be added to the cone list
            enough_distance_between_cones = [True] * len(cone_candidates)
            for i, (x, y) in enumerate(cone_candidates):
                dist_to_goal = calculate_dist_of_points(self.goal_point, (x, y))
                dist_to_robot = calculate_dist_of_points((self.robot_x, self.robot_y), (x, y)) - self.cone_radius
                dist_to_goal_cone1 = calculate_dist_of_points(self.goal_cones[0], (x, y)) - 2 * self.cone_radius
                dist_to_goal_cone2 = calculate_dist_of_points(self.goal_cones[1], (x, y)) - 2 * self.cone_radius

                if dist_to_goal < max(self.gate_width, 1.2 * self.system_dynamics[5]) or dist_to_robot < 1.2 * \
                        self.system_dynamics[5] or dist_to_goal_cone1 < 1.2 * self.system_dynamics[
                    5] or dist_to_goal_cone2 < 1.2 * self.system_dynamics[5]:
                    enough_distance_between_cones[i] = False
            for idx, is_good in enumerate(enough_distance_between_cones):
                if is_good and len(cones) < self.num_obstacles:
                    cones.append(cone_candidates[idx])

        if len(cones) < self.num_obstacles:
            print(f"Number of obstacles is not correct. {len(cones)} because based on the trials {trials_}")
            # adding dummy cones due to infeasible scenario
            cones = cones + (self.goal_cones * (self.num_obstacles - len(cones)))[:(self.num_obstacles - len(cones))]
        assert len(cones) == self.num_obstacles, f"Number of obstacles is not correct. {len(cones)}"
        return cones

    def _check_collision_with_cones(self, cones):
        # Determine whether the robot collided into one of the cones
        for cone in cones:
            if calculate_dist_of_points((self.robot_x, self.robot_y), cone) < self.cone_radius + \
                    self.system_dynamics[5] / 2:
                return True
        return False

    def _check_standing_still(self):
        # If the velocity is low enough, and the robot is not turning, the robot is standing still
        if np.sqrt(self.robot_state[5] ** 2 + self.robot_state[6] ** 2) < 0.1 and abs(self.robot_state[7]) < 0.3:
            self.standing_still_counter += 1

        # If the robot is standing still for too long, return too so the episode can be terminated
        return self.standing_still_counter >= 50

    def _get_angle_to_face_the_gate(self):
        # Calculating "how many angles" should the robot turn to face the center of the gate
        # If the gate is on the right, it is negative, if it is on the left, it is positive
        if (self.goal_point[1] - self.robot_y) != 0:
            angle_to_gate = np.arctan((self.goal_point[0] - self.robot_x) / (self.robot_y - self.goal_point[1]))
        else:
            angle_to_gate = np.arctan(
                (self.goal_point[0] - self.robot_x) / (self.robot_y - self.goal_point[1] + 0.0000001))

        # If it should turn clockwise, it is negative, if it should turn counterclockwise, it is positive
        angle_to_gate = angle_to_gate + (np.pi / 2) - self.robot_state[2]
        angle_to_gate = limit_angles(angle_to_gate)

        return angle_to_gate

    def _get_angle_to_perpendicular_w_gate(self):
        # How many angles should the robot turn to face perpendicular to the gate
        gate_vector = np.array([np.cos(self.goal_point[2] + np.pi / 2), np.sin(self.goal_point[2] + np.pi / 2)])
        robot_vector = np.array([np.cos(self.robot_theta), np.sin(self.robot_theta)])
        angle_to_perpendicular = np.arccos(np.dot(gate_vector, robot_vector) / (
                np.linalg.norm(gate_vector) * np.linalg.norm(robot_vector)))
        sign = np.sign(np.cross(gate_vector, robot_vector))
        angle_to_perpendicular = sign * limit_angles(angle_to_perpendicular)

        if np.isnan(angle_to_perpendicular):
            angle_to_perpendicular = 0.0

        return angle_to_perpendicular

    # def _create_pygame_visualization(self, width: int, height: int):
    #     # Creating the PyGame window
    #     pygame.init()
    #     self.screen = pygame.display.set_mode((width, height))
    #     pygame.display.set_caption("Robot visualisation")
    #     # Loading the robot image
    #     filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot.png")
    #     self.robot_img = pygame.image.load(filename).convert()
    #     self.robot_img.set_colorkey((255, 255, 255))
    #     # Loading the arrow image
    #     filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arrow.png")
    #     self.arrow_img = pygame.image.load(filename).convert()
    #     self.arrow_img.set_colorkey((255, 255, 255))
    #     self.arrow_img = pygame.transform.scale(self.arrow_img, (50, 25))

    def render(self, mode="human"):
        pass
    #     if self.rendering:
    #         # If the PyGame window visualizer is not created, create it
    #         if self.screen is None:
    #             self._create_pygame_visualization(width=self.screen_size, height=self.screen_size)
    #
    #         # Clearing the screen
    #         self.screen.fill((230, 230, 230))
    #
    #         # Scale factor
    #         x, y = self.screen_size / 2, self.screen_size / 2
    #
    #         # Rotating and displaying the arrow
    #         image_rotated = pygame.transform.rotate(self.arrow_img, np.rad2deg(self.goal_point[2] + np.pi / 2))
    #         image_rect = image_rotated.get_rect(center=(x, y))
    #         self.screen.blit(image_rotated, image_rect)
    #
    #         # rescaling the robot image with robot width
    #         image_rotated = pygame.transform.scale(self.robot_img, (int(self.scale * self.system_dynamics[5]),
    #                                                                 int(self.system_dynamics[5] * (
    #                                                                         18 / 48) * self.scale)))
    #         # Rotating and displaying the robot image
    #         pygame.draw.circle(self.screen, (0, 0, 0), (x + self.scale * (self.robot_x - self.goal_point[0]),
    #                                                     y - self.scale * (self.robot_y - self.goal_point[1])),
    #                            radius=(self.system_dynamics[5] / 2 * self.scale), width=2, )
    #         image_rotated = pygame.transform.rotate(image_rotated, np.rad2deg(self.robot_state[2] - np.pi / 2))
    #         image_rect = image_rotated.get_rect(center=(x + self.scale * (self.robot_x - self.goal_point[0]),
    #                                                     y - self.scale * (self.robot_y - self.goal_point[1])))
    #         self.screen.blit(image_rotated, image_rect)
    #
    #         # Drawing the goal point in the center of the screen (red circle)
    #         pygame.draw.circle(self.screen, (255, 0, 0), (x, y), radius=(self.cone_radius * self.scale), width=0)
    #
    #         # Drawing gate cones (black circles) and obstacles (blue circles)
    #         for i, cone in enumerate(self.goal_cones + self.obstacles):
    #             pygame.draw.circle(self.screen, color=(0, 0, 0 if i < len(self.goal_cones) else 255),
    #                                radius=(self.cone_radius * self.scale), width=0,
    #                                center=(x + self.scale * (cone[0] - self.goal_point[0]),
    #                                        y - self.scale * (cone[1] - self.goal_point[1])))
    #
    #         # Drawing the maximum distance that the roboto can go from the gate (black circle)
    #         pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius=self.max_dist_from_goal * self.scale, width=2)
    #         # Writing the distance to gate in the top right corner of the screen
    #         font = pygame.font.Font('freesansbold.ttf', 10)
    #         texts_to_display = [f"Distance to gate: {self.prev_dist_from_goal:.2f}",
    #                             f"Angle to gate: {self.prev_ang_between_robot_and_gate:.2f}",
    #                             f"Angle to face: {self._get_angle_to_perpendicular_w_gate():.2f}"]
    #         for i, text in enumerate(texts_to_display):
    #             text = font.render(text, True, (0, 0, 0))
    #             textRect = text.get_rect()
    #             textRect.topleft = (10, 10 + 15 * i)
    #             self.screen.blit(text, textRect)
    #
    #         # Drawing system dynamics onto the left side of the screen
    #         text = font.render(f"System dynamics: {str([f'{a:.2f}' for a in self.system_dynamics])}", True, (0, 0, 0))
    #         textRect = text.get_rect()
    #         textRect.topleft = (10, self.screen_size - 10)
    #         self.screen.blit(text, textRect)
    #
    #         # Stopping the program via closing the PyGame window
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 pygame.quit()
    #                 sys.exit()
    #
    #         pygame.display.update()
    #
    #     return True
    #
    # def stop(self):
    #     if self.rendering:
    #         pygame.quit()


class DynDiffRobot(KinDiffRobot, gym.Env):
    def __init__(self, env_config):
        gym.Env.__init__(self)
        KinDiffRobot.__init__(self, env_config)

        # Loading the RHS of the derived robot model
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_rhs.func")
        with open(filename, "rb") as file:
            dill.settings['recurse'] = True
            self.rhs_func = dill.load(file)
        self.action_dict = DYN_ACTION_DICT
        self.action_multiplier = 50.
        self.range = env_config.get("random_range", 1.)
        self.randomize_every_n_steps = env_config.get("randomize_every_n_steps", None)

    def _robot_step(self, action):
        if self.randomize_every_n_steps and self.range and self.current_step and \
                self.current_step % self.randomize_every_n_steps == 0:
            self._generate_random_dynamics()
        # Stepping the robot dynamics
        us = np.vstack((action, action))
        xs = solve_ivp(right_hand_side, [0., self.dt], self.robot_state,
                       args=([0., self.dt], us, self.system_dynamics, self.rhs_func),
                       t_eval=None)
        result = xs.y

        assert not np.isnan(result).any(), "Robot state contains NaNs"
        assert not np.isnan(result).any(), "Robot state contains Infs"

        return result, xs.t

    def _robot_reset(self):
        if self.range > 0.:
            self._generate_random_dynamics()
        assert self.system_dynamics[5] < self.gate_width * 2, "The robot is too big for the gate"

    def _generate_random_dynamics(self, just_sampling_tasks=False):
        ranges_of_dynamics = HARD_DYNAMIC_CONSTANTS_RANGE if self.use_hard_scenarios else DYNAMIC_CONSTANTS_RANGE

        system_dynamics = []
        for name, param in ranges_of_dynamics.items():
            if isinstance(param, (float, int)):
                system_dynamics.append(param)
            elif isinstance(param, tuple):
                if self.range == 1.:
                    value = create_lognormal_dist(min_value=param[0], max_value=param[1], size=1,
                                                  np_random=self.np_random)
                elif self.range == 0.:
                    value = DYNAMIC_CONSTANTS[name]
                else:
                    mean = (param[1] - param[0]) / 2
                    std = (param[1] - mean) * self.range
                    value = create_lognormal_dist(mean_value=mean, std_value=std, size=1,
                                                  np_random=self.np_random)
                value = max(min(value, param[1]), param[0])
                system_dynamics.append(value)
            else:
                raise ValueError("The dynamic constants range is not defined correctly.")
        # adding the second motor constant to be the same as the first
        # system_dynamics.append(system_dynamics[-1])
        assert len(system_dynamics) == len(DYNAMIC_CONSTANTS.keys())

        if just_sampling_tasks:
            return system_dynamics
        else:
            self.system_dynamics = system_dynamics


class DynDiffRobotESCP(DynDiffRobot):
    def __init__(self):
        # Environment configuration
        env_config = {"num_obstacles": 1,
                      "action_space_type": "continuous",
                      "observation_space_type": "goal_gates_obstacles_robot",
                      "random_range": 1.,
                      "seed": 42,}
        super(DynDiffRobotESCP, self).__init__(env_config)

        # ESCP related variables
        self.diy_env = True
        self.fix_env = None
        # Removing 0s from the normalization factors, to avoid division by zero
        self.dyn_norm_factors = [1.0 if i == 0 else i for i in DYNAMIC_CONSTANTS_NORM_FACTS]

    def sample_tasks(self, n_tasks):
        # Generates randomized parameter sets (n_tasks = number of different meta-tasks needed)
        tasks = []
        task_set = set()
        while len(task_set) < n_tasks:
            task_set.add(tuple(self._generate_random_dynamics(just_sampling_tasks=True)))
        for item in task_set:
            tasks.append(list(item))

        return tasks

    def _robot_reset(self):
        # The dynamics will not be regenerated by calling the super().reset() function
        pass

    def reset(self):
        if self.fix_env is None:
            self._generate_random_dynamics()
            # In GridWorld:
            # self.renv_flag = (random.randint(-self.max_offset, self.max_offset),
            #                   random.randint(-self.max_offset, self.max_offset))
            # self.env_flag = self.renv_flag
        else:
            self.system_dynamics = self.fix_env
            # In GridWorld:
            # self.renv_flag = self.env_flag = self.fix_env

        reset_state, _ =  super(DynDiffRobotESCP, self).reset()

        return reset_state

    def step(self, action):
        # ESCP uses gym instead of gymnasium, so step function should not return the truncated flag
        state, reward, done, _, info = super(DynDiffRobotESCP, self).step(action)
        return state, reward, done, info

    def set_task(self, task):
        self.system_dynamics = self.fix_env = task
        # In GridWorld:
        # self.set_fix_env((task[0], task[0]))
        # def set_fix_env(self, fix_env):
        #     self.renv_flag = self.env_flag = self.fix_env = fix_env

    @property
    def env_parameter_vector_(self):
        # In GridWorld: return [self.env_flag[0] / self.max_offset, self.env_flag[1] / self.max_offset]
        # self.env_flag is a task randomly generated in the init function
        return [self.system_dynamics[i] / self.dyn_norm_factors[i] for i in range(len(self.system_dynamics))]

    @property
    def env_parameter_length(self):
        # In GridWorld: return 2
        return len(self.system_dynamics)


from gym.envs.registration import register
register(id='DynDiffRobotESCP-v0', entry_point=DynDiffRobotESCP)

if __name__ == '__main__':
    seed_everything(42)

    env = gym.make('DynDiffRobotESCP-v0')
    # env = DynDiffRobot(env_config=default_robot_config)

    env.reset()
    env.action_space.seed(42)

    for episode in range(10):
        _done = False
        env.reset()
        while not _done:
            _action = env.action_space.sample()
            _state, _reward, _done, _info = env.step(_action)
            env.render()
            time.sleep(0.1)
        print("Required steps: ", env.current_step, " Reward: ", _reward, " Info: ", _info["dynamics"])

    env.close()
