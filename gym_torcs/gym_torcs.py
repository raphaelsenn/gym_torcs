from typing import Dict, List, Tuple, Union

import copy
import os
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import gym_torcs.snakeoil3_gym as snakeoil3
from gym_torcs.constants import (
    PORT,
    TERMINATION_LIMIT_PROGRESS,
    TERMINAL_JUDGE_START,   
    
    DEFAULT_SPEED,
    MAX_FOCUS,
    MAX_OPPONENTS,
    MAX_TRACK,

    OBS_NOISE_STD,
    ACTION_NOISE_STD
)


class TorcsEnv(gym.Env):
    def __init__(self, throttle: bool=False, max_episode_steps: int = 100_000) -> None:
        super().__init__()

        self.throttle = throttle
        self.max_episode_steps = max_episode_steps 
        self.initial_run = True
        self.initial_reset = True
        self.client = None 
        self.time_step = None

        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        high = np.array([1.0, np.inf, np.inf, np.inf, 1.0, np.inf, 1.0, np.inf])
        low = np.array([0.0, -np.inf, -np.inf, -np.inf, 0.0, -np.inf, 0.0, -np.inf])
        self.observation_space = spaces.Box(low=low, high=high)

        self._launch_torcs()

    def reset(self, relaunch = False, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed) 

        if self.initial_reset is False:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## NOTE: Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self._launch_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Set start timestep
        self.time_step = 0
        
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=PORT, vision=False)  # Open new UDP in vtorcs

        # Get initial observation from torcs
        self.client.get_servers_input()     # Initial input from torcs
        raw_obs = self.client.S.d           # Current full-observation from torcs
        self.initial_reset = False

        return self._get_obs(raw_obs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # Apply some noise to action
        noise = np.random.normal(0, ACTION_NOISE_STD, size=action.shape)
        action = (action + noise).clip(-1.0, 1.0)

        client = self.client

        # Convert this action to the actual torcs action
        this_action = self._agent_action_to_torcs(action)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = DEFAULT_SPEED
            if client.S.d['speedX'] < target_speed - (client.R.d['steer'] * 50):
                client.R.d['accel'] += 0.01
            else:
                client.R.d['accel'] -= 0.01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1 / (client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2] + client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0] + client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']

        #  Automatic gear change
        action_torcs["gear"] = self._automatic_gear(client.S.d["speedX"])

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        observation = self._get_obs(obs)

        # Reward computation
        # direction-dependent positive reward
        track = np.array(obs['track'])
        sp = np.array(obs['speedX'])
        progress = sp*np.cos(obs['angle'])
        reward = progress

        # Collision detection
        reward_bonus = 0.0 
        if obs['damage'] - obs_pre['damage'] > 0:
            reward_bonus += -1
        reward += reward_bonus

        # Termination judgement
        terminated, truncated = False, False
        if track.min() < 0:                         # Episode is terminated if the car is out of track
            reward = -1
            terminated = True

        if TERMINAL_JUDGE_START < self.time_step:   # Episode terminates if the progress of agent is small
            if progress < TERMINATION_LIMIT_PROGRESS:
                terminated = True

        if np.cos(obs['angle']) < 0:                # Episode is terminated if the agent runs backward
            terminated = True

        if self.time_step > self.max_episode_steps: # Episode truncates
            truncated = True

        done = terminated or truncated

        if done:
            client.R.d['meta'] = True
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return observation, reward, terminated, truncated, {}

    def close(self) -> None:
        os.system('pkill torcs')
        self.client.shutdown()

    def _get_obs(self, raw_obs: Dict[str, np.ndarray]) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        # Read more about the features here: https://arxiv.org/pdf/1304.1672
        focus = np.asarray(raw_obs["focus"], dtype=np.float32) / MAX_FOCUS
        speedX = np.array([raw_obs["speedX"]], dtype=np.float32) / DEFAULT_SPEED
        speedY = np.array([raw_obs["speedY"]], dtype=np.float32) / DEFAULT_SPEED
        speedZ = np.array([raw_obs["speedZ"]], dtype=np.float32) / DEFAULT_SPEED
        opponents = np.asarray(raw_obs["opponents"], dtype=np.float32) / MAX_OPPONENTS
        rpm = np.array([raw_obs["rpm"]], dtype=np.float32)
        track = np.asarray(raw_obs["track"], dtype=np.float32) / MAX_TRACK
        wheel_spin_vel = np.asarray(raw_obs["wheelSpinVel"], dtype=np.float32)

        obs = np.concatenate([
            focus,
            speedX,
            speedY,
            speedZ,
            opponents,
            rpm,
            track,
            wheel_spin_vel,
        ]).astype(np.float32)

        noise = np.random.normal(0, OBS_NOISE_STD, size=obs.shape)
        obs = obs + noise

        return obs

    def _automatic_gear(self, speed: float) -> int:
        if speed > 170:
            return 6
        if speed > 140:
            return 5
        if speed > 110:
            return 4
        if speed > 80:
            return 3
        if speed > 50:
            return 2
        return 1

    def _agent_action_to_torcs(self, action: np.ndarray) -> Dict[str, Union[int, float]]:
        torcs_action = {'steer': float(action[0])}

        if self.throttle is True:       # Throttle action is enabled
            torcs_action.update({'accel': float(action[1])})

        return torcs_action

    def _launch_torcs(self) -> None:
        os.system("pkill torcs")
        time.sleep(0.5)
        cmd = "torcs -nofuel -nodamage -nolaptime &"
        os.system(cmd) 
        time.sleep(0.5)
        os.system("sh autostart.sh")
        time.sleep(0.5)
        print(cmd)