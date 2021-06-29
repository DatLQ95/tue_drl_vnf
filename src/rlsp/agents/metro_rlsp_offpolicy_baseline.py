from rlsp.agents.agent_helper import AgentHelper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DDPG
from stable_baselines3.ddpg import MlpPolicy
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
from torch import nn
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, StopTrainingOnRewardThreshold, EveryNTimesteps, StopTrainingOnMaxEpisodes, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.td3.policies import TD3Policy, CnnPolicy
from rlsp.agents.rlsp_offPolicy_baseline import OffPolicy_BaseLine
# from keras.layers import Concatenate, Dense, Flatten, Input
# from keras.models import Model
# from keras.optimizers import Adam
# from keras.callbacks import TensorBoard, LambdaCallback
# from rl.agents import DDPGAgent
# from rl.memory import SequentialMemory
# from rlsp.envs.action_norm_processor import ActionScheduleProcessor
# from rl.random import GaussianWhiteNoiseProcess
from rlsp.agents.rlsp_agent import RLSPAgent
from rlsp.utils.util_functions import create_simulator
# from rlsp.envs.gym_env import GymEnv
import copy
import csv
import logging
import os
from tqdm.auto import tqdm
import csv
import sys

logger = logging.getLogger(__name__)

class Metro_OffPolicy_BaseLine(OffPolicy_BaseLine):
    """
    RLSP DDPG Agent
    This class creates a DDPG agent with params for RLSP
    """
    def __init__(self, agent_helper):
        super().__init__(agent_helper)
        pass

    def test(self, env, episodes, verbose, episode_steps, callbacks):
        """Mask the agent fit function"""
        logger.info(f"episodes: {episodes}, episode_steps: {episode_steps}")
        if self.agent_helper.train:
            # Create a fresh simulator with test argument
            logger.info("Create new Environment!")
            self.agent_helper.env.simulator = create_simulator(self.agent_helper)
        obs = self.env.reset()
        self.setup_writer()
        self.setup_run_writer()
        episode = 1
        step = 0
        episode_reward = 0.0
        done = False
        # action, _states = self.model.predict(obs)
        # obs, reward, dones, info = self.env.step(action)
        # logger.info(f"info: {info}")

        # Test for 1 episode
        while not done:
            action, _states = self.model.predict(obs)
            obs, reward, dones, info = self.env.step(action)
            episode_reward += reward
            self.write_run_reward(step, reward)
            if info['episode'] >= self.agent_helper.episode_steps:
                done = True
                self.write_reward(episode, episode_reward)
                # episode += 1
            sys.stdout.write(
                "\rTesting:" +
                f"Current Simulator Time: {info['sim_time']}. Testing duration: {self.agent_helper.episode_steps * self.agent_helper.n_steps_per_episode}")
            sys.stdout.flush()
            step += 1
        print("")
        pass