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

class DDPG_BaseLine(OffPolicy_BaseLine):
    """
    RLSP DDPG Agent
    This class creates a DDPG agent with params for RLSP
    """
    def __init__(self, agent_helper):
        super().__init__(agent_helper)
        pass

    def create(self, n_envs=1):
        """Create the agent"""
        self.env = self.agent_helper.env
        log_dir = self.agent_helper.config_dir
        os.makedirs(log_dir, exist_ok=True)
        self.env = Monitor(self.env, log_dir)
        #TODO: 
        # Create DDPG policy and define its hyper parameter here! even the action space and observation space.
        # add policy
        policy_name = self.agent_helper.config['policy']
        self.policy = eval(policy_name)
        # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        n_actions = int(self.agent_helper.env.action_space.shape[0])
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=self.agent_helper.config['rand_sigma'] * np.ones(n_actions))
        
        #FIXME: test:
        # self.model = DDPG("MlpPolicy", self.env, action_noise=action_noise, verbose=1, tensorboard_log=self.agent_helper.graph_path)

        # TODO: fix the obvervation space and action space later. Test if the obervation space input is correct? Output action space is correct?
        # activ_function_name = self.agent_helper.config['nn_activ']
        # activ_function = eval(activ_function_name)

        # policy_kwargs = dict(activation_fn=activ_function,
        #              net_arch=[dict(pi=[32, 32], qf=[32, 32])])
        policy_kwargs = dict(net_arch=self.agent_helper.config['layers'])
        self.model = DDPG(
            self.policy,
            self.env,
            learning_rate=self.agent_helper.config['learning_rate'],
            buffer_size = self.agent_helper.config['buffer_size'],
            batch_size=self.agent_helper.config['batch_size'],
            tau=self.agent_helper.config['tau'],
            gamma=self.agent_helper.config['gamma'],
            gradient_steps=self.agent_helper.config['gradient_steps'],
            action_noise=action_noise,
            optimize_memory_usage=self.agent_helper.config['optimize_memory_usage'],
            create_eval_env=self.agent_helper.config['create_eval_env'],
            policy_kwargs=policy_kwargs,
            verbose=self.agent_helper.config['verbose'],
            # learning_starts=self.agent_helper.config['learning_starts'],
            tensorboard_log=self.agent_helper.graph_path,
            seed=self.agent_helper.seed
        )
        pass

    def load_weights(self, weights_file):
        """ Load the model from a zip archive """
        self.model = DDPG.load(weights_file)
        pass