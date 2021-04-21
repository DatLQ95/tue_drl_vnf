from rlsp.agents.agent_helper import AgentHelper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.ddpg import MlpPolicy
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
from torch import nn
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, StopTrainingOnRewardThreshold, EveryNTimesteps, StopTrainingOnMaxEpisodes, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.td3.policies import TD3Policy, CnnPolicy
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

class CollectRewardsCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, env, verbose=1):
        super(CollectRewardsCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

EPISODE_REWARDS = {}
#TODO: fix or add this? 
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record('random_value', value)
        return True

class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

class CustomPolicy(ContinuousCritic):
    """
    Custom policy. Exactly the same as MlpPolicy but with different NN configuration
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        self.agent_helper: AgentHelper = _kwargs['AgentHelper']
        pi_layers = self.agent_helper.agent_config['pi_nn']
        vf_layers = self.agent_helper.agent_config['vf_nn']
        activ_function_name = self.agent_helper.agent_config['nn_activ']
        activ_function = eval(activ_function_name)
        net_arch = [dict(vf=vf_layers, pi=pi_layers)]
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        net_arch=net_arch, act_fun=activ_function, feature_extraction="spr", **_kwargs)



class OffPolicy_BaseLine(RLSPAgent):
    """
    RLSP DDPG Agent
    This class creates a DDPG agent with params for RLSP
    """
    def __init__(self, agent_helper):
        self.agent_helper = agent_helper
        # create model 
        #TODO: add number of env for multiple processing later for faster traing:
        self.create()
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
        self.model = OffPolicyAlgorithm(
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
            learning_starts=self.agent_helper.config['learning_starts'],
            tensorboard_log=self.agent_helper.graph_path,
            seed=self.agent_helper.seed
        )
        pass

    def fit(self, env, episodes, verbose, episode_steps, callbacks, log_interval, agent_id=-1):
        """Mask the agent fit function
        To train the agent
        """
        logger.info("herer")
        # self.model.learn(total_timesteps=100, log_interval=10)
        #FIXME: use the tb logname meaningful!

        #TODO: Write callback funcs here:
        # List of callback:
        # Checkpoint Callback: save the model every 10 episodes.
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                         name_prefix='rl_model')
        # Eval Callback: evaluate every eval_freq, save the best model to best_model_save_path. 
        eval_env = env
        eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)
        # StopTrainingOnRewardThreshold: stop the training on reward threshold, show that this is good enough 
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=70, verbose=1)
        eval_callback_reward_threshold = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)
        # EveryNTimeSteps: to call every n time steps to save the model.
        checkpoint_on_event = CheckpointCallback(save_freq=1, save_path='./logs/')
        event_callback_after_n_steps = EveryNTimesteps(n_steps=500, callback=checkpoint_on_event)

        # StopTrainingOnMaxEpisodes: 
        # Stops training when the model reaches the maximum number of episodes
        callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=5, verbose=1)

        # CallbackList: to call several callback together.
        callbacklist = CallbackList([checkpoint_callback, eval_callback, checkpoint_callback])

        with ProgressBarManager(log_interval) as callback:
            self.model.learn(
                total_timesteps=log_interval,
                callback=callback)
        mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=10)
        self.eval_writer(mean_reward, std_reward)
        pass

    def test(self, env, episodes, verbose, episode_steps, callbacks):
        """Mask the agent fit function"""
        logger.info(f"episodes: {episodes}, episode_steps: {episode_steps}")
        if self.agent_helper.train:
            # Create a fresh simulator with test argument
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
            if info['sim_time'] >= (self.agent_helper.episode_steps * self.agent_helper.n_steps_per_episode):
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

    def save_weights(self, file, overwrite=True):
        weights_file = f"{file}weights"
        dir_path = os.path.dirname(os.path.realpath(weights_file))
        os.makedirs(dir_path, exist_ok=True)

        # After training is done, we save the final weights in the result_base_path.
        logger.info("saving model and weights to %s", weights_file)
        # self.agent.save_weights(weights_file, overwrite)
        self.model.save(weights_file)
        pass

    def load_weights(self, weights_file):
        """ Load the model from a zip archive """
        self.model = OffPolicyAlgorithm.load(weights_file)
        pass

    def setup_writer(self):
        episode_reward_filename = f"{self.agent_helper.config_dir}/episode_reward.csv"
        episode_reward_header = ['episode', 'reward']
        self.episode_reward_stream = open(episode_reward_filename, 'a+', newline='')
        self.episode_reward_writer = csv.writer(self.episode_reward_stream)
        self.episode_reward_writer.writerow(episode_reward_header)
    
    def setup_run_writer(self):
        run_reward_filename = f"{self.agent_helper.config_dir}/run_reward.csv"
        run_reward_header = ['run', 'reward']
        self.run_reward_stream = open(run_reward_filename, 'a+', newline='')
        self.run_reward_writer = csv.writer(self.run_reward_stream)
        self.run_reward_writer.writerow(run_reward_header)

    def write_reward(self, episode, reward):
        self.episode_reward_writer.writerow([episode, reward])

    def write_run_reward(self, step, reward):
        self.run_reward_writer.writerow([step, reward])

    def eval_writer(self, mean_reward, std_reward):
        episode_reward_filename = f"{self.agent_helper.config_dir}evaluate_agent.csv"
        episode_reward_header = ['mean_reward', 'std_reward']
        self.episode_reward_stream = open(episode_reward_filename, 'a+', newline='')
        self.episode_reward_writer = csv.writer(self.episode_reward_stream)
        self.episode_reward_writer.writerow(episode_reward_header)
        self.episode_reward_writer.writerow([mean_reward, std_reward])

    def eval_writer(self, mean_reward, std_reward):
        episode_reward_filename = f"{self.agent_helper.config_dir}evaluate_agent.csv"
        episode_reward_header = ['mean_reward', 'std_reward']
        self.episode_reward_stream = open(episode_reward_filename, 'a+', newline='')
        self.episode_reward_writer = csv.writer(self.episode_reward_stream)
        self.episode_reward_writer.writerow(episode_reward_header)
        self.episode_reward_writer.writerow([mean_reward, std_reward])
