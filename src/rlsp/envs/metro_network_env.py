# -*- coding: utf-8 -*-
"""
Gym envs representing the coordination-simulation from
REAL NFV https://github.com/RealVNF/coordination-simulation


For help on "Implementing New Environments" see:
https://github.com/openai/gym/blob/master/gym/core.py
https://github.com/rll/rllab/blob/master/docs/user/implement_env.rst

"""
import inspect
import logging
from typing import Tuple
import gym
from gym.utils import seeding
import numpy as np
from rlsp.envs.environment_limits import EnvironmentLimits
from rlsp.utils.constants import SUPPORTED_OBJECTIVES
from spinterface import SimulatorInterface, SimulatorState
from coordsim.reader.reader import read_network, get_sfc, get_sf, network_diameter, get_sfc_requirement
import json
import random
from stable_baselines3.common.env_checker import check_env
from rlsp.envs.docker_helper import DockerHelper
from rlsp.envs.capture_helper import CaptureHelper
from rlsp.envs.action_norm_processor import ActionScheduleProcessor

from rlsp.agents.main import get_config

logger = logging.getLogger(__name__)


class MetroNetworkEnv(gym.Env):
    def __init__(self, agent_config, network_file, service_file, user_trace_file, ingress_distribution_file, service_requirement_file, container_client_file, container_server_file, container_lb_file):
        """
        Pass the configure file
        Test if the container is working fine!?
        """
        self.ingress_distribution_file = ingress_distribution_file
        self.container_client_file = container_client_file
        self.container_server_file = container_server_file
        self.container_lb_file = container_lb_file

        self.network_file = network_file
        self.agent_config = agent_config
        self.user_trace_file = user_trace_file
        self.service_requirement = get_sfc_requirement(service_requirement_file)

        #TODO: read csv file to list of arrays
        # self.trace = 

        # self.np_random = np.random.RandomState()
        # self.seed(seed)

        self.network, _, _ = read_network(self.network_file)
        self.network_diameter = network_diameter(self.network)
        
        self.sfc_list = get_sfc(service_file)
        self.sf_list = get_sf(service_file)

        # # logger.info('service_requirement: ' + str(self.service_requirement))
        self.env_limits = EnvironmentLimits(len(self.network.nodes), self.sfc_list,
                                            len(agent_config['observation_space']))
        # self.min_delay, self.max_delay = self.min_max_delay()
        self.action_space = self.env_limits.action_space
        self.observation_space = self.env_limits.observation_space
        # # logger.info('Observation space: ' + str(self.agent_config['observation_space']))

        # # order of permutation for shuffling state
        # self.permutation = None

        # # related to objective/reward
        self.objective = self.agent_config['objective']
        self.episode_count = -1
        self.dockerHelper = DockerHelper(user_trace_file,
        ingress_distribution_file_path = ingress_distribution_file, docker_client_services_path = container_client_file, docker_lb_container_path = container_lb_file)
        self.captureHelper = CaptureHelper(docker_client_services_path = container_client_file, docker_server_services_path= container_server_file, ingress_distribution_file_path = ingress_distribution_file)
        self.ingress_node = ["node3", "node4"]
        pass


    def reset(self):
        """
        Update the client container with the number of user to starting point.
        Capture the ingress traffic data.
        TODO: what is the action? 
        -------
        observation : the initial observation of the space.
        (Initial reward is assumed to be 0.)
        Observation : 
        [node_number x ]
        """
        # Add 1 to the number of episode
        self.episode_count += 1
        # Create step count
        self.step_count = 0
        self.dockerHelper.set_user_number(self.step_count)
        latency, dropped_conn, success_conn, ingress_traffic = self.captureHelper.capture_data(self.ingress_node)
        capture_traffic = self.normalize_ingress_traffic(ingress_traffic)
        #FIXME: replace this with the real comand
        capture_traffic = [random.random() for _ in range(12)]
        logger.info(f"capture_traffic: {capture_traffic}")
        observation = np.array(capture_traffic).astype(np.float32)
        return observation

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray):
        """
        Get the action from Agent RL
        Update the weight in each load balancer accordingly.
        Update the number user in each client node according to the time (step count).
        Capture the ingress traffic using NetData.
        Capture the latency using Prometheous.
        Calculate reward.
        Normalize observation space.


        Returns -> Tuple[object, float, bool, dict]:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether episode has ended, in which case further step calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        done = False
        #FIXME: testing only NEDD TO BE TESTED here!
        action = [random.random() for _ in range(72)]
        logger.info(f"action : {action}")
        action_processor = ActionScheduleProcessor(self.env_limits.MAX_NODE_COUNT, self.env_limits.MAX_SF_CHAIN_COUNT,
                                                   self.env_limits.MAX_SERVICE_FUNCTION_COUNT)
        action = action_processor.process_action(action)
        scheduling = np.reshape(action, self.env_limits.scheduling_shape)
        logger.info(f"scheduling : {scheduling}")
        self.step_count +=1
        self.dockerHelper.set_weight(scheduling)
        self.dockerHelper.set_user_number(self.step_count)
        #TODO: packet_loss
        # packet_loss = self.captureHelper.get_packet_loss()
        
        latency, dropped_conn, success_conn, ingress_traffic = self.captureHelper.capture_data(self.ingress_node)
        # reward = self.calculate_reward(latency)

        # Normalize the ingress traffic
        capture_traffic = self.normalize_ingress_traffic(ingress_traffic)
        #FIXME: replace this with the real comand
        capture_traffic = [random.random() for _ in range(12)]
        reward = random.random()
        observation = np.array(capture_traffic).astype(np.float32)
        if self.step_count == self.agent_config['episode_steps']:
            done = True
            self.step_count = 0
        info = {}
        return observation, reward, done, info

    def render(self, mode='cli'):
        """Renders the envs.
        Implementation required by Gym.
        """
        assert mode in ['human']

    def get_flow_reward_sfcs(self, dropped_conn, success_conn):
        """Calculate and return both success ratio and flow reward"""
        cur_succ_flow = 0
        cur_drop_flow = 0
        # logger.info(f"simulator_state.sfcs.keys() : {simulator_state.sfcs.keys()}")
        # calculate ratio of successful flows in the last run
        service_list = list(dropped_conn.keys())
        for service in service_list:
            # logger.info(f"sfc : {sfc}")
            # logger.info(f"simulator_state.network_stats['run_successful_flows_sfcs'] : {simulator_state.network_stats['run_successful_flows_sfcs'][sfc]}")
            # logger.info(f"self.service_requirement_file[sfc]['priority'] : {self.service_requirement[sfc]['priority']}")
            cur_succ_flow += success_conn[service] * self.service_requirement[service]['priority']
            cur_drop_flow += dropped_conn[service] * self.service_requirement[service]['priority']
        logger.info(f"cur_succ_flow : {cur_succ_flow}")
        logger.info(f"cur_drop_flow : {cur_drop_flow}")

        succ_ratio = 0
        flow_reward = 0
        if cur_succ_flow + cur_drop_flow > 0:
            succ_ratio = cur_succ_flow / (cur_succ_flow + cur_drop_flow)
            # use this for flow reward instead of succ ratio to use full [-1, 1] range rather than just [0,1]
            flow_reward = (cur_succ_flow - cur_drop_flow) / (cur_succ_flow + cur_drop_flow)

        logger.info(f"flow_reward : {flow_reward}")
        return flow_reward

    def get_delay_reward_sfcs(self, delay_sfcs):
        """
        Calculate the reward from latency of each sfcs.
        input: dict: {latency_search: 24, latency_shop: 28, latency_web: 30, latency_media: 21}
        """
        delay_reward_sfc = list()
        total_priority = 0
        
        for sfc, sfc_delay in delay_sfcs.items():
            if sfc_delay == 0: 
                continue
            else:
                reward_delay = (2 * self.service_requirement[sfc]['delay_requirement'] /sfc_delay) - 1
                logger.info(f"reward_delay: {reward_delay} of sfc: {sfc}")
                reward_delay = min(reward_delay, 1)
            # logger.info(f"reward_delay: {reward_delay} of sfc: {sfc}")
            reward_delay *= self.service_requirement[sfc]['priority']
            delay_reward_sfc.append(reward_delay)
            total_priority += self.service_requirement[sfc]['priority']
        # logger.info(f"reward_delay: {reward_delay} of sfc: {sfc}")
        # logger.info(f"total_priority: {total_priority}")
        # logger.info(f"delay_reward_sfc: {delay_reward_sfc}")
        
        delay_reward = sum([delay/total_priority for delay in delay_reward_sfc])
        logger.info(f"reward_delay: {delay_reward}")
        return delay_reward

    def calculate_reward(self, latency, dropped_conn, success_conn):
        """
        Calculate reward per step based on the chosen objective.
        input: latency: dict: {latency_search: 24, latency_shop: 28, latency_web: 30, latency_media: 21}
        :return -> float: The agent's reward
        """
        #TODO: test the calculate reward func again!
        
        # Calculate the reward:
        delay_reward = self.get_delay_reward_sfcs(latency)
        conn_reward = self.get_flow_reward_sfcs(dropped_conn, success_conn)
 
        if self.objective == 'network_latency_priority':
            # weight all objectives as configured before summing them
            delay_reward = delay_reward
            conn_reward = 0
            pass
        elif self.objective == 'sfc_with_priority':
            conn_reward *= self.agent_config['flow_weight']
            delay_reward *= self.agent_config['delay_weight']
        else:
            raise ValueError(f"Unexpected objective {self.objective}. Must be in {SUPPORTED_OBJECTIVES}.")
        
        # calculate and return the sum, ie, total reward
        total_reward = conn_reward + delay_reward
        assert -2 <= total_reward <= 2, f"Unexpected total reward: {total_reward}."
        logger.info(f"total_reward: {total_reward}")
        return total_reward

    def min_max_delay(self):
        """Return the min and max e2e-delay for the current network topology and SFC. Independent of capacities."""
        
        # vnf_delays = sum([sf['processing_delay_mean'] for sf in self.sf_list['sfc_1']])
        # logger.info(f"self.sf_list: {self.sf_list}")
        # # min delay = sum of VNF delays (corresponds to all VNFs at ingress)
        # min_delay = vnf_delays
        # # max delay = VNF delays + num_vnfs * network diameter (corresponds to max distance between all VNFs)
        # max_delay = vnf_delays + len(self.sf_list) * self.network_diameter
        
        #FIXME: change this!!!

        min_delay = self.sf_list['a1']['processing_delay_mean'] + self.sf_list['b1']['processing_delay_mean']
        max_delay = min_delay + 2 * self.network_diameter
        logger.info(f"min_delay: {min_delay}, max_delay: {max_delay}, diameter: {self.network_diameter}")
        return min_delay, max_delay

    def apply(action):
        """
        Transform the action to docker helper
        run containers / network for delta t (sec)
        Capture the latency and BW ingress and also BW come into all node!?
        TODO: If possible, packet loss!?
        Return the obersvation state
        """
        logger.info(f"Applying action to network: {action}")
        pass

    def normalize_ingress_traffic(self, ingress_traffic):
        """
        normalize the ingress traffic to fit with observation space.
        divided by 10 Gb/s.
        Ex: 1 Gb/s -> 0.1
        ingress traffic: [search_client_node0, shop_client_node0, web_Client_node0, media_client_node0, search_client_node1, shop_client_node1, web_client_node1, media_client_node1]
        ouput: [search_client_node0, shop_client_node0, web_Client_node0, media_client_node0, search_client_node1, shop_client_node1, web_client_node1, media_client_node1]
        """
        #TODO: convert dict to list !
        logger.info(f"normalize_ingress_traffic: {ingress_traffic}")
        pass

AGENT_CONFIG = 'res/config/agent/sample_agent_100_DDPG_Baseline.yaml'
NETWORK_TRIANGLE =  'res/networks/tue_network_triangle.graphml'
USER_TRACE = 'res/traces/trace_metro_network_users.csv'
SERVICE_TRIANGLE = 'res/service_functions/metro_network_services.yaml'
SERVICE_REQUIREMENT_TRIANGLE = 'res/service_functions/metro_network_service_requirement.yaml'
docker_client_services_path = 'res/containers/client_containers.yaml'
docker_server_services_path = 'res/containers/server_containers.yaml'
ingress_distribution_file_path = 'res/service_functions/metro_network_ingress_distribution.yaml'
docker_lb_container_path = 'res/containers/load_balancer_containers.yaml'

config = get_config(AGENT_CONFIG)
env = MetroNetworkEnv(agent_config=config, network_file=NETWORK_TRIANGLE, service_file=SERVICE_TRIANGLE, user_trace_file = USER_TRACE, service_requirement_file = SERVICE_REQUIREMENT_TRIANGLE, ingress_distribution_file=ingress_distribution_file_path, container_client_file=docker_client_services_path, container_server_file=docker_server_services_path, container_lb_file=docker_lb_container_path)

# latency = {'search': 457.0, 'shop': 464.0, 'web': 476.0, 'media': 433.0}
# dropped_conn = {'search': 0.0, 'shop': 0.0, 'web': 0.0, 'media': 0.0}
# succ_conn = {'search': 84.0, 'shop': 96.0, 'web': 99.0, 'media': 40.0}
# env.calculate_reward(latency= latency, dropped_conn=dropped_conn, success_conn= succ_conn)
        