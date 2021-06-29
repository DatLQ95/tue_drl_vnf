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
from coordsim.reader.reader import read_network, get_sfc, get_sf, network_diameter, get_sfc_requirement
import json
import random
from stable_baselines3.common.env_checker import check_env
from rlsp.envs.docker_helper import DockerHelper
from rlsp.envs.capture_helper import CaptureHelper
from rlsp.envs.action_norm_processor import ActionScheduleProcessor
from rlsp.utils.util_functions import get_docker_services
from rlsp.envs.writer import ResultWriter
# from rlsp.agents.main import get_config
import time

logger = logging.getLogger(__name__)

class MetroNetworkEnv(gym.Env):
    def __init__(self, agent_config, network_file, service_file, user_trace_file, ingress_distribution_file, service_requirement_file, container_client_file, container_server_file, container_lb_file, log_metrics_dir):
        """
        Pass the configure file
        Test if the container is working fine!?
        """
        self.log_metrics_dir = log_metrics_dir
        self.ingress_distribution_file = ingress_distribution_file
        self.container_client_file = container_client_file
        self.container_server_file = container_server_file
        self.container_lb_file = container_lb_file

        self.ingress_distribution = get_docker_services(self.ingress_distribution_file)

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
        ingress_distribution_file_path = ingress_distribution_file, docker_client_services_path = container_client_file, docker_lb_container_path = container_lb_file, service_list = list(self.sfc_list.keys()))
        self.captureHelper = CaptureHelper(docker_client_services_path = container_client_file, docker_server_services_path= container_server_file, ingress_distribution_file_path = ingress_distribution_file, docker_lb_container_path=container_lb_file, service_list = list(self.sfc_list.keys()))
        self.ingress_node = ["node3", "node4"]
        self.info = dict()

    def reward_func_repr(self):
        """returns a string describing the reward function"""
        return inspect.getsource(self.calculate_reward)

    def get_ingress_nodes(self):
        """
        get the ingress node list
        """
        ingress_distribution = get_docker_services(self.ingress_distribution_file)
        logger.info(ingress_distribution)
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
        logger.info("-------------------------------RESET------------------------")
        # Add 1 to the number of episode
        self.episode_count += 1
        log_metrics_dir_episode = self.log_metrics_dir + "/" + str(self.episode_count)
        self.writer = ResultWriter(log_metrics_dir_episode)
        # Create step count
        self.step_count = 0

        logger.info("Setting user number")
        self.dockerHelper.set_user_number(self.step_count)

        logger.info("Checking client container is working")
        time_up_already = self.check_client_container_working(self.step_count)
        latency, dropped_conn, success_conn, ingress_traffic = self.captureHelper.capture_data(self.ingress_node, time_up_already)

        self.writer.record_capture_data(self.episode_count, self.step_count, latency, dropped_conn, success_conn, ingress_traffic)

        capture_traffic = self.normalize_ingress_traffic(ingress_traffic)
        #FIXME: replace this with the real comand
        # capture_traffic = [random.random() for _ in range(12)]
        # logger.info(f"capture_traffic: {capture_traffic}")
        observation = np.array(capture_traffic).astype(np.float32)
        self.agent_start_time = time.time()
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
        self.step_count +=1
        start_time = time.time()
        logger.info(f"ACTION at {self.step_count}")
        logger.info(f"action : {action}")
        action_processor = ActionScheduleProcessor(self.env_limits.MAX_NODE_COUNT, self.env_limits.MAX_SF_CHAIN_COUNT,
                                                   self.env_limits.MAX_SERVICE_FUNCTION_COUNT)
        action = action_processor.process_action(action)
        scheduling = np.reshape(action, self.env_limits.scheduling_shape)

        logger.info(f"set action : {scheduling}")

        self.writer.record_action(self.episode_count, self.step_count, scheduling)

        self.dockerHelper.set_weight(scheduling)
        self.dockerHelper.set_user_number(self.step_count)

        logger.info("Checking if all actions has been updated")
        latency, dropped_conn, success_conn, ingress_traffic = self.capture_data(scheduling=scheduling, step_count=self.step_count)
        # self.check_all_container_working(scheduling=scheduling, step_count=self.step_count)
        # logger.info("Capturing")
        # latency, dropped_conn, success_conn, ingress_traffic = self.captureHelper.capture_data(self.ingress_node)

        # Normalize the ingress traffic
        capture_traffic = self.normalize_ingress_traffic(ingress_traffic)
        #FIXME: replace this with the real comand
        # capture_traffic = [random.random() for _ in range(12)]
        logger.info(f"latency: {latency}")
        logger.info(f"dropped_conn: {dropped_conn}")
        logger.info(f"success_conn: {success_conn}")
        self.writer.record_capture_data(self.episode_count, self.step_count, latency, dropped_conn, success_conn, capture_traffic)
        reward = self.calculate_reward(latency=latency, dropped_conn= dropped_conn, success_conn= success_conn)
        observation = np.array(capture_traffic).astype(np.float32)
        if self.step_count == self.agent_config['episode_steps']:
            logger.info("END---------------------------------------------------------------")
            done = True
            self.writer.close_stream()
        self.info['step'] = self.step_count
        logger.info(f"observation: {observation}")
        logger.info(f"reward: {reward}")
        logger.info(f"done: {done}")
        stop_time = time.time()
        runtime = stop_time - start_time
        agent_time = start_time - self.agent_start_time
        self.writer.write_runtime(runtime, agent_time)
        self.agent_start_time = time.time()
        return observation, reward, done, self.info

    def capture_data(self, scheduling, step_count):
        """
        Make sure the containers are working good!
        Check if the data capture is good enough for next processing!
        """
        time_up_already = self.check_all_container_working(scheduling=scheduling, step_count=self.step_count)
        logger.info("Capturing")
        latency, dropped_conn, success_conn, ingress_traffic = self.captureHelper.capture_data(self.ingress_node, time_up_already)
        return latency, dropped_conn, success_conn, ingress_traffic

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
        logger.info(delay_sfcs)
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
        
        elif self.objective == 'sfc_with_flow_only':
            conn_reward = conn_reward
            delay_reward = 0

        else:
            raise ValueError(f"Unexpected objective {self.objective}. Must be in {SUPPORTED_OBJECTIVES}.")
        
        # calculate and return the sum, ie, total reward
        total_reward = conn_reward + delay_reward
        assert -2 <= total_reward <= 2, f"Unexpected total reward: {total_reward}."
        logger.info(f"total_reward: {total_reward}")
        self.writer.record_reward(self.episode_count, self.step_count, total_reward, conn_reward, delay_reward)
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
        Normalize based on max -> 500 request/sec!
        Ex: 300 -> 0.6
        ingress traffic: [search_client_node0, shop_client_node0, web_Client_node0, media_client_node0, search_client_node1, shop_client_node1, web_client_node1, media_client_node1]
        ouput: [search_client_node0, shop_client_node0, web_Client_node0, media_client_node0, search_client_node1, shop_client_node1, web_client_node1, media_client_node1]
        """
        #TODO: convert dict to list !
        logger.info(f"ingress traffic: {ingress_traffic}")
        ingress_traffic_list = list()
        for node, cont in ingress_traffic.items():
            for traffic_value in cont.values():
                ingress_traffic_list.append(float(traffic_value / 250))
        logger.info(f"normalize_ingress_traffic: {ingress_traffic_list}")
        return ingress_traffic_list

    def check_all_container_working(self, step_count, scheduling):
        time.sleep(30)
        time_up_already = 0
        time_counter = 0 
        working = False
        while(not working):
            # Make it work here!
            lb_working, lb_container_list = self.captureHelper.check_lb_containers()
            logger.info(f"load balancer: {lb_working} with list: {lb_container_list}")
            time_up_already, client_working, client_container_list = self.captureHelper.check_client_containers()
            logger.info(f"client container: {client_working} with list: {client_container_list}")
            container_list = lb_container_list + client_container_list
            working = lb_working and client_working
            logger.info(f"time_counter: {time_counter}")
            if time_counter == 1:
                # Update it here!
                logger.info("Reseting the error containers")
                self.dockerHelper.reset_containers(container_list)
                logger.info("Waiting the containers to be updated")
            if time_counter == 45: 
                if(not client_working):
                    logger.info("Remove all the client containers")
                    self.dockerHelper.remove_containers(client_container_list)
                    time.sleep(10)
                    self.dockerHelper.restore_clients(client_container_list, step_count)
                elif (not lb_working):
                    logger.info("Remove all the lb containers")
                    self.dockerHelper.remove_containers(lb_container_list)
                    time.sleep(10)
                    self.dockerHelper.restore_lbs(lb_container_list, scheduling)
            if not working:
                time.sleep(1)
                time_counter +=1
        logger.info(f"time up already: {time_up_already}")
        return time_up_already


    def check_client_container_working(self, step_count):
        time_up_already = 0
        time_counter = 0 
        working = False
        while(not working):
            # Make it work here!
            time_up_already, working, client_container_list = self.captureHelper.check_client_containers()
            logger.info(f"client container: {working} with list: {client_container_list}")
            logger.info(f"time_counter: {time_counter}")
            if time_counter == 30:
                # Update it here!
                logger.info("Reseting the error containers")
                self.dockerHelper.reset_containers(client_container_list)
                logger.info("Waiting the containers to be updated")
            if time_counter == 60: 
                logger.info("Remove all the client containers")
                self.dockerHelper.remove_containers(client_container_list)
                time.sleep(10)
                self.dockerHelper.restore_clients(client_container_list, step_count)
            if not working:
                time.sleep(1)
                time_counter +=1
        return time_up_already
    
    def check_lb_container_working(self, scheduling):
        time.sleep(45)
        lb_working, container_list = self.captureHelper.check_lb_containers()
        while(not lb_working):
            self.dockerHelper.reset_lb_container(container_list, scheduling)
            time.sleep(45)
            lb_working, container_list = self.captureHelper.check_lb_containers()
    


# logs_dir_test = 'results/test_random/1'

# agent_config = 'res/config/agent/sample_agent_100_DDPG_Baseline_one_service_flow_objective.yaml'
# sim_config = 'res/config/simulator/trace_config_100_sim_duration_pop1_pop2.yaml'
# network =  'res/networks/tue_network_triangle_15_cap_pop1_pop2_ingress_diff_cap_7_5.graphml'
# user_trace = 'res/traces/trace_metro_network_search_test_model_users.csv'
# service = 'res/service_functions/metro_network_services.yaml'
# service_requirement = 'res/service_functions/metro_network_service_requirement.yaml'
# client_containers = 'res/containers/client_containers.yaml'
# server_containers = 'res/containers/server_containers.yaml'
# ingress_distribution = 'res/service_functions/metro_network_ingress_distribution.yaml'
# lb_containers = 'res/containers/load_balancer_containers.yaml'

# config = get_config(agent_config)
# env = MetroNetworkEnv(agent_config=config, network_file=network, service_file=service, user_trace_file = user_trace, service_requirement_file = service_requirement, ingress_distribution_file=ingress_distribution, container_client_file=client_containers, container_server_file=server_containers, container_lb_file=lb_containers, log_metrics_dir=logs_dir_test)


# # logger.info(f"obs {obs}")

# # action = [
# # 1, 1, 1, 
# # 1, 1, 1,
# # 0, 1, 0, 
# # 1, 0, 0,
# # 0, 0, 1, 
# # 1, 0, 0
# # ]
# # env.step(action)
# # action = [1 for _ in range(72)]
# logger.info("START HERE")
# obs = env.reset()
# done = False
# while not done:
#     action = [random.random() for _ in range(18)]
#     obs, reward, done, info = env.step(action)
#     logger.info(f"done: {done}")


# logger.info(f"observation_space: {env.observation_space.shape}")
# logger.info(f"action_space: {env.action_space.shape[0]}")
# logger.info(f"self.sfc_list: {env.sfc_list.keys()}, sf_list:{env.sf_list}")
# check_env(env)
# latency = {'search': 457.0, 'shop': 464.0, 'web': 476.0, 'media': 433.0}
# dropped_conn = {'search': 0.0, 'shop': 0.0, 'web': 0.0, 'media': 0.0}
# succ_conn = {'search': 84.0, 'shop': 96.0, 'web': 99.0, 'media': 40.0}
# env.calculate_reward(latency= latency, dropped_conn=dropped_conn, success_conn= succ_conn)
        