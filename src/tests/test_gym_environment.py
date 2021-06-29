# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
import gym
# noinspection PyUnresolvedReferences
import rlsp
from rlsp.agents.main import get_config
from siminterface.simulator import Simulator
from simulator.omnet_simulator import OmnetSimulator
import logging
from rlsp.envs.gym_env import GymEnv
from rlsp.envs.metro_network_env import MetroNetworkEnv
import logging
from stable_baselines3.common.env_checker import check_env

logger = logging.getLogger(__name__)
logging.basicConfig()
logging.root.setLevel(logging.DEBUG)


AGENT_CONFIG = 'res/config/agent/sample_agent_100_DDPG_Baseline.yaml'
NETWORK = 'res/networks/tue_network.graphml'
SERVICE = 'res/service_functions/tue_abc.yaml'
SIM_CONFIG = 'res/config/simulator/trace_config_100_sim_duration.yaml'
SERVICE_REQUIREMENT = 'res/service_functions/sfc_requirement.yaml'


NETWORK_TRIANGLE =  'res/networks/tue_network_triangle.graphml'
USER_TRACE = 'res/traces/trace_metro_network_users.csv'
SERVICE_TRIANGLE = 'res/service_functions/metro_network_services.yaml'
SERVICE_REQUIREMENT_TRIANGLE = 'res/service_functions/metro_network_service_requirement.yaml'

# AGENT_CONFIG = 'res/config/agent/sample_agent_100_DDPG_Baseline.yaml'
# NETWORK_TRIANGLE =  'res/networks/tue_network_triangle.graphml'
# USER_TRACE = 'res/traces/trace_metro_network_users.csv'
# SERVICE_TRIANGLE = 'res/service_functions/metro_network_services.yaml'
# SERVICE_REQUIREMENT_TRIANGLE = 'res/service_functions/metro_network_service_requirement.yaml'
docker_client_services_path = 'res/containers/client_containers.yaml'
docker_server_services_path = 'res/containers/server_containers.yaml'
ingress_distribution_file_path = 'res/service_functions/metro_network_ingress_distribution.yaml'
docker_lb_container_path = 'res/containers/load_balancer_containers.yaml'

agent_config = 'res/config/agent/sample_agent_100_DDPG_Baseline_one_service_flow_objective_node_load_obs.yaml'
network = 'res/networks/tue_network_triangle_15_cap_pop1_pop2_ingress_diff_cap_10_5.graphml'
service = 'res/service_functions/tue_abc_one_service_10.yaml'
sim_config = 'res/config/simulator/trace_config_100_sim_duration_pop1_pop2.yaml'
service_requirement = 'res/service_functions/sfc_requirement_one_service.yaml'

class TestGymEnvironment(TestCase):
    def test_metro_network_env(self):
        # config = get_config(AGENT_CONFIG)
        # # simulator = Simulator(NETWORK, SERVICE, USER_CONFIG)
        # env = gym.make('metro_network-env-v1', agent_config=config, network_file=NETWORK, service_file=SERVICE, user_trace_file = USER_TRACE)
        # np.random.seed(123)
        # env.seed(123)
        # obs = env.reset()
        # logger.info(f"obs {obs}")
        # action = env.action_space.sample()
        # env.step(action)

        config = get_config(AGENT_CONFIG)
        env = MetroNetworkEnv(agent_config=config, network_file=NETWORK_TRIANGLE, service_file=SERVICE_TRIANGLE, user_trace_file = USER_TRACE, service_requirement_file = SERVICE_REQUIREMENT_TRIANGLE, ingress_distribution_file=ingress_distribution_file_path, container_client_file=docker_client_services_path, container_server_file=docker_server_services_path, container_lb_file=docker_lb_container_path)
        check_env(env)

    def test_gym_registration(self):
        config = get_config(AGENT_CONFIG)
        simulator = Simulator(NETWORK, SERVICE, SIM_CONFIG)
        env = GymEnv(agent_config=config, simulator=simulator, network_file=NETWORK,
                       service_file=SERVICE)
        np.random.seed(123)
        env.seed(123)

    def test_step(self):
        config = get_config(agent_config)
        simulator = Simulator(network, service, sim_config)
        env = gym.make('rlsp-env-v1', agent_config=config, simulator=simulator, service_requirement_file=service_requirement, network_file=network, service_file=service)
        np.random.seed(123)
        env.seed(123)
        obs = env.reset()
        logger.info(f"obs {obs}")
        action = env.action_space.sample()
#         action = [0.1528252,  0.19189513, 0.13835746, 0.5151016,  0.8268603,  0.11794249,
#  0.07534754, 0.8041661,  0.6766331,  0.83818173, 0.74750483, 0.2683317,
#  0.1885326,  0.17871094, 0.48484454, 0.80052173, 0.5470235,  0.939061,
#  0.7532571,  0.95109504, 0.7161905,  0.437634,   0.8504298,  0.56406283,
#  0.44520622, 0.9043397,  0.7369696  ,0.06333494, 0.85315746 ,0.35849094,
#  0.7802861,  0.42221898, 0.00383798 ,0.5678534 , 0.24563274 ,0.7632712 ,
#  0.8775916,  0.5324118 , 0.40174866 ,0.5244014 , 0.53869325 ,0.62608093,
#  0.55334944, 0.65927726 ,0.3466312  ,0.81105894, 0.5835827  ,0.4352289,
#  0.01409177, 0.68956935, 0.2725983  ,0.83922625, 0.12063584 ,0.85230696,
#  0.24010704, 0.7814054 , 0.6302035  ,0.7396873  ,0.3951772  ,0.42432162,
#  0.8822647 , 0.5477109 , 0.6531822  ,0.8758906  ,0.12640443, 0.31692567,
#  0.54074305, 0.7312232 , 0.40281135 ,0.46530938 ,0.29194754 ,0.1864658,
#  0.906782  , 0.73796546, 0.9524628  ,0.9118524  ,0.8770285  ,0.57136434,
#  0.6312382 , 0.5789188 , 0.14968178 ,0.41891995 ,0.09121216 ,0.01003964,
#  0.6886566 , 0.13875785, 0.99779034 ,0.36943817 ,0.5710171  ,0.41438,
#  0.81586087, 0.14191775, 0.2810072  ,0.8571065  ,0.83860654 ,0.2865804,
#  0.17616676, 0.11686828, 0.66831094 ,0.27806306 ,0.24364603 ,0.3952267,
#  0.1256759 , 0.5283989 , 0.7713088  ,0.01707859 ,0.86026347 ,0.96787435,
#  0.76898026, 0.24870834, 0.02020283 ,0.26120338 ,0.11047094 ,0.0451458,
#  0.7435941 , 0.14921127, 0.29213443 ,0.22803156 ,0.34240273 ,0.7448049,
#  0.34957966, 0.6426194 , 0.8075939  ,0.00796008 ,0.961782   ,0.8310099,
#  0.9726601 , 0.3459133 ]

        action = [
                1, 0, 0, 
                1, 0, 0,
                0, 1, 0, 
                0, 1, 0,
                0, 0, 1, 
                0, 0, 1
                ]

        logger.info(f"action is : {action}")
        env.step(action)

    def test_min_max_delay(self):
        config = get_config(AGENT_CONFIG)
        simulator = Simulator(NETWORK, SERVICE, SIM_CONFIG)
        env = gym.make('rlsp-env-v1', agent_config=config, simulator=simulator, network_file=NETWORK,
                       service_file=SERVICE)
        self.assertEqual(env.min_delay, 15)
        self.assertEqual(env.max_delay, 45)

if __name__ == "__main__":
    testGym = TestGymEnvironment()
    testGym.test_step()
