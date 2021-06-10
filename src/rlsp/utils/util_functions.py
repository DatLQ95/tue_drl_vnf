"""
RLSP utility functions module
"""


def create_simulator(agent_helper):
    """Create a simulator object"""
    from siminterface.simulator import Simulator

    return Simulator(agent_helper.network_path, agent_helper.service_path, agent_helper.sim_config_path,
                     test_mode=agent_helper.test_mode, test_dir=agent_helper.config_dir)

def create_new_env(agent_helper):
    """
    Create new env:
    """
    from rlsp.envs.metro_network_env import MetroNetworkEnv

    return MetroNetworkEnv(agent_config=agent_helper.config,
                   network_file=agent_helper.network_path,
                   service_file=agent_helper.service_path,
                   user_trace_file = agent_helper.user_trace_path,
                   service_requirement_file=agent_helper.service_requirement_path,
                   ingress_distribution_file=agent_helper.ingress_distribution_path,
                   container_client_file=agent_helper.client_containers_path,
                   container_server_file=agent_helper.server_containers_path,
                   container_lb_file=agent_helper.lb_containers_path, test_dir= agent_helper.config_dir)

def get_docker_services(docker_services_path):
    import yaml
    with open(docker_services_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_trace(trace_file):
    """
    input: trace file .csv
    output: list of dict of user: 
    user_trace[0]:
    node0 : search_client: 200
            shop_client: 300
            web_client: 500
            media_client:80
    node1 : search_client: 200
            shop_client: 300
            web_client: 500
            media_client:80  
    user_trace[1]:
    node0 : search_client: 250
            shop_client: 360
            web_client: 580
            media_client:80
    node1 : search_client: 200
            shop_client: 300
            web_client: 500
            media_client:80  
    """
    import csv
    with open(trace_file) as f:
        trace_rows = csv.DictReader(f)
        traces = []
        for row in trace_rows:
            traces.append(dict(row))
    return traces