from os import name
from typing import List
import docker
import logging
from rlsp.utils.util_functions import get_trace
from rlsp.utils.util_functions import get_docker_services
import random
logger = logging.getLogger(__name__)

docker_client_services_path = 'res/containers/client_containers.yaml'
docker_lb_container_path = 'res/containers/load_balancer_containers.yaml'
ingress_distribution_file_path = 'res/service_functions/metro_network_ingress_distribution.yaml'

class DockerHelper():
    def __init__(self, user_number_trace_file, ingress_distribution_file_path, docker_client_services_path, docker_lb_container_path):
        """
        Create docker helper with the user_trace from all clients in all node
        user_number_trace: list of dict 
        """
        self.client = docker.from_env()
        self.trace = get_trace(user_number_trace_file)
        self.get_ingress_distribution = get_docker_services(ingress_distribution_file_path)
        self.docker_clients = get_docker_services(docker_client_services_path)
        self.docker_lbs = get_docker_services(docker_lb_container_path)

    def test_container(self):
        print("test")
        mode = docker.types.ServiceMode("global")
        port = docker.types.EndpointSpec()
        a = self.client.services.create("luongquocdat01091995/network_services:search-server", mode=mode, )
        print(a)

    def set_user_number(self, time_step):
        """
        Update the user number to each client containers.
        Input: time step 
        """
        time_trace = self.get_trace_time(time_step)
        self.update_clients(time_trace)
    
    def set_weight(self, weight):
        """
        Update the weight to each load balancer containers.
        Input: weight array:
        [node, sfc, sf, node_dst ]
        [(node_0, search_service, search_lb, node_0), (node_0, search_service, search_lb, node_1), ...]
        Output: update the load balancer service.
        """
        logger.info(f"weights: {weight}")
        print(weight)
        cnt = 0
        for container in self.docker_lbs.values():
            # print(container)
            for name, weights in container.items():
                for key in weights.keys():
                    if("WEIGHT" in str(key)):
                        print(str(key))
                        weights[key] = round(weight[cnt] * 100)
                        cnt = cnt + 1
                
                print(weights)
                service = self.get_service(name)
                self.update_service(service, weights)

    def get_service(self, service_name):
        """
        return the id of service
        input: service name
        output: service id
        """
        found = False
        list_services = self.client.services.list()
        for service in list_services:
            if service.name == service_name:
                service_id = service.id
                found = True
        if found == False:
            return 0
        else:
            return service_id
        
    def update_service(self, service_id, env_dict):
        """
        update the env list of a service:
        input: env_list: dict(): {"env_1": 14, "env_2": 16}
                service_id: id of service to update
        """
        env_list = list()
        for env_key,value in env_dict.items():
            env_list.append(f"{env_key}={value}")
        service = self.client.services.get(service_id)
        service.update(env=env_list)
        pass
        
    def update_clients(self, traces):
        """
        input: list of trace dict:
        {time: 1, search_client: 200, shop_client: 300, web_client: 500, media_client:80}  
        """
        for trace in traces: 
            # TODO: check if success after update by using raise to catch error:
            # node4 = edge4
            # node3 = edge3
            # node2 = edge2
            # print(self.docker_services[trace['node']])
            node_list = list(self.docker_clients.keys())
            for node in node_list:
                for key,value in self.docker_clients[node].items():
                    service = self.get_service(key)
                    if value['PORT_NUMBER'] == 8001: 
                        user = float(trace['search_service']) * self.get_ingress_distribution[node]['search_service']
                        user_no = int(user)
                        value['USER_NO'] = user_no
                    elif value['PORT_NUMBER'] == 8002: 
                        user = float(trace['shop_service']) * self.get_ingress_distribution[node]['shop_service']
                        user_no = int(user)
                        value['USER_NO'] = user_no
                    elif value['PORT_NUMBER'] == 8003: 
                        user = float(trace['web_service']) * self.get_ingress_distribution[node]['web_service']
                        user_no = int(user)
                        value['USER_NO'] = user_no
                    elif value['PORT_NUMBER'] == 8004: 
                        user = float(trace['media_service']) * self.get_ingress_distribution[node]['media_service']
                        user_no = int(user)
                        value['USER_NO'] = user_no
                    else :
                        print("Fuckin wrong here!")
                    self.update_service(service,value)

    def get_trace_time(self, time_step):
        """
        input: time stepdate.
        output: list of trace in that time step
        """
        return_trace = list()
        for i in self.trace:
            if int(i['time']) == time_step:
                return_trace.append(i)
        return return_trace





# docker_helper = DockerHelper(user_number_trace_file='res/traces/trace_metro_network_users.csv', ingress_distribution_file_path=ingress_distribution_file_path, docker_client_services_path=docker_client_services_path, docker_lb_container_path=docker_lb_container_path)
# # weight = [random.random() for _ in range(72)]
# docker_helper.set_user_number(3)

# service_id = docker_helper.get_service("web_client_4")
# env_list = {"USER_NO":90}
# docker_helper.update_service(service_id, env_list)
