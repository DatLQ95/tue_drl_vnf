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
        node_no = 0
        service_no = 0
        vnf_no = 0
        node_dst_no = 0
        node_list = list(self.docker_lbs.keys())
        len(node_list)
        # logger.info(f"len(node_list): {len(node_list)}")
        # logger.info(f"self.docker_lbs: {self.docker_lbs}")
        for container in self.docker_lbs.values():
            # logger.info(f"container: {container}")
            for name, weights in container.items():
                
                for key in weights.keys():
                    # logger.info(f"weights: {weights}")
                    if("WEIGHT" in str(key)):
                        # logger.info(f"node_no: {key}")
                        # logger.info(f"node_no: {node_no}")
                        # logger.info(f"service_no: {service_no}")
                        # logger.info(f"vnf_no: {vnf_no}")
                        # logger.info(f"node_dst_no: {node_dst_no}")
                        
                        weights[key] = round(weight[node_no][service_no][vnf_no][node_dst_no] * 100)

                        if node_dst_no < len(node_list):
                            node_dst_no = node_dst_no + 1

                        if node_dst_no == len(node_list):
                            service_no = service_no + 1
                            node_dst_no = 0

                        if service_no == 4 : 
                            vnf_no = vnf_no + 1
                            service_no = 0
                            node_dst_no = 0
                            break
                logger.info(f"container: {name}, weights: {weights}")
                service = self.get_service(name)
                self.update_service(service, weights)
            node_no = node_no + 1
            vnf_no = 0
            service_no = 0
            node_dst_no = 0

    def get_service(self, service_name):
        """
        return the id of service
        input: service name
        output: service id
        """
        found = False
        list_services = self.client.services.list()
        for service in list_services:
            if service_name in service.name:
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
            logger.info(f"ingress distributuon  {self.get_ingress_distribution}")
            logger.info(f"trace {trace}")
            node_list = list(self.docker_clients.keys())
            for node in node_list:
                for key,value in self.docker_clients[node].items():
                    logger.info(f"container name {key}")
                    service = self.get_service(key)
                    if value['PORT_NUMBER'] == 8001: 
                        user = float(trace['search_service']) * self.get_ingress_distribution[node]['search']
                        user_no = int(user)
                        value['USER_NO'] = user_no
                    elif value['PORT_NUMBER'] == 8002: 
                        user = float(trace['shop_service']) * self.get_ingress_distribution[node]['shop']
                        user_no = int(user)
                        value['USER_NO'] = user_no
                    elif value['PORT_NUMBER'] == 8003: 
                        user = float(trace['web_service']) * self.get_ingress_distribution[node]['web']
                        user_no = int(user)
                        value['USER_NO'] = user_no
                    elif value['PORT_NUMBER'] == 8004: 
                        user = float(trace['media_service']) * self.get_ingress_distribution[node]['media']
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
