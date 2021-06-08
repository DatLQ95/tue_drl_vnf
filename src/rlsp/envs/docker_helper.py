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

    def reset_containers(self, container_list):
        for container in container_list:
            service = self.get_service(container)
            self.force_update_service(service_id=service)

    def reload_containers(self, container_list):
        for container in container_list:
            service = self.get_service(container)
            self.reload_service(service_id=service)

    def reset_lb_container(self, container_list, scheduling):
        """
        reset the only container in the list:
        container_list = ['load_balancer_node_4']
        scheduling: [(node_0, search_service, search_lb, node_0), (node_0, search_service, search_lb, node_1), ...]
        update again only the container in the list
        """
        for container in container_list:
            # self.set_weight_container(container_name=container, weight=scheduling)
            service = self.get_service(container)
            self.force_update_service(service_id=service)

    def reset_client_container(self, container_list, step_count):
        """
        reset the only container in the list:
        container_list = ['search_client_2']
        step_count = 3
        update only the container in the list
        """
        for container in container_list:
            # time_trace = self.get_trace_time(step_count)
            # self.update_client(container, time_trace)
            service = self.get_service(container)
            self.force_update_service(service_id=service)

    def set_user_number(self, time_step):
        """
        Update the user number to each client containers.
        Input: time step 
        """
        time_trace = self.get_trace_time(time_step)
        self.update_clients(time_trace)

    def create_lb(self, container_name, weight):
        node_no = 0
        service_no = 0
        vnf_no = 0
        node_dst_no = 0
        node_list = list(self.docker_lbs.keys())
        len(node_list)
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

                        if weights[key] == 0 :
                            weights[key] = 1

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
                if container_name in name:
                    self.create_lb_service(container_name, weights)
            node_no = node_no + 1
            vnf_no = 0
            service_no = 0
            node_dst_no = 0
        pass

    def set_weight(self, weight):
        """
        Update the weight to each load balancer containers.
        Input: weight array:
        [node, sfc, sf, node_dst ]
        [(node_0, search_service, search_lb, node_0), (node_0, search_service, search_lb, node_1), ...]
        Output: update the load balancer service.
        """
        node_no = 0
        service_no = 0
        vnf_no = 0
        node_dst_no = 0
        node_list = list(self.docker_lbs.keys())
        for container in self.docker_lbs.values():
            for name, weights in container.items():
                for key in weights.keys():
                    if("WEIGHT" in str(key)):
                        weights[key] = int(round(weight[node_no][service_no][vnf_no][node_dst_no] * 100))

                        if weights[key] == 0 :
                            weights[key] = 1

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
                        
                service = self.get_service(name)
                logger.info(f"service: {name} -> weights: {weights}")
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
        try: 
            service_updated = service.update(env=env_list)
            logger.info(f"service_updated: {service_updated}")
            return True
        except docker.errors.APIError:
            print(docker.errors.APIError)
            return False

    def create_lb_service(self,client_container_name, env_dict):
        """
        Create the client with the env list
        """
        #prepare image:
        image = "localhost:5000/load-balancer:latest"
        #prepare env list:
        env_list = list()
        print(env_list)
        search_port_number = env_dict["PORT_ADDRESS_SEARCH_LISTEN"]
        shop_port_number = env_dict["PORT_ADDRESS_SHOP_LISTEN"]
        web_port_number = env_dict["PORT_ADDRESS_WEB_LISTEN"]
        media_port_number = env_dict["PORT_ADDRESS_MEDIA_LISTEN"]

        for env_key,value in env_dict.items():
            env_list.append(f"{env_key}={value}")
        print(env_list)
        #prepare published port:
        endpoint_spec = docker.types.EndpointSpec(mode= 'vip', ports= {search_port_number: (search_port_number, "tcp", "host"),
                                                                       shop_port_number: (shop_port_number, "tcp", "host"),
                                                                       web_port_number: (web_port_number, "tcp", "host"),
                                                                       media_port_number: (media_port_number, "tcp", "host")})
        
        #constraint:
        node_number = ''.join(x for x in client_container_name if x.isdigit())
        constraint_str = str('node.labels.node_number==' + str(node_number))
        print(constraint_str)
        # constraint = docker.types.Placement(maxreplicas=1)
        self.client.services.create(image, name= client_container_name, endpoint_spec=endpoint_spec, env=env_list, constraints=[constraint_str])
        # self.client.services.create(image, name= client_container_name, env=env_list)
        pass

    def create_client_service(self, container_name, weights):
        """
        Create the lb with the env list
        """
        #prepare image:
        image = "localhost:5000/service-client:latest"
        #prepare env list:
        env_list = list()
        port_number = weights["PORT_NUMBER"] + 100
        for env_key,value in weights.items():
            env_list.append(f"{env_key}={value}")
        #prepare published port:
        endpoint_spec = docker.types.EndpointSpec(mode= 'vip', ports= {port_number: (port_number, "tcp", "host")})
        
        #constraint:
        node_number = ''.join(x for x in container_name if x.isdigit())
        constraint_str = str('node.labels.node_number==' + str(node_number))
        print(constraint_str)
        # constraint = docker.types.Placement(maxreplicas=1)
        self.client.services.create(image, name= container_name, endpoint_spec=endpoint_spec, env=env_list, constraints=[constraint_str])
        # self.client.services.create(image, name= client_container_name, env=env_list)
        pass

    def reload_service(self, service_id):
        """
        update the env list of a service:
        input: env_list: dict(): {"env_1": 14, "env_2": 16}
                service_id: id of service to update
        """
        # env_list = list()
        # for env_key,value in env_dict.items():
        #     env_list.append(f"{env_key}={value}")
        service = self.client.services.get(service_id)
        service.reload()

    def remove_service(self, service_id):
        """
        remove a service
        """
        service = self.client.services.get(service_id)
        service.remove()


    def force_update_service(self, service_id):
        """
        update the env list of a service:
        input: env_list: dict(): {"env_1": 14, "env_2": 16}
                service_id: id of service to update
        """
        # env_list = list()
        # for env_key,value in env_dict.items():
        #     env_list.append(f"{env_key}={value}")
        service = self.client.services.get(service_id)
        try: 
            success = service.force_update()
        except:
            logger.info(f"error in update service: {service_id}")
        logger.info(f"success = {success}")
        return success

    def create_client(self, client_container_name, traces):
        """
        create client with container name 
        """
        for trace in traces: 
            node_list = list(self.docker_clients.keys())
            for node in node_list:
                for key,value in self.docker_clients[node].items():

                    if client_container_name in key:
                        if value['PORT_NUMBER'] == 8001: 
                            user = float(trace['search_service']) * self.get_ingress_distribution[node]['search']
                        elif value['PORT_NUMBER'] == 8002: 
                            user = float(trace['shop_service']) * self.get_ingress_distribution[node]['shop']
                        elif value['PORT_NUMBER'] == 8003: 
                            user = float(trace['web_service']) * self.get_ingress_distribution[node]['web']
                        elif value['PORT_NUMBER'] == 8004: 
                            user = float(trace['media_service']) * self.get_ingress_distribution[node]['media']
                        else :
                            print("Fuckin wrong here!")
                        user_no = int(user)
                        value['USER_NO'] = user_no
                        print(value)
                        self.create_client_service(client_container_name, value)
        pass

    def update_clients(self, traces):
        """
        input: list of trace dict:
        {time: 1, search_client: 200, shop_client: 300, web_client: 500, media_client:80}  
        """
        for trace in traces: 
            node_list = list(self.docker_clients.keys())
            for node in node_list:
                for key,value in self.docker_clients[node].items():
                    service = self.get_service(key)
                    if value['PORT_NUMBER'] == 8001: 
                        user = float(trace['search_service']) * self.get_ingress_distribution[node]['search']
                    elif value['PORT_NUMBER'] == 8002: 
                        user = float(trace['shop_service']) * self.get_ingress_distribution[node]['shop']
                    elif value['PORT_NUMBER'] == 8003: 
                        user = float(trace['web_service']) * self.get_ingress_distribution[node]['web']
                    elif value['PORT_NUMBER'] == 8004: 
                        user = float(trace['media_service']) * self.get_ingress_distribution[node]['media']
                    else :
                        print("Fuckin wrong here!")
                    user_no = int(user)

                    if value['USER_NO'] == user_no:
                        self.force_update_service(service)
                    else:
                        value['USER_NO'] = user_no
                        self.update_service(service,value)
    
    def remove_containers(self, container_list):
        for container in container_list:
            service = self.get_service(container)
            self.remove_service(service)

    def restore_clients(self, client_container_list, step_count):
        """
        Input: container list needed to be set up.
        Input: time step count. 
        restore the client containers with the correct time step.
        """
        for container in client_container_list:
            time_trace = self.get_trace_time(step_count)
            self.create_client(container, time_trace)

    def restore_lbs(self, lb_container_list, scheduling): 
        """
        Input: container list needed to be set up.
        Input: time step count. 
        restore the client containers with the correct time step.
        """
        logger.info(f"Restore the lb container: {lb_container_list}, scheduling: {scheduling}")
        for container in lb_container_list:
            self.create_lb(container, scheduling)

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
# # docker_helper.set_weight(weight=weight)
# # docker_helper.set_user_number(3)
# container_list = ["search_client_4"]
# # docker_helper.reset_lb_container(container_list, weight)
# docker_helper.restore_clients(container_list, 10)

# service_id = docker_helper.get_service("web_client_4")
# env_list = {"USER_NO":90}
# docker_helper.update_service(service_id, env_list)
