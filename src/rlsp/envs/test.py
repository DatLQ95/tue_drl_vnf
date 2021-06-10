from os import name
from typing import List
import docker
import logging
import csv
import yaml
logger = logging.getLogger(__name__)

docker_services_path = 'res/containers/client_containers.yaml'

class DockerHelper():
    def __init__(self, user_number_trace_file):
        """
        Create docker helper with the user_trace from all clients in all node
        user_number_trace: list of dict 
        """
        self.client = docker.from_env()
        self.trace = get_trace(user_number_trace_file)
        self.docker_services = get_docker_services(docker_services_path)
        pass

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
        """
        logger.info(f"weights: {weight}")
        
        #TODO: round the weights then set it in each node!
        pass

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
        {time: 1, node : node0, search_client: 200
                shop_client: 300
                web_client: 500
                media_client:80},
        {time: 1, node : node1, search_client: 200
                shop_client: 300
                web_client: 500
                media_client:80}   
        """
        for trace in traces: 
            logger.info(f"update_client {trace}")
            print(trace)
            # TODO: check if success after update by using raise to catch error:
            # node4 = edge4
            # node3 = edge3
            # node2 = edge2
            # print(self.docker_services[trace['node']])
            for key,value in self.docker_services[trace['node']].items():
                service = self.get_service(key)
                if value['PORT_NUMBER'] == 8001: 
                    value['USER_NO'] = trace['search_service']
                elif value['PORT_NUMBER'] == 8002: 
                    value['USER_NO'] = trace['shop_service']
                elif value['PORT_NUMBER'] == 8003: 
                    value['USER_NO'] = trace['web_service']
                elif value['PORT_NUMBER'] == 8004: 
                    value['USER_NO'] = trace['media_service']
                else :
                    print("Fuckin wrong here!")
                print(value)
                self.update_service(service,value)
        pass

    def create_servers(): 
        """
        container: search_server 
        image: search-server
        port: 8983
        lb port: 8984
        input request port: 8001
        docker service create --mode global --publish mode=host,target=80,published=8983 --name search_server luongquocdat01091995/network_services:search-server

        -------------------------------------------------------------------

        container: shop_server 
        image: shop-server
        port: 8080
        lb port: 8081
        input request port: 8002
        docker service create --mode global --publish mode=host,target=80,published=8080 --name shop_server luongquocdat01091995/network_services:shop-server

        -------------------------------------------------------------------

        container: web_server 
        image: web-server
        port: 8096
        lb port: 8097
        input request port: 8003
        docker service create --mode global --publish mode=host,target=80,published=8096 --name web_server luongquocdat01091995/network_services:web-server

        -------------------------------------------------------------------

        container: media_server 
        image: media-server
        port: 8088
        lb port: 8089
        input request port: 8004
        docker service create --mode global --publish mode=host,target=80,published=8088 --publish mode=host,target=1935,published=1935 --name media_server media-server
        """

        pass

    def create_load_balancers():
        pass

    def create_node_load_balancers():
        pass

    def create_service_clients():
        pass

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
    with open(trace_file) as f:
        trace_rows = csv.DictReader(f)
        traces = []
        for row in trace_rows:
            traces.append(dict(row))
    return traces

def get_docker_services(docker_services_path):
    with open(docker_services_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# docker_helper = DockerHelper(user_number_trace_file='res/traces/trace_metro_network_users.csv')
# weight = [random.random()]
# docker_helper.set_user_number(3)

# service_id = docker_helper.get_service("web_client_4")
# env_list = {"USER_NO":90}
# docker_helper.update_service(service_id, env_list)
