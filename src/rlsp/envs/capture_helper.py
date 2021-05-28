import urllib.request
import json
import requests
from prometheus_api_client import PrometheusConnect
from rlsp.utils.util_functions import get_docker_services
import time

docker_client_services_path = 'res/containers/client_containers.yaml'
docker_server_services_path = 'res/containers/server_containers.yaml'
ingress_distribution_file_path = 'res/service_functions/metro_network_ingress_distribution.yaml'


class CaptureHelper():
    def __init__(self, docker_client_services_path, docker_server_services_path, ingress_distribution_file_path):
        self.docker_client_services = get_docker_services(docker_client_services_path)
        self.docker_server_services = get_docker_services(docker_server_services_path)
        self.get_ingress_distribution = get_docker_services(ingress_distribution_file_path)
        pass

    def capture_data(self, ingress_nodes):
        """
        Input: array of ingress nodes to measure
        ex: ingress_nodes = ["node1", "node2"]
        Get the latency from Prometheous server API
        return:
        [

            latency = {"node4":{"search":24, "shop":26, "web":30, "media":10}, 
                    "node3":{"search":24, "shop":26, "web":30, "media":10}} -> for each client in ingress node
            dropped_traffic = {"node4":{"search":24, "shop":26, "web":30, "media":10}, 
                    "node3":{"search":24, "shop":26, "web":30, "media":10}
                    "node2: {"search":24, "shop":26, "web":30, "media":10}} -> for each server in edge node
            ingress_bw = {"node4":{"search":200, "shop":560, "web":100, "media":3000}, 
                    "node3":{"search":250, "shop":700, "web":450, "media":2900}} -> for each client in ingress node
        ]
        """
        self.prom = PrometheusConnect(url="http://131.155.35.54:9090", disable_ssl=True)
        # sleep 15s, 5s for stablize, 10s for calculation
        # time.sleep(10)
        # Latency:
        pre_latency = self.calculate_latency(ingress_nodes)
        latency = self.pre_process(pre_latency)
        # Ingress request number:
        ingress_request = self.calculate_ingress_request(ingress_nodes)
        # Dropped connections:
        pre_dropped_conn = self.calculate_dropped_connection()
        dropped_conn = self.pre_process(pre_dropped_conn)

        pre_success_conn = self.calculate_success_connection()
        success_conn = self.pre_process(pre_success_conn)

        # latency = {"node4": {"search": 24, "shop": 26, "web": 30, "media": 10},
        #            "node3": {"search": 24, "shop": 26, "web": 30, "media": 10}}

        # ingress_request = {"node4": {"search": 200, "shop": 560, "web": 100, "media": 3000},
        #                    "node3": {"search": 250, "shop": 700, "web": 450, "media": 2900}}

        return latency, dropped_conn, success_conn, ingress_request

    def pre_process(self, arr_data):
        """
        input: latency = {"node4": {"search_client_4": 24, "shop": 26, "web": 30, "media": 10},
                    "node3": {"search": 24, "shop": 26, "web": 30, "media": 10}}
        based on ingress_distribution: 
        output: latency = {"search_service": 24, "shop_service": 26, "web_service": 30, "media_service":10}
        """
        data = dict()
        service_list = list(self.get_ingress_distribution['node4'].keys())
        for service in service_list:
            data[service] = 0
        # for each service: 
        for service in service_list:
            total_service_factor = 0
            for node, container in arr_data.items():
                for client, value in container.items():
                    if service in client:
                        if value == -1 :
                            continue
                        else :
                            total_service_factor += 1
                            data[service] += value
            data[service] = data[service] / total_service_factor
        print(data)
        return data

    def calculate_latency(self, ingress_nodes):
        metrics_array = self.prom.custom_query(query="rate(summary_request_latency_seconds_sum[10s])")
        latency = dict()
        for node, container in self.docker_client_services.items():
            cont_dict = dict()
            for container in container.keys():
                latency_value = -1
                flag_got_value, latency_value_get = self.get_value(container, metrics_array)
                if flag_got_value == True:
                    latency_value = int(float(latency_value_get[1]))
                cont_dict[container] = latency_value
            latency[node] = cont_dict
        print(latency)
        return latency
        
    def calculate_ingress_request(self, ingress_nodes):
        metrics_array = self.prom.custom_query(query="request_number_total[10s]")
        ingress_request = dict()
        for node, container in self.docker_client_services.items():
            cont_dict = dict()
            for container in container.keys():
                average_ingress_requests = 0
                flag_got_value, ingress_request_values = self.get_values(container, metrics_array)
                if flag_got_value == True:
                    arr = [int(float(value[1])) for value in ingress_request_values]
                    diff = [arr[i+1] - arr[i] for i in range(len(arr) - 1)]
                    average_ingress_requests = float(sum(diff)/len(diff))
                cont_dict[container] = average_ingress_requests
            ingress_request[node] = cont_dict
        print(ingress_request)
        return ingress_request

    def calculate_dropped_connection(self):
        metrics_array_accepted = self.prom.custom_query(query="nginx_connections_accepted[10s]")
        metrics_array_handled = self.prom.custom_query(query="nginx_connections_handled[10s]")
        dropped_connection = dict()
        for node, container in self.docker_server_services.items():
            cont_dict = dict()
            for container in container.keys():
                dropped_conn = -1
                flag_got_value_accepted, accepted_conn = self.get_values(container, metrics_array_accepted)
                flag_got_value_handled, handled_conn = self.get_values(container, metrics_array_handled)
                if flag_got_value_handled == True and flag_got_value_accepted == True:
                    array_accepted = [int(float(value[1])) for value in accepted_conn]
                    array_handled = [int(float(value[1])) for value in handled_conn]
                    array_dropped = [array_handled[i] - array_accepted[i] for i in range(len(array_accepted))]
                    dropped_conn = sum(array_dropped)
                cont_dict[container] = dropped_conn
            dropped_connection[node] = cont_dict
        print(dropped_connection)
        return dropped_connection
    
    def calculate_success_connection(self):
        metrics_array_handled = self.prom.custom_query(query="nginx_connections_handled[10s]")
        succ_connection = dict()
        for node, container in self.docker_server_services.items():
            cont_dict = dict()
            for container in container.keys():
                succ_conn = -1
                flag_got_value_handled, handled_conn = self.get_values(container, metrics_array_handled)
                if flag_got_value_handled == True:
                    array_handled = [int(float(value[1])) for value in handled_conn]
                    succ_conn = array_handled[-1] - array_handled[0]
                cont_dict[container] = succ_conn
            succ_connection[node] = cont_dict
        print(succ_connection)
        return succ_connection

    def get_value(self, container, metrics_array):
        return_value = 1
        flag_got_value = False
        for job in metrics_array:
            if job['metric']['job'] == container:
                return_value = job['value']
                flag_got_value = True
                break
        return flag_got_value, return_value

    def get_values(self, container, metrics_array):
        return_value = 1
        flag_got_value = False
        for job in metrics_array:
            if job['metric']['job'] == container:
                return_value = job['values']
                flag_got_value = True
                break
        return flag_got_value, return_value

ingress_nodes = ["node4"]
capture_helper = CaptureHelper(docker_client_services_path= docker_client_services_path, docker_server_services_path= docker_server_services_path, ingress_distribution_file_path = ingress_distribution_file_path)
capture_helper.capture_data(ingress_nodes)
