import requests
import urllib3
from prometheus_api_client import PrometheusConnect
from rlsp.utils.util_functions import get_docker_services
import time
import logging
logger = logging.getLogger(__name__)

docker_client_services_path = 'res/containers/client_containers.yaml'
docker_server_services_path = 'res/containers/server_containers.yaml'
docker_lb_container_path = 'res/containers/load_balancer_containers.yaml'
ingress_distribution_file_path = 'res/service_functions/metro_network_ingress_distribution.yaml'
CAPTURE_TIME = 20

class CaptureHelper():
    def __init__(self, docker_client_services_path, docker_server_services_path, ingress_distribution_file_path, docker_lb_container_path, service_list):
        self.docker_client_services = get_docker_services(docker_client_services_path)
        self.docker_server_services = get_docker_services(docker_server_services_path)
        self.get_ingress_distribution = get_docker_services(ingress_distribution_file_path)
        self.docker_lb_services = get_docker_services(docker_lb_container_path)
        self.prom = PrometheusConnect(url="http://131.155.35.54:9090", disable_ssl=True)
        self.capture_time = CAPTURE_TIME
        self.service_list = service_list
        pass

    def capture_data(self, ingress_nodes, time_up_already):
        """
        Input: array of ingress nodes to measure
        ex: ingress_nodes = ["node1", "node2"]
        Get the latency from Prometheous server API
        return:
        [

            latency = {"node4":{"search":24, "shop":26, "web":{self.capture_time}, "media":10}, 
                    "node3":{"search":24, "shop":26, "web":{self.capture_time}, "media":10}} -> for each client in ingress node
            dropped_traffic = {"node4":{"search":24, "shop":26, "web":{self.capture_time}, "media":10}, 
                    "node3":{"search":24, "shop":26, "web":{self.capture_time}, "media":10}
                    "node2: {"search":24, "shop":26, "web":{self.capture_time}, "media":10}} -> for each server in edge node
            ingress_bw = {"node4":{"search":200, "shop":560, "web":100, "media":{self.capture_time}00}, 
                    "node3":{"search":250, "shop":700, "web":450, "media":2900}} -> for each client in ingress node
        ]
        """
        # sleep 15s, 5s for stablize, 10s for calculation
        sleep_time = self.capture_time
        if time_up_already < self.capture_time:
            sleep_time = self.capture_time - time_up_already
        elif time_up_already < self.capture_time * 1.5:
            sleep_time = 1
        logger.info(f"sleep_time: {sleep_time}")
        time.sleep(sleep_time)
        # Latency:
        
        
        # Ingress request number:
        ingress_request = self.calculate_ingress_request()
        # Dropped connections:
        # pre_dropped_conn = self.calculate_dropped_connection()
        # dropped_conn = self.pre_process(pre_dropped_conn)

        # pre_success_conn = self.calculate_success_connection()
        # success_conn = self.pre_process(pre_success_conn)
        pre_succ_request = self.get_metric_value("success_conn_total")
        succ_request = self.pre_process(pre_succ_request)

        pre_drop_request = self.get_metric_value("drop_conn_total")
        drop_request = self.pre_process(pre_drop_request)

        # pre_latency = self.calculate_latency(ingress_nodes)
        pre_latency = self.calculate_latency_histogram()
        latency = self.pre_process(pre_latency)

        # self.calculate_latency_histogram()
        # latency = {"node4": {"search": 24, "shop": 26, "web": {self.capture_time}, "media": 10},
        #            "node3": {"search": 24, "shop": 26, "web": {self.capture_time}, "media": 10}}

        # ingress_request = {"node4": {"search": 200, "shop": 560, "web": 100, "media": {self.capture_time}00},
        #                    "node3": {"search": 250, "shop": 700, "web": 450, "media": 2900}}

        return latency, drop_request, succ_request, ingress_request

    def check_lb_containers(self):
        """
        Check if any lb container (except media for now) is empty or not up yet and return the name!
        """
        working = True
        container_list = list()
        for node, container in self.docker_lb_services.items():
            for cont_name, cont_value in container.items():
                number = ''.join(x for x in cont_name if x.isdigit())
                url = "http://" + cont_value['IP_ADDRESS_NODE_' + str(number)] + ":" + str(cont_value['PORT_ADDRESS_SEARCH_LISTEN'])
                # TODO: try and catch here for the module!
                # try:
                response = check_connection(url)
                if not response:
                    logger.error(f"Error in container: {cont_name}")
                    container_list.append(cont_name)
        return working, container_list

    def check_client_containers(self):
        """
        Check if any client container (except media for now) is empty or not up yet and return the name!
        """
        time_up_already = list()
        working = True
        container_list = list()
        # print(metrics_array)
        for node, container in self.docker_client_services.items():
            for cont_name, cont_value in container.items():
                if self.is_container_in_service_list(container_name=cont_name) and self.is_container_has_user(node=node, container_name=cont_name):
                    url = "http://" + cont_value['IP_ADDRESS'] + ":" + str(int(cont_value['PORT_NUMBER']) + 100) + "/metrics"
                    # print(url)
                    # print(cont_name)
                    response = check_connection(url)
                    # print(response.status_code)
                    if not response :
                        logger.error(f"Error in container: {cont_name}")
                        container_list.append(cont_name)
                    else:
                        conn_value = -1
                        metrics_array = self.prom.custom_query(query="summary_request_latency_seconds_count")
                        
                        flag_got_value, conn_value_get = self.get_value(cont_name, metrics_array)
                        logger.error(f"container: {cont_name} conn_value_get: {conn_value_get} with flag: {flag_got_value}")
                        if flag_got_value == True:
                            conn_value = int(float(conn_value_get[1]))
                            if conn_value == 0:
                                container_list.append(cont_name)
                            else:
                                time_up_already.append(conn_value)
                        else: 
                            container_list.append(cont_name)
        #FIXME: ignore media containers for now to work with other 3 services first!
        con_list = []
        for con in container_list:
            if "media" in con:
                continue
            con_list.append(con)
        if len(con_list) > 0:
            working = False
        logger.info(f"time for each node: {time_up_already}")
        time_return = 0
        if(len(time_up_already) > 0):
            time_return = min(time_up_already)
        return time_return, working, con_list
        # for container in name_list:
        #     if requests.get(URL)
        # self.prom

    def pre_process(self, arr_data):
        """
        input: latency = {"node4": {"search_client_4": 24, "shop": 26, "web": {self.capture_time}, "media": 10},
                    "node3": {"search": 24, "shop": 26, "web": {self.capture_time}, "media": 10}}
        based on ingress_distribution: 
        output: latency = {"search_service": 24, "shop_service": 26, "web_service": {self.capture_time}, "media_service":10}
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
                        if value == -1 or self.get_ingress_distribution[node][service] == 0:
                            continue
                        else :
                            total_service_factor += 1
                            data[service] += value
            if total_service_factor != 0:
                data[service] = data[service] / total_service_factor
        return data
    
    
    def calculate_latency_histogram(self):
        latency = dict()
        for node, container in self.docker_client_services.items():
            cont_dict = dict()
            for container in container.keys():
                if self.is_container_in_service_list(container):
                    logger.info(f"histogram_quantile(0.9, sum(rate(request_latency_seconds_{container}_bucket[{self.capture_time}s])) by (le))")
                    metric_array = self.prom.custom_query(query=f"histogram_quantile(0.9, sum(rate(request_latency_seconds_{container}_bucket[{self.capture_time}s])) by (le))")
                    logger.info(f"metric_array: {metric_array}")
                    value_latency = metric_array[0]['value'][1]
                    # if container in list(metric_dict.keys()):
                    if(value_latency == 'NaN'):
                        value = 0
                    else: 
                        value = float(value_latency)
                    cont_dict[container] = value
            latency[node] = cont_dict
        print(latency)
        return latency    

    def calculate_latency(self, ingress_nodes):
        metrics_array = self.prom.custom_query(query=f"rate(summary_request_latency_seconds_sum[{self.capture_time}s])")
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
        # print(latency)
        return latency

    def get_metric_value(self, cmd):
        metrics_array = self.prom.custom_query(query=cmd)
        return_value = dict()
        for node, container in self.docker_client_services.items():
            cont_dict = dict()
            for container in container.keys():
                if self.is_container_in_service_list(container_name=container):
                    val = -1
                    flag_got_value, value_get = self.get_value(container, metrics_array)
                    if flag_got_value == True:
                        val = int(float(value_get[1]))
                    cont_dict[container] = val
            return_value[node] = cont_dict
        # print(return_value)
        return return_value

    def get_array_values(self, container, query_cmd):
        metrics_array_succ = self.prom.custom_query(query=query_cmd)
        flag_got_value, values = self.get_values(container, metrics_array_succ)
        arr = list()
        if flag_got_value: 
            arr = [int(float(value[1])) for value in values]
        return arr

    def is_container_has_ingress(self, container_name, node):
        has_ingress_traffic = False
        # logger.info(f"container_name: {container_name}")
        for service, value in self.get_ingress_distribution[node].items():
            # logger.info(f"service: {service}")
            # logger.info(f"value: {value}")
            if service in container_name and value != 0:
                has_ingress_traffic =True
                return has_ingress_traffic
        return has_ingress_traffic
        
    def is_container_in_service_list(self, container_name):
        is_found = False
        for service in self.service_list:
            if service in container_name:
                is_found = True
        return is_found

    def is_container_has_user(self, node, container_name):
        is_found = False
        for service, value in self.get_ingress_distribution[node].items():
            if service in container_name and value > 0:
                is_found = True
        return is_found

    def calculate_ingress_request(self):
        # metrics_array_succ = self.prom.custom_query(query="success_conn_total[10s]")
        # metrics_array_drop = self.prom.custom_query(query="drop_conn_total[10s]")
        # print(metrics_array_succ)
        # print(metrics_array_drop)
        
        # working = True
        ingress_request = dict()
        for node, container in self.docker_client_services.items():
            cont_dict = dict()
            for container in container.keys():
                if self.is_container_in_service_list(container_name=container):
                    average_ingress_requests = 0
                    qualify_array_capture = False
                    while(qualify_array_capture == False):
                        arr_succ = self.get_array_values(container, f"success_conn_total[{self.capture_time}s]")
                        arr_drop = self.get_array_values(container, f"drop_conn_total[{self.capture_time}s]")
                        # logger.info(f"container: {container}")
                        # logger.info(self.get_ingress_distribution)

                        logger.info(f"arr_succ: {arr_succ}")
                        logger.info(f"arr_drop: {arr_drop}")
                        # logger.info(f"length arr_succ: {len(arr_succ)}")
                        # logger.info(f"length arr_drop: {len(arr_drop)}")
                        # logger.info(f"self.capture_time: {self.capture_time}")
                        # check if the length is enough?
                        if len(arr_drop) == self.capture_time and len(arr_succ) == self.capture_time:
                            if self.is_container_has_ingress(container_name=container, node=node):
                                if 0 in arr_succ: 
                                    logger.info(f"There is a 0")
                                    qualify_array_capture = False
                                    time.sleep(1)
                                else:
                                    qualify_array_capture = True
                            else:
                                qualify_array_capture = True
                        else:
                            logger.info(f"In here")
                            qualify_array_capture = False
                            time.sleep(1)

                    arr_sum = [arr_succ[i] + arr_drop[i] for i in range(len(arr_succ))]
                    logger.info(f"arr_sum: {arr_sum}")
                    diff = [arr_sum[i+1] - arr_sum[i] for i in range(len(arr_sum) - 1)]
                    logger.info(f"diff: {diff}")
                    logger.info(f"sum(diff): {sum(diff)} len(diff): {len(diff)}")
                    average_ingress_requests = float(sum(diff)/len(diff))
                    cont_dict[container] = average_ingress_requests
            ingress_request[node] = cont_dict
        print(ingress_request)
        return ingress_request

    def calculate_dropped_connection(self):
        metrics_array_accepted = self.prom.custom_query(query=f"nginx_connections_accepted[{self.capture_time}s]")
        metrics_array_handled = self.prom.custom_query(query=f"nginx_connections_handled[{self.capture_time}s]")
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
        return dropped_connection
    
    def calculate_capacity(self):
        metrics_array_handled = self.prom.custom_query(query=f"nginx_connections_handled[{self.capture_time}s]")
        succ_connection = dict()
        for node, container in self.docker_server_services.items():
            cont_dict = dict()
            for container in container.keys():
                succ_conn = -1
                flag_got_value_handled, handled_conn = self.get_values(container, metrics_array_handled)

                print("\n\n")
                print(container)
                print(handled_conn)
                if flag_got_value_handled == True:
                    array_handled = [int(float(value[1])) for value in handled_conn]
                    print(array_handled)
                    succ_conn = array_handled[-1] - array_handled[0]
                cont_dict[container] = succ_conn / self.capture_time
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

def check_connection(url):
    try:
        requests.get(url)
        return True
    except : 
        return False

# ingress_nodes = ["node3", "node2"]
# service_list = ["search"]
# capture_helper = CaptureHelper(docker_client_services_path= docker_client_services_path, docker_server_services_path= docker_server_services_path, ingress_distribution_file_path = ingress_distribution_file_path, docker_lb_container_path=docker_lb_container_path, service_list=service_list)
# capture_helper.calculate_capacity()
# capture_helper.calculate_latency_histogram()
# # capture_helper.calculate_latency(ingress_nodes)
# # capture_helper.calculate_ingress_request(ingress_nodes)
# capture_helper.capture_data(ingress_nodes)
