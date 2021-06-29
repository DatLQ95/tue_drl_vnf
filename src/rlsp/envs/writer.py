"""
Simulator file writer module
"""

import csv
import os
import yaml

class ResultWriter():
    """
    Result Writer
    Helper class to write results to CSV files.
    """
    def __init__(self, test_dir):
        """
        If the simulator is in test mode, create result folder and CSV files
        """
        self.placement_file_name = f"{test_dir}/placements.csv"
        self.drop_request_metrics_file_name = f"{test_dir}/drop_request.csv"
        self.succ_request_metrics_file_name = f"{test_dir}/succ_request.csv"
        self.latency_metric_file_name = f"{test_dir}/latency.csv"
        self.scheduling_file_name = f"{test_dir}/scheduling.csv"
        self.rewards_file_name = f"{test_dir}/rewards.csv"
        self.ingress_traffic_file_name = f"{test_dir}/ingress_traffic.csv"
        self.runtimes_file_name = f"{test_dir}/runtimes.csv"

        # Create the results directory if not exists
        os.makedirs(os.path.dirname(self.placement_file_name), exist_ok=True)

        self.placement_stream = open(self.placement_file_name, 'a+', newline='')
        self.drop_request_metrics_stream = open(self.drop_request_metrics_file_name, 'a+', newline='')
        self.succ_request_metrics_stream = open(self.succ_request_metrics_file_name, 'a+', newline='')
        self.latency_metric_stream = open(self.latency_metric_file_name, 'a+', newline='')
        self.scheduling_stream = open(self.scheduling_file_name, 'a+', newline='')
        self.rewards_stream = open(self.rewards_file_name, 'a+', newline='')
        self.ingress_traffic_stream = open(self.ingress_traffic_file_name, 'a+', newline='')
        self.runtimes_stream = open(self.runtimes_file_name, 'a+', newline='')

        # Create CSV writers
        self.placement_writer = csv.writer(self.placement_stream)
        self.drop_request_metrics_writer = csv.writer(self.drop_request_metrics_stream)
        self.succ_request_metrics_writer = csv.writer(self.succ_request_metrics_stream)
        self.latency_metric_writer = csv.writer(self.latency_metric_stream)
        self.scheduling_writer = csv.writer(self.scheduling_stream)
        self.rewards_writer = csv.writer(self.rewards_stream)
        self.ingress_traffic_writer = csv.writer(self.ingress_traffic_stream)
        self.runtimes_writer = csv.writer(self.runtimes_stream)
        self.action_number = 0

        # Write the headers to the files
        self.create_csv_headers()

    def close_stream(self):
        # Close all writer streams
        self.placement_stream.close()
        self.drop_request_metrics_stream.close()
        self.succ_request_metrics_stream.close()
        self.latency_metric_stream.close()
        self.scheduling_stream.close()
        self.ingress_traffic_stream.close()
        self.rewards_stream.close()

    def create_csv_headers(self):
        """
        Creates statistics CSV headers and writes them to their files
        """
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        # Create CSV headers
        placement_output_header = ['episode', 'time', 'node', 'service', 'sf']
        scheduling_output_header = ['episode', 'time', 'origin_node', 'service', 'sf', 'schedule_node', 'schedule_prob']
        drop_request_metrics_output_header = ['episode', 'time', 'search', 'shop', 'web', 'media']
        succ_request_metrics_output_header = ['episode', 'time', 'search', 'shop', 'web', 'media']
        latency_output_header = ['episode', 'time', 'search', 'shop', 'web', 'media']
        reward_output_header = ['episode', 'time', 'total_reward', 'request_reward', 'latency_reward']
        ingress_output_header = ['episode', 'time', 'node4', 'node3', 'node2']
        runtimes_output_header = ['run', 'runtime', 'agent_time']


        # Write headers to CSV files
        print("Write headers to file!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.placement_writer.writerow(placement_output_header)
        self.scheduling_writer.writerow(scheduling_output_header)
        self.drop_request_metrics_writer.writerow(drop_request_metrics_output_header)
        self.succ_request_metrics_writer.writerow(succ_request_metrics_output_header)
        self.latency_metric_writer.writerow(latency_output_header)
        self.rewards_writer.writerow(reward_output_header)
        self.ingress_traffic_writer.writerow(ingress_output_header)
        self.runtimes_writer.writerow(runtimes_output_header)

    def write_runtime(self, time, agent_time):
        """
        Write runtime results to output file
        """
        self.action_number += 1
        self.runtimes_writer.writerow([self.action_number, time, agent_time])

    # def write_placement_file(self, action):
        
    #     pass

    # def write_scheduling_file(self, action):
    #     pass

    # def write_request_stats_file(self, succ_request, drop_request):
    #     pass

    # def write_latency_file(self, latency):
    #     pass

    # def reward_file(self, total_reward, request_reward, latency_reward):
        
    #     pass

    def record_reward(self, episode, step, total_reward, request_reward, delay_reward):
        reward_content = list()
        reward_content.append(episode)
        reward_content.append(step)
        reward_content.append(total_reward)
        reward_content.append(request_reward)
        reward_content.append(delay_reward)
        self.rewards_writer.writerow(reward_content)
        pass

    def record_capture_data(self, episode, step, latency, dropped_conn, success_conn, capture_traffic):
        
        latency_content = list()
        latency_content.append(episode)
        latency_content.append(step)
        for service, value in latency.items():
            latency_content.append(value)
        self.latency_metric_writer.writerow(latency_content)

        drop_request_content = list()
        drop_request_content.append(episode)
        drop_request_content.append(step)
        for service, value in dropped_conn.items():
            drop_request_content.append(value)
        self.drop_request_metrics_writer.writerow(drop_request_content)

        succ_request_content = list()
        succ_request_content.append(episode)
        succ_request_content.append(step)
        for service, value in success_conn.items():
            succ_request_content.append(value)
        self.succ_request_metrics_writer.writerow(succ_request_content)

        #FIXME: change this when the scenario is changed
        ingress_traffic = list()
        ingress_traffic.append(episode)
        ingress_traffic.append(step)
        ingress_traffic.extend(capture_traffic)
        self.ingress_traffic_writer.writerow(ingress_traffic)

    def record_action(self, episode, step, action):
        node_list = ['node4', 'node3', 'node2']
        service_list = ['search']
        sf_list = ['lb', 'server']

        for node_index, node in enumerate(node_list):
            for service_index, service in enumerate(service_list):
                for sf_index, sf in enumerate(sf_list):
                    for node_dst_index, node_dist in enumerate(node_list):
                        scheduling_content = list()
                        placement_content = list()
                        if action[node_index][service_index][sf_index][node_dst_index] != 0:
                            placement_content.append(episode)
                            placement_content.append(step)
                            placement_content.append(node)
                            placement_content.append(service)
                            placement_content.append(sf)
                            self.placement_writer.writerow(placement_content)
                        scheduling_content.append(episode)
                        scheduling_content.append(step)
                        scheduling_content.append(node)
                        scheduling_content.append(service)
                        scheduling_content.append(sf)
                        scheduling_content.append(node_dist)
                        scheduling_content.append(action[node_index][service_index][sf_index][node_dst_index])
                        self.scheduling_writer.writerow(scheduling_content)
        pass