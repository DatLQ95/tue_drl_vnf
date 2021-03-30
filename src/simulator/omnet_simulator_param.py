"""

Flow Simulator parameters.
- Allows for clean and quick access to parameters from the flow simulator.
- Facilitates the quick changing of schedule decisions and
other parameters for the simulator.

"""
import logging
import numpy as np
import random
from spinterface import SimulatorAction, SimulatorInterface, SimulatorState
import json
logger = logging.getLogger(__name__)

class SimulatorParam:
    def __init__(self, network, ing_nodes, eg_nodes, sfc_list, sf_list, config,
                 schedule=None, sf_placement=None):
        # NetworkX network object: DiGraph
        self.network = network
        # Ingress nodes of the network (nodes at which flows arrive): list
        self.ing_nodes = ing_nodes
        # Egress nodes of the network (nodes at which flows may leave the network): list
        self.eg_nodes = eg_nodes
        # List of available SFCs and their child SFs: defaultdict(None)
        self.sfc_list = sfc_list
        # List of every SF and it's properties (e.g. processing_delay): defaultdict(None)
        self.sf_list = sf_list
        # Set sim config
        self.config: dict = config

        self.episode = None
        self.use_trace = True

        self.writer = None

        if schedule is None:
            schedule = {}
        if sf_placement is None:
            sf_placement = {}
        # read dummy placement and schedule if specified
        # Flow forwarding schedule: dict
        self.schedule = schedule
        # Placement of SFs in each node: defaultdict(list)
        self.sf_placement = sf_placement

        # Update which sf is available at which node
        for node_id, placed_sf_list in sf_placement.items():
            for sf in placed_sf_list:
                self.network.nodes[node_id]['available_sf'][sf] = self.network.nodes[node_id]['available_sf'].get(sf, {
                    'load': 0.0,
                    'startup_time': 0.0,
                    'last_active': 0.0
                })

        # Node usage resource represent how much CPU resource have been used
        # It present each node resource 
        # node_usage_resource = [node1_resource, node2_resource, node3_resource, node4_resource]
        self.node_usage_resource = list()
        
        # Same for node source
        self.edge_used_data_rate = list()

        # Traffic like in Simulator State
        self.traffic = dict()

        # Total drop flow in 1 step
        self.drop_flow = list()

        # Latency for each SFC 
        # latency = [SFC1_latency, SFC2_latency, SFC3_ latency, SFC4_latency]
        self.latency = list()

        # Packet loss for each SFC
        # packet loss = [SFC1 packet loss, SFC2 packet loss, SFC3 packet loss, SFC4 packet loss]
        self.packet_loss = list()
        self.network_stats = dict()
        

    # string representation for logging
    #TODO: add diff params here
    def __str__(self):
        params_str = "Simulator parameters: \n"
        params_str += "inter_arr_mean: {}\n".format(self.inter_arr_mean)
        params_str += f"deterministic_arrival: {self.deterministic_arrival}\n"
        params_str += "flow_dr_mean: {}\n".format(self.flow_dr_mean)
        params_str += "flow_dr_stdv: {}\n".format(self.flow_dr_stdev)
        params_str += "flow_size_shape: {}\n".format(self.flow_size_shape)
        params_str += f"deterministic_size: {self.deterministic_size}\n"
        return params_str

    def get_init_state(self):
        """
        Return the init simulator state
        """
        # TODO:
        # Reset the whole metrics with 0 
        # Reset the params with it reset values
        self.reset_parameters()
        self.parse_network()
        simulator_state = SimulatorState(self.network_dict, self.sf_placement, self.sfc_list, self.sf_list, self.traffic, self.network_stats)
        return simulator_state

    def get_next_state(self, action):
        """
        Return the simulator state after apply action to the network.
        Write to metrics. 
        Input: simulator action  <dict>
        Output: simulator state
        """
        # self.writer.write_action_result(self.episode, self.env.now, action)
        # self.writer.write_schedule_table(self.params, self.env.now, action)

        # Get the new placement from the action passed by the RL agent
        # Modify and set the placement parameter of the instantiated simulator object.
        self.sf_placement = action.placement
        # Update which sf is available at which node
        for node_id, placed_sf_list in action.placement.items():
            available = {}
            # Keep only SFs which still process
            for sf, sf_data in self.network.nodes[node_id]['available_sf'].items():
                if sf_data['load'] != 0:
                    available[sf] = sf_data
            # Add all SFs which are in the placement
            for sf in placed_sf_list:
                if sf not in available.keys():
                    available[sf] = available.get(sf, {
                        'load': 0.0,
                        'last_active': self.env.now,
                        'startup_time': self.env.now
                    })
            self.network.nodes[node_id]['available_sf'] = available

        # Get the new schedule from the SimulatorAction
        # Set it in the params of the instantiated simulator object.
        self.schedule = action.scheduling
        
        #TODO: Load each network node with the schedule we want.
        # 
        # Create source list for each node
        # input : scheduling
        # output: source list
        # Generate .init file from source list
        # Run OMNET++
        # Update the metric based on the OMNET result
        # Update the metric based on flow result  
        self.parse_network()
        # self.network_metrics()

        # Create a new SimulatorState object to pass to the RL Agent
        simulator_state = SimulatorState(self.network_dict, self.sf_placement, self.sfc_list,
                                         self.sf_list, self.traffic, self.network_stats)
        return simulator_state

    def get_current_ingress_Traffic(self):
        """
        Return the load in each node
        """
        pass

    def network_metrics(self):
        """
        Processes the metrics and parses them in a format specified in the SimulatorState class.
        """
        stats = self.metrics.get_metrics()
        self.traffic = stats['run_total_requested_traffic']
        self.network_stats = {
            'processed_traffic': stats['run_total_processed_traffic'],
            'total_flows': stats['generated_flows'],
            'successful_flows': stats['processed_flows'],
            'dropped_flows': stats['dropped_flows'],
            'run_successful_flows': stats['run_processed_flows'],
            'run_dropped_flows': stats['run_dropped_flows'],
            'run_dropped_flows_per_node': stats['run_dropped_flows_per_node'],
            'in_network_flows': stats['total_active_flows'],
            'avg_end2end_delay': stats['avg_end2end_delay'],
            'run_avg_end2end_delay': stats['run_avg_end2end_delay'],
            'run_max_end2end_delay': stats['run_max_end2end_delay'],
            'run_avg_path_delay': stats['run_avg_path_delay'],
            'run_total_processed_traffic': stats['run_total_processed_traffic']
        }

    def get_current_ingress_traffic(self) :
        """
        Get current ingress traffic for the LSTM module
        Current limitation: works for 1 SFC and 1 ingress node
        return: -> float
        """
        # Get name of ingress SF from first SFC
        first_sfc = list(self.params.sfc_list.keys())[0]
        ingress_sf = self.params.sfc_list[first_sfc][0]
        ingress_node = self.params.ing_nodes[0][0]
        ingress_traffic = self.metrics.metrics['run_total_requested_traffic'][ingress_node][first_sfc][ingress_sf]
        return ingress_traffic

    def parse_network(self):
        """
        Converts the NetworkX network in the simulator to a dict in a format specified in the SimulatorState class.
        Return: -> dict
        """
        self.network_dict = {'nodes': [], 'edges': []}
        index = 0
        for node in self.network.nodes(data=True):
            node_cap = node[1]['cap']
            run_max_node_usage = self.node_usage_resource[index]
            index = index + 1
            # 'used_resources' here is the max usage for the run.
            self.network_dict['nodes'].append({'id': node[0], 'resource': node_cap,
                                               'used_resources': run_max_node_usage})
        for edge in self.network.edges(data=True):
            edge_src = edge[0]
            edge_dest = edge[1]
            edge_delay = edge[2]['delay']
            edge_dr = edge[2]['cap']
            # We use a fixed user data rate for the edges here as the functionality is not yet incorporated in the
            # simulator.
            # TODO: Implement used edge data rates in the simulator.
            edge_used_dr = 0
            self.network_dict['edges'].append({
                'src': edge_src,
                'dst': edge_dest,
                'delay': edge_delay,
                'data_rate': edge_dr,
                'used_data_rate': edge_used_dr
            })
    
    def parse_network_stats(self):
        """
        Convert all latency, flow and packet loss to network stats dict
        """
        self.network_stats['total_flows'] = self.total_flow
        self.network_stats['successful_flows'] = self.success_flow
        self.network_stats['dropped_flows'] = self.drop_flow
        self.network_stats['avg_end_2_end_delay'] = self.latency
        self.network_stats['packet_loss'] = self.packet_loss
        pass

    def reset_parameters(self):
        """
        Reset all the key state parameters:
        node_usage = [0, 0 ,0 ,0 ]
        """
        self.node_usage_resource = [0, 0, 0, 0]
        self.edge_used_data_rate = [0, 0, 0, 0]
        # logger.info(json.dumps(self.network_dict, indent=2))
        logger.info(json.dumps(self.sfc_list, indent=2))
        logger.info(json.dumps(self.sf_list, indent=2))
        # for node in self.network_dict['nodes']:
        #     for sfcs in self.sfc_list
        #     self.traffic[node['id']]
        #     logger.debug(node)
        self.total_flow = 0
        self.success_flow = 0
        self.drop_flow = 0
        self.latency = [0, 0, 0, 0]
        self.packet_loss = [0, 0, 0, 0]
        self.parse_network_stats()
        pass