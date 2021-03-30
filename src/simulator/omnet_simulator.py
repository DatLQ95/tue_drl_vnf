import logging
import random
import time
import os
from shutil import copyfile
import coordsim.reader.reader as reader
import numpy
from spinterface import SimulatorAction, SimulatorInterface, SimulatorState
from coordsim.writer.writer import ResultWriter
from simulator.omnet_simulator_param import SimulatorParam
from simulator.omnet_metric import Metrics
from collections import defaultdict
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig()
logging.root.setLevel(logging.DEBUG)

class OmnetSimulator(SimulatorInterface):
    def __init__(self, network_file, service_functions_file, config_file, resource_functions_path="", test_mode=False,
                 test_dir=None):
        super().__init__(test_mode)
        # Number of time the simulator has run. Necessary to correctly calculate env run time of apply function
        self.network_file = network_file
        self.test_dir = test_dir
        # init network, sfc, sf, and config files
        self.network, self.ing_nodes, self.eg_nodes = reader.read_network(self.network_file)
        self.sfc_list = reader.get_sfc(service_functions_file)
        self.sf_list = self.get_sf(service_functions_file)
        self.config = reader.get_config(config_file)
        # Assume result path is the path where network file is in.
        self.result_base_path = os.path.dirname(self.network_file)
        if 'trace_path' in self.config:
            # Quick solution to copy trace file to same path for network file as provided by calling algo.
            trace_path = os.path.join(os.getcwd(), self.config['trace_path'])
            copyfile(trace_path, os.path.join(self.result_base_path, os.path.basename(trace_path)))

        self.prediction = False
        # Check if future ingress traffic setting is enabled
        if 'future_traffic' in self.config and self.config['future_traffic']:
            self.prediction = True

        write_schedule = False
        if 'write_schedule' in self.config and self.config['write_schedule']:
            write_schedule = True
        write_flow_actions = False
        if 'write_flow_actions' in self.config and self.config['write_flow_actions']:
            write_flow_actions = True
        # Create CSV writer
        self.writer = ResultWriter(self.test_mode, self.test_dir, write_schedule, write_flow_actions)
        self.episode = 0
        self.last_apply_time = None
        # Load trace file
        if 'trace_path' in self.config:
            trace_path = os.path.join(os.getcwd(), self.config['trace_path'])
            self.trace = reader.get_trace(trace_path)
        
        #TODO: 
        # Create a simulator runner, which take all the parameters in and hold them, process them
        # Interact and store the result in metrics.
        self.param = SimulatorParam(config=self.config, network=self.network, ing_nodes=self.ing_nodes, eg_nodes=self.eg_nodes, sfc_list=self.sfc_list, sf_list=self.sf_list)

    # def __del__(self):
    #     # write dropped flow locs to yaml
    #     self.writer.write_dropped_flow_locs(self.metrics.metrics['dropped_flows_locs'])

    def init(self, seed):
        # increment episode count
        self.episode += 1
        # reset network caps and available SFs:
        reader.reset_cap(self.network)
        # Initialize metrics, record start time
        self.start_time = time.time()

        self.seed = seed
        random.seed(self.seed)
        numpy.random.seed(self.seed)

        self.end_time = time.time()
        self.last_apply_time = time.time()
        # in the future

        #TODO: Process to get the init state of network and parse them to the state variables.
        # reset all the parameters here!!!
        simulator_state = self.param.get_init_state()
        
        return simulator_state

    def apply(self, actions):

        logger.debug(f"t={self.env.now}: {actions}")

        # calc runtime since last apply (or init): that's the algorithm's runtime without simulation
        alg_runtime = time.time() - self.last_apply_time
        self.writer.write_runtime(alg_runtime)
        self.last_apply_time = time.time()

        #TODO: From action, process to get the state of network and parse them to the state variables.

        simulator_state = self.param.get_next_state(action=actions)

        return simulator_state

    def get_active_ingress_nodes(self):
        """Return names of all ingress nodes that are currently active, ie, produce flows."""
        return None

    def get_sf(self, sf_file):
        """
        Get the list of SFs and their properties from the yaml data.
        """
        with open(sf_file) as yaml_stream:
            sf_data = yaml.load(yaml_stream, Loader=yaml.FullLoader)

        # Configurable default mean and stddev defaults
        default_processing_delay_mean = 1.0
        default_processing_delay_stdev = 1.0

        sf_list = defaultdict(None)
        for sf_name, sf_details in sf_data['sf_list'].items():
            sf_list[sf_name] = sf_details
            # Set defaults (currently processing delay mean and stdev)
            sf_list[sf_name]["processing_delay_mean"] = sf_list[sf_name].get("processing_delay_mean",default_processing_delay_mean)
            sf_list[sf_name]["processing_delay_stdev"] = sf_list[sf_name].get("processing_delay_stdev",default_processing_delay_stdev)
        return sf_list

# for debugging
if __name__ == "__main__":
    # run from project root for file paths to work
    # I changed triangle to have 2 ingress nodes for debugging
    AGENT_CONFIG = 'res/config/agent/sample_agent.yaml'
    NETWORK = 'res/networks/tue_network.graphml'
    SERVICE = 'res/service_functions/tue_abc.yaml'
    SIM_CONFIG = 'res/config/simulator/test.yaml'

    sim = OmnetSimulator(NETWORK, SERVICE, SIM_CONFIG)
    state = sim.init(seed=1234)
    logger.info(f"state :{state}")
    # dummy_action = SimulatorAction(placement={}, scheduling={})
    # FIXME: this currently breaks - negative flow counter?
    #  should be possible to have an empty action and just drop all flows!
    # state = sim.apply(dummy_action)
