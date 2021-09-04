#!/bin/bash

# use GNU parallel to run multiple repetitions and scenarios in parallel
# run from project root! (where Readme is)
parallel rlsp :::: scripts/agent_config_files.txt :::: scripts/network_files.txt :::: scripts/service_files.txt :::: scripts/config_files.txt :::: scripts/service_requirement.txt ::: "1" ::: "-t" ::: "2021-06-22_12-45-45_seed9834"