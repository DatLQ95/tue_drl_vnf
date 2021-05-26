# -*- coding: utf-8 -*-

from gym.envs.registration import register

register(
    id='rlsp-env-v1',
    entry_point='rlsp.envs:GymEnv',
)

register(
    id='metro_network-env-v1',
    entry_point='rlsp.envs:MetroNetworkEnv',
)