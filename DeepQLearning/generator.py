#!/usr/bin/env python
# coding: utf-8

# In[17]:

import os
import numpy as np
import math
import traci  # noqa
from sumolib import checkBinary  # noqa
import randomTrips  # noqa

# directory of this script
CONTROL_FOLDER = 'DeepQLearning'
DATA_FOLDER = os.path.join(CONTROL_FOLDER, 'Intersection')
# directory of sumocfg
SUMOCFG_FOLDER = os.path.join('sumo_config', 'simple_crosswalk')
# config files for sumo
NET_FILE = os.path.join(SUMOCFG_FOLDER, 'pedcrossing.net.xml')
OUTPUT_TRIP_FILE = os.path.join(DATA_FOLDER,'pedestrians.trip.xml')

# In[18]:


class TrafficGenerator:
    def __init__(self, n_peds_generated, n_period):
        self._n_peds_generated = n_peds_generated  # how many peds per episode
        self._n_period = n_period
        
    def generate_tripfile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        randomTrips.main(randomTrips.get_options([
        '--net-file', NET_FILE,
        '--output-trip-file', OUTPUT_TRIP_FILE,
        '--seed', str(seed),  # make runs reproducible
        '--pedestrians',
        '--prefix', 'ped',
        '--allow-fringe',
        # prevent trips that start and end on the same edge
        '--min-distance', '1',
        '--trip-attributes', 'departPos="random" arrivalPos="random"',
        '--binomial', str(self._n_peds_generated),
        '--period', str(self._n_period)]))
        


# In[ ]:




