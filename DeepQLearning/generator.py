#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import math
import traci  # noqa
from sumolib import checkBinary  # noqa
import randomTrips  # noqa


# In[18]:


class TrafficGenerator:
    def __init__(self, n_peds_generated, n_period):
        self._n_peds_generated = n_peds_generated  # how many peds per episode
        self._n_period = n_period
        
    def generate_tripfile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        net = "./Intersection/pedcrossing.net.xml"
        randomTrips.main(randomTrips.get_options([
        '--net-file', net,
        '--output-trip-file', 'Intersection\pedestrians.trip.xml',
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




