#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess


# In[2]:


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci  # noqa
from sumolib import checkBinary  # noqa
import randomTrips  # noqa


# In[3]:


MIN_GREEN_TIME = 25
# the first phase in tls plan. see 'pedcrossing.tll.xml'
VEHICLE_GREEN_PHASE = 0
PEDESTRIAN_GREEN_PHASE = 2
# the id of the traffic light (there is only one). This is identical to the
# id of the controlled intersection (by default)
TLSID = 'C'

# pedestrian edges at the controlled intersection
WALKINGAREAS = [':C_w0', ':C_w1']
CROSSINGS = [':C_c0']


# In[4]:


def collect_waiting_times():
    """
    Retrieve the waiting time of every car in the incoming roads
    """
    # incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
    incoming_roads = ["EC", "WC"]
    car_list = traci.vehicle.getIDList()
    waiting_times = {}
    for car_id in car_list:
        wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
        road_id = traci.vehicle.getRoadID(car_id)   # get the road id where the car is located
        if road_id in incoming_roads:               # consider only the waiting times of cars in incoming roads
            waiting_times[car_id] = wait_time
        else:
            if car_id in waiting_times:       # a car that was tracked has cleared the intersection
                del waiting_times[car_id] 
    
    total_waiting_time = sum(waiting_times.values())
    return total_waiting_time


# In[5]:


def run():
    """execute the TraCI control loop"""
    # main loop. do something every simulation step until no more vehicles are
    # loaded or running
    total_ped_time = 0
    total_veh_time = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        total_ped_time += get_waiting_ped()
        total_veh_time += get_queue_length()
        # decide wether there is a waiting pedestrian and switch if the green
        # phase for the vehicles exceeds its minimum duration

    sys.stdout.flush()
    traci.close()
    return([total_ped_time, total_veh_time])
        # phase for the vehicles exceeds its minimum duration


# In[6]:


def get_queue_length():
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_EC = traci.edge.getLastStepHaltingNumber("EC")
        halt_WC = traci.edge.getLastStepHaltingNumber("WC")
        queue_length = halt_EC + halt_WC
        return queue_length


# In[7]:


def get_waiting_ped():
        """
        Retrieve the number of peds with speed = 0 in every incoming lane
        """
        numWaiting = 0
        for edge in WALKINGAREAS:
            peds = traci.edge.getLastStepPersonIDs(edge)
            for ped in peds:
                if (traci.person.getWaitingTime(ped) > 0 and
                    traci.person.getNextEdge(ped) in CROSSINGS):
                    numWaiting = traci.trafficlight.getServedPersonCount(TLSID, PEDESTRIAN_GREEN_PHASE)
        return numWaiting


# In[8]:


# this is the main entry point of this script
pedwaiting = []
vehwaiting = []
if __name__ == "__main__":
    
    for i in range(1000):
        sumoBinary = checkBinary('sumo')
        net = 'pedcrossing.net.xml'
        
        # generate the pedestrians for this simulation
        randomTrips.main(randomTrips.get_options([
        '--net-file', net,
        '--output-trip-file', 'pedestrians.trip.xml',
        '--seed', str(i),  # make runs reproducible
        '--pedestrians',
        '--prefix', 'ped',
        '--allow-fringe',
        # prevent trips that start and end on the same edge
        '--min-distance', "1",
        '--trip-attributes', 'departPos="random" arrivalPos="random"',
        '--binomial', '5',
        '--period', '6']))

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start([sumoBinary, "-c", os.path.join('run.sumocfg')])
        [pedestrian_waiting, veh_waiting] = run()
        pedwaiting.append(pedestrian_waiting)
        vehwaiting.append(veh_waiting)
        
        print("Iteration %s: pedestrian waiting time- %s and vehicle waiting time - %s)" %
                  (i, pedestrian_waiting, veh_waiting))
        


# In[9]:


with open('ped_fixed_data_mod.txt', "w") as file:
            for value in pedwaiting:
                    file.write("%s\n" % value)
                
with open('veh_fixed_data_mod.txt', "w") as file:
            for value in vehwaiting:
                    file.write("%s\n" % value)


# In[10]:


totalwaiting = [pedwaiting[i] + vehwaiting[i] for i in range(len(vehwaiting))] 
with open('total_fixed_data_mod.txt', "w") as file:
            for value in totalwaiting:
                    file.write("%s\n" % value)


# In[11]:


import matplotlib.pyplot as plt
import numpy as np


# In[12]:


np.mean(vehwaiting)


# In[ ]:





# In[ ]:




