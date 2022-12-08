#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import print_function
from argparse import ArgumentParser

import os
import sys


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

# directory of this script
CONTROL_FOLDER = 'Adaptive Pedestrian Signal'
DATA_FOLDER = os.path.join(CONTROL_FOLDER, 'data')
# directory of sumocfg
SUMOCFG_FOLDER = os.path.join('sumo_config', 'simple_crosswalk')
# config files for sumo
NET_FILE = os.path.join(SUMOCFG_FOLDER, 'pedcrossing.net.xml')
OUTPUT_TRIP_FILE = os.path.join(SUMOCFG_FOLDER,'pedestrians.trip.xml')

# minimum green time for the vehicles
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


def run():
    """execute the TraCI control loop"""
    # track the duration for which the green phase of the vehicles has been
    # active
    greenTimeSoFar = 0
    total_ped_time = 0
    total_veh_time = 0
    # whether the pedestrian button has been pressed
    activeRequest = False

    # main loop. do something every simulation step until no more vehicles are
    # loaded or running
    
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        total_ped_time += get_waiting_ped()
        total_veh_time += get_queue_length()
        # decide wether there is a waiting pedestrian and switch if the green
        # phase for the vehicles exceeds its minimum duration
        
        if not activeRequest:
            activeRequest = checkWaitingPersons()
        if traci.trafficlight.getPhase(TLSID) == VEHICLE_GREEN_PHASE:
            greenTimeSoFar += 1
            if greenTimeSoFar > MIN_GREEN_TIME:
                # check whether someone has pushed the button

                if activeRequest:
                    # switch to the next phase
                    traci.trafficlight.setPhase(
                        TLSID, VEHICLE_GREEN_PHASE + 1)
                    # reset state
                    activeRequest = False
                    greenTimeSoFar = 0

    sys.stdout.flush()
    traci.close()
    return([total_ped_time, total_veh_time])


# In[5]:


def get_queue_length():
    """
    Retrieve the number of cars with speed = 0 in every incoming lane
    """
    halt_EC = traci.edge.getLastStepHaltingNumber("EC")
    halt_WC = traci.edge.getLastStepHaltingNumber("WC")
    queue_length = halt_EC + halt_WC
    return queue_length


# In[6]:


def get_waiting_ped():
    """
    Retrieve the number of peds with speed = 0 in every incoming lane
    """
    numWaiting = 0
    for edge in WALKINGAREAS:
        peds = traci.edge.getLastStepPersonIDs(edge)
        for ped in peds:
            if (traci.person.getWaitingTime(ped) == 1 and
                traci.person.getNextEdge(ped) in CROSSINGS):
                numWaiting = traci.trafficlight.getServedPersonCount(TLSID, PEDESTRIAN_GREEN_PHASE)
    return numWaiting


# In[7]:


def checkWaitingPersons():
    """check whether a person has requested to cross the street"""
    # check both sides of the crossing
    for edge in WALKINGAREAS:
        peds = traci.edge.getLastStepPersonIDs(edge)
        # check who is waiting at the crossing
        # we assume that pedestrians push the button upon
        # standing still for 1s
        # print(peds)
        for ped in peds:
            if (traci.person.getWaitingTime(ped) == 1 and
                    traci.person.getNextEdge(ped) in CROSSINGS):
                
                numWaiting = traci.trafficlight.getServedPersonCount(TLSID, PEDESTRIAN_GREEN_PHASE)
                
                # print("%s: pedestrian %s pushes the button (waiting: %s)" %
                #      (traci.simulation.getTime(), ped, numWaiting))
            
                return True
    return False


def setup_traci(args, i=42):    
    # generate the pedestrians for this simulation
    randomTrips.main(randomTrips.get_options([
        '--net-file', NET_FILE,
        '--output-trip-file', OUTPUT_TRIP_FILE,
        '--seed', str(i),  # make runs reproducible
        '--pedestrians',
        '--prefix', 'ped',
        '--allow-fringe',
        # prevent trips that start and end on the same edge
        '--min-distance', "1",
        '--trip-attributes', 'departPos="random" arrivalPos="random"',
        '--binomial', str(args.binomial),
        '--period', str(args.period)]
    ))

# In[8]:

 
def run_cli(args):        
    # this is the main entry point of this script
    pedwaiting = []
    vehwaiting = []
    sumoBinary = checkBinary('sumo')

    for i in range(args.runs):        
        # generate the pedestrians for this simulation
        setup_traci(args, i)

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start([sumoBinary, "-c", os.path.join(SUMOCFG_FOLDER, f'run_{args.vehicle}_adapt.sumocfg')])
        [pedestrian_waiting_time, veh_waiting_time] = run()
        pedwaiting.append(pedestrian_waiting_time)
        vehwaiting.append(veh_waiting_time)
        
        print(f"Iteration {i}: {pedestrian_waiting_time = } and {veh_waiting_time = }")
        


    # In[9]:
    totalwaiting = [ped + veh for ped, veh in zip(pedwaiting, vehwaiting)] 

    CONFIG_NAME = f'adapt_data_{args.vehicle}_{args.binomial}_{args.period}'

    PED_FILE = os.path.join(DATA_FOLDER, f'ped_{CONFIG_NAME}.txt')
    VEH_FILE = os.path.join(DATA_FOLDER, f'veh_{CONFIG_NAME}.txt')
    TOT_FILE = os.path.join(DATA_FOLDER, f'tot_{CONFIG_NAME}.txt')

    with open(PED_FILE, "w") as file:
        for value in pedwaiting:
            file.write(f"{value}\n")
                    
    with open(VEH_FILE, "w") as file:
        for value in vehwaiting:
            file.write(f"{value}\n")
                    
    with open(TOT_FILE, "w") as file:
        for value in totalwaiting:
            file.write(f"{value}\n")


# In[ ]:


def run_gui(args):
    sumoBinary = checkBinary('sumo-gui')
        
    # generate the pedestrians for this simulation
    setup_traci(args)

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", os.path.join(SUMOCFG_FOLDER, f'run_{args.vehicle}_adapt.sumocfg')])
    run()

def main():
    parser = ArgumentParser()
    parser.add_argument('-b', '--binomial', type=int, default=5, help='max number of simultaneous arrivals')
    parser.add_argument('-p', '--period', type=float, default=6, help='inverse of expected arrival rate')
    parser.add_argument('-v', '--vehicle', type=str, default='mod', help='low/mod/high vehicle traffic')
    parser.add_argument('-r', '--runs', type=int, default=100, help='number of runs')
    parser.add_argument('-g', '--gui', action='store_true')
    args = parser.parse_args()

    if args.gui:
        run_gui(args)
    else:
        run_cli(args)

if __name__ == '__main__':
    main()


