#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import print_function


# In[2]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import datetime
from shutil import copyfile
import traci


# In[3]:

from training_simulation import Simulation
from training_simulation_exp import SimulationExploration
from training_simulation_bm import SimulationSoftmax
from training_simulation_timing import SimulationTiming
from generator import TrafficGenerator
from memory import Memory
from td_learning import TDLearning
from utils import import_train_configuration, set_sumo, set_train_path
from visual import Visualization

CONTROL_FOLDER = 'DeepQLearning'
CONFIG_FILE = os.path.join(CONTROL_FOLDER, 'training_settings.ini')


# In[4]:


if __name__ == "__main__":

    config = import_train_configuration(config_file=CONFIG_FILE)
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(os.path.join(CONTROL_FOLDER, config['models_path_name']))

    Model = TDLearning(
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )

    Memory = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['n_peds_generated'], 
        config['n_period']
    )

    Visualization = Visualization(
        path, 
        dpi=96
    )
        
    Simulation = SimulationTiming(
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_feats'],
        config['num_actions'],
        config['training_epochs']
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = Simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    Model.save_model(path)

    copyfile(src=CONFIG_FILE, dst=os.path.join(path, 'training_settings.ini'))

    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')


# In[ ]:




