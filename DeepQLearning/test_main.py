from test_simulation import SimulationTest
from generator import TrafficGenerator
from td_learning import TDLearningTest
from utils import import_test_configuration, set_sumo, set_test_path
import os

CONTROL_FOLDER = 'DeepQLearning'
CONFIG_FILE = os.path.join(CONTROL_FOLDER, 'test_settings.ini')

def main():
    config = import_test_configuration(config_file=CONFIG_FILE)
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_folder_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    Model = TDLearningTest(
        input_dim=config['num_states'], 
        model_path=model_folder_path
    )

    TrafficGen = TrafficGenerator(
        config['n_peds_generated'], 
        config['n_period']
    )
        
    Simulation = SimulationTest(
        Model,
        None,
        TrafficGen,
        sumo_cmd,
        0,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_feats'],
        config['num_actions'],
        None
    )

    Simulation.run(config['episode_seed'])

if __name__ == '__main__':
    main()