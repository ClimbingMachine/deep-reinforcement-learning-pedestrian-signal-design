from training_simulation import Simulation

class SimulationTD(Simulation):
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        super().__init__(Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs)

    def _get_state(self):
        lane_group, lane_cell =  super()._get_state()
        return 10 * lane_group + lane_cell