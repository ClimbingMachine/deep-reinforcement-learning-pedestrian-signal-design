from training_simulation import Simulation
import numpy as np

class SimulationTD(Simulation):
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        super().__init__(Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs)
        self._n_count = np.ones((num_states, num_actions), dtype=int)
        self._t = 1

    def _get_state(self):
        lane_group, lane_cell =  super()._get_state()
        return 10 * lane_group + lane_cell

    def _choose_action(self, state, epsilon):
        q_vals = self._Model.predict_one(state)
        f = q_vals + np.random.random(q_vals.shape)
        f += np.sqrt(2 * np.log(self._t) / self._n_count[state])
        
        action = np.argmax(f)

        self._n_count[state, action] += 1
        self._t += 1

        # print(self._n_count[state])
        # print(self._t)
        # print(q_vals)
        
        return action