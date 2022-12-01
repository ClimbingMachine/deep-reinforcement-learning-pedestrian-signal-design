from training_simulation import Simulation
import numpy as np

class SimulationExploration(Simulation):
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_feats, num_actions, training_epochs):
        super().__init__(Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_feats, num_actions, training_epochs)
        self._n_count = np.ones((num_states, num_actions), dtype=int)
        self._t = 1

    def _get_state_idx(self, state):
        lane_group, lane_cell = state
        return 10 * lane_group + lane_cell

    def _choose_action(self, state, epsilon):
        """
        Choose action according to exploration function
        r+(s,a) = r(s,a) + B(N(s))
        B(N(s)) = sqrt(2 ln(t) / N(s))
        """
        q_vals = self._Model.predict_one(state)

        state_idx = self._get_state_idx(state)
        f = q_vals + np.random.random(q_vals.shape)
        f += np.sqrt(2 * np.log(self._t) / self._n_count[state_idx])
        
        action = np.argmax(f)

        self._n_count[state_idx, action] += 1
        self._t += 1
        
        return action