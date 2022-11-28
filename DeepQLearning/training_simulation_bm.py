from training_simulation import Simulation
import numpy as np

class SimulationSoftmax(Simulation):
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        super().__init__(Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs)

    def _choose_action(self, state, epsilon):
        """
        Choose action according to softmax (a.k.a. Boltzmann) policy
        """
        output = self._Model.predict_one(state) # the best action given the current state
        output = output.reshape(-1)
        e_x = np.exp(output - np.max(output))
        dsb = e_x / e_x.sum(axis=0)
        idx = np.arange(output.shape[0])
        act = np.random.choice(idx, 1, p=dsb)
        return act