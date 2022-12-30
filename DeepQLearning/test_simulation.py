import traci

from training_simulation import Simulation

class SimulationTest(Simulation):
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_feats, num_actions, training_epochs):
        super().__init__(Model, None, TrafficGen, sumo_cmd, None, max_steps, green_duration, yellow_duration, num_states, num_feats, num_actions, None)

    def run(self, episode, epsilon=0):
        """execute the TraCI control loop"""

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_tripfile(seed=str(episode))
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._ped_waiting_times = {}  
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_action = -1

        # main loop. do something every simulation step until no more vehicles are
        # loaded or running
        
        while traci.simulation.getMinExpectedNumber() > 0:
            # get current state of the intersection
            current_state = self._get_state()

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon=0)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later
            old_action = action
            
        traci.close()