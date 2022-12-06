"""
Simulation class to include bigger action space including timing of lights
"""
import timeit
import traci

from training_simulation import PHASE_VEH_GREEN
from training_simulation_bm import SimulationSoftmax
from training_simulation_exp import SimulationExploration

GREEN_STEP = 10

class SimulationTiming(SimulationSoftmax):
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_feats, num_actions, training_epochs):
        super().__init__(Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_feats, num_actions, training_epochs)

    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

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
        self._passed_min_time = False
        old_total_wait = 0
        old_state = -1
        old_action = -1

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times() + self._collect_ped_waiting_times()
            reward = old_total_wait - current_total_wait

            # saving the data into the memory
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection
            action, time_id = self._choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)
                self._passed_min_time = False

            # execute the phase selected before
            self._set_green_phase(action)
            green_sim_time = self._green_duration
            if action == PHASE_VEH_GREEN:
                green_sim_time += (time_id - 1) * GREEN_STEP
            # print(green_sim_time)
            self._simulate(green_sim_time)
            self._passed_min_time = True

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 3)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 3)

        return simulation_time, training_time

    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        action = super()._choose_action(state, epsilon)

        action_id = action // (self._num_actions // 2)
        time_id = action % (self._num_actions // 2)
        return action_id, time_id