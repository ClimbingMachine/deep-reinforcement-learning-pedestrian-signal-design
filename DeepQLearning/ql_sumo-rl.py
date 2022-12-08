import argparse
import os
import sys
from datetime import datetime

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from env_cfg import SumoCfgEnvironment
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
from generator import TrafficGenerator

OUT_FOLDER = os.path.join('outputs', 'simple_crosswalk')
SUMOCFG_FOLDER = os.path.join('sumo_config', 'simple_crosswalk')
CFG_FILE = os.path.join(SUMOCFG_FOLDER, 'run.sumocfg')
NET_FILE = os.path.join(SUMOCFG_FOLDER, 'pedcrossing.net.xml')
ROU_FILE = os.path.join(SUMOCFG_FOLDER, 'pedcrossing.rou.xml')
TRIP_FILE = os.path.join(SUMOCFG_FOLDER, 'pedcrossing.trip.xml')

if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Single-Intersection""")
    prs.add_argument("-route", dest="route", type=str, default=ROU_FILE, help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=30, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-r", dest="reward", type=str, default='wait', required=False, help="Reward function: [-r queue] for average queue reward or [-r wait] for waiting time reward.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=10, help="Number of runs.\n")
    args = prs.parse_args()
    experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = os.path.join(OUT_FOLDER, '{}_alpha{}_gamma{}_eps{}_decay{}_reward{}'.format(experiment_time, args.alpha, args.gamma, args.epsilon, args.decay, args.reward))

    env = SumoCfgEnvironment(
        net_file=NET_FILE,
        route_file=args.route,
        cfg_file=CFG_FILE,
        out_csv_name=out_csv,
        use_gui=args.gui
    )

    for run in range(args.runs):
        generator = TrafficGenerator(5, 6)
        generator.generate_tripfile(run)
        initial_states = env.reset()
        ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                                 state_space=env.observation_space,
                                 action_space=env.action_space,
                                 alpha=args.alpha,
                                 gamma=args.gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay)) for ts in env.ts_ids}

        done = {'__all__': False}
        infos = []
        if args.fixed:
            while not done['__all__']:
                _, _, done, _ = env.step({})
        else:
            while not done['__all__']:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, _ = env.step(action=actions)

                for agent_id in ql_agents.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
                    ql_agents[agent_id].save(os.path.join(OUT_FOLDER, f'q_table_{agent_id}'))
        df = env.save_csv(out_csv, run)
        env.close()

        print(f"{run = }: sum(TCWT) = {sum(df['system_total_waiting_time'].values)}")

