# %%

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

def get_plot(path: Path, label: str = None):
    label = label or 'Mean'
    with open(path, 'r') as f:
        d = np.loadtxt(f)
    d = d[:40]
    plt.plot(d)
    plt.xlabel("Iterations")
    plt.ylabel("Total Cumulative Waiting Time")
    plt.margins(0)
    # plt.hlines(np.mean(d), xmin=0, xmax =d.shape[0], label=label, linestyle='--')
    print('\n', label)
    print('Minimum', np.min(d))
    print('Mean', np.mean(d))
    print('Median', np.median(d))
    return d

def main():
    pass
    # %%
    FIXED_CONTROL_PATH = os.path.join('Baseline_Fixed_Time_Control', 'data', 'total_fixed_data_mod.txt')
    ADAPT_CONTROL_PATH = os.path.join('Adaptive Pedestrian Signal', 'data', 'total_adapt_data_moderate_5_6.txt')
    RL_CONTROL_PATH = os.path.join('DeepQLearning', 'saved_models')
    E_GREEDY_PATH = os.path.join(RL_CONTROL_PATH, 'model_egreedy', 'plot_delay_data.txt')
    UCB_PATH = os.path.join(RL_CONTROL_PATH, 'model_ucb', 'plot_delay_data.txt')
    SOFTMAX_PATH = os.path.join(RL_CONTROL_PATH, 'model_softmax', 'plot_delay_data.txt')
    TBL_E_GREEDY_PATH = os.path.join(RL_CONTROL_PATH, 'model_tbl_egreedy', 'plot_delay_data.txt')
    TBL_UCB_PATH = os.path.join(RL_CONTROL_PATH, 'model_tbl_ucb', 'plot_delay_data.txt')
    TBL_SOFTMAX_PATH = os.path.join(RL_CONTROL_PATH, 'model_tbl_softmax', 'plot_delay_data.txt')

    # %%
    fixed = get_plot(FIXED_CONTROL_PATH, "Fixed")
    adapt = get_plot(ADAPT_CONTROL_PATH, "Adaptive")
    
    e_greedy = get_plot(E_GREEDY_PATH, "DQN: Epsilon-Greedy")
    ucb_data = get_plot(UCB_PATH, "DQN: UCB")
    softmax = get_plot(SOFTMAX_PATH, "DQN: Softmax")

    tbl_e_greedy_data = get_plot(TBL_E_GREEDY_PATH, "Q-Table: Epsilon-Greedy")
    tbl_ucb_data = get_plot(TBL_UCB_PATH, "Q-Table: UCB")
    tbl_softmax_data = get_plot(TBL_SOFTMAX_PATH, "Q-Table: Softmax")

    combined_data = np.array([e_greedy, ucb_data, softmax, tbl_e_greedy_data, tbl_ucb_data, tbl_softmax_data])
    ylim = [np.min(combined_data), np.max(combined_data)]


    # %%
    plt.clf()
    plt.plot(fixed, label="Fixed", linestyle="dashed")
    plt.plot(adapt, label="Adaptive", linestyle="dashed")

    plt.plot(e_greedy, label="DQN: ε-Greedy")
    plt.plot(ucb_data, label="DQN: UCB")
    plt.plot(softmax, label="DQN: Softmax")

    plt.ylim(ylim)

    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Total Cumulative Waiting Time")
    plt.margins(0)
    plt.title('Control Strategies and Waiting Times - DQN')
    plt.show()

    # %%
    plt.clf()
    plt.plot(fixed, label="Fixed", linestyle="dashed")
    plt.plot(adapt, label="Adaptive", linestyle="dashed")

    plt.plot(tbl_e_greedy_data, label="Q-Table: ε-Greedy")
    plt.plot(tbl_ucb_data, label="Q-Table: UCB")
    plt.plot(tbl_softmax_data, label="Q-Table: Softmax")

    plt.ylim(ylim)

    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Total Cumulative Waiting Time")
    plt.margins(0)
    plt.title('Control Strategies and Waiting Times - Q-Table')
    plt.show()
# %%
# if __name__ == "__main__":
#     main()
