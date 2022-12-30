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

def plot_data(data, ylim):
    plt.clf()

    for d, label in data:
        linestyle = "-" 
        if "fixed" in label.lower() or "adaptive" in label.lower():
            linestyle = "dashed"
        plt.plot(d, label=label, linestyle=linestyle)
        

    plt.ylim(ylim)

    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Total Cumulative Waiting Time")
    plt.margins(0)
    plt.title('Control Strategies and Waiting Times')
    plt.show()


def main():
    pass
    # %%
    FIXED_CONTROL_PATH = os.path.join('Baseline_Fixed_Time_Control', 'data', 'total_fixed_data_mod_5_6.txt')
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
    data_to_plot = [(fixed, "Fixed")]
    plot_data(data_to_plot, ylim)

    # %%
    data_to_plot.append((adapt, "Adaptive"))
    plot_data(data_to_plot, ylim)

    # %%
    dqn_data_to_plot = data_to_plot.copy()
    dqn_data_to_plot.append((e_greedy, "DQN: Epsilon-Greedy"))
    plot_data(dqn_data_to_plot, ylim)

    # %%
    dqn_data_to_plot.extend([
        (ucb_data, "DQN: UCB"),
        (softmax, "DQN: Softmax")
    ])
    plot_data(dqn_data_to_plot, ylim)

    # %%
    q_tbl_data_to_plot = data_to_plot.copy()
    q_tbl_data_to_plot.extend([
        (tbl_e_greedy_data, "Q-Table: Epsilon-Greedy"), 
        (tbl_ucb_data, "Q-Table: UCB"),
        (tbl_softmax_data, "Q-Table: Softmax")
    ])
    plot_data(q_tbl_data_to_plot, ylim)

# %%
# if __name__ == "__main__":
#     main()
