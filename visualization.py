# %%

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

def get_plot(path: Path, label: str = None):
    label = label or 'Mean'
    with open(path, 'r') as f:
        d = np.loadtxt(f)
    d = d[:20]
    plt.plot(d)
    plt.xlabel("Iterations")
    plt.ylabel("Total Cumulative Waiting Time")
    plt.margins(0)
    # plt.hlines(np.mean(d), xmin=0, xmax =d.shape[0], label=label, linestyle='--')
    print('\nStats for', label)
    print('Minimum', np.min(d))
    print('Mean', np.mean(d))
    print('Median', np.median(d))
    return d

def main():
    pass
    # %%
    FIXED_CONTROL_PATH = os.path.join('Baseline_Fixed_Time_Control', 'total_fixed_data_mod.txt')
    ADAPT_CONTROL_PATH = os.path.join('Adaptive Pedestrian Signal', 'total_adapt_data_moderate.txt')
    RL_CONTROL_PATH = os.path.join('DeepQLearning', 'saved_models')
    E_GREEDY_PATH = os.path.join(RL_CONTROL_PATH, 'model_egreedy', 'plot_delay_data.txt')
    EXP_PATH = os.path.join(RL_CONTROL_PATH, 'model_exp', 'plot_delay_data.txt')
    SOFTMAX_PATH = os.path.join(RL_CONTROL_PATH, 'model_softmax', 'plot_delay_data.txt')

    # %%
    fixed = get_plot(FIXED_CONTROL_PATH, "Fixed")

    
    adapt = get_plot(ADAPT_CONTROL_PATH, "Adaptive")

    
    e_greedy = get_plot(E_GREEDY_PATH, "Epsilon-Greedy")
    exp_data = get_plot(EXP_PATH, "Exploration")
    softmax = get_plot(SOFTMAX_PATH, "Softmax")

    plt.plot(fixed, label="Fixed")
    plt.plot(adapt, label="Adaptive")

    plt.plot(e_greedy, label="Epsilon-Greedy")
    plt.plot(exp_data, label="Exploration")
    plt.plot(softmax, label="Softmax")
    plt.legend()
    plt.show()

# %%
# if __name__ == "__main__":
#     main()
