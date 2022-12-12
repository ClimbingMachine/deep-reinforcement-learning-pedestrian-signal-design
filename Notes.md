# RL Internship

This document details the notes on questions answered and tasks completed.

- [RL Internship](#rl-internship)
  - [A1](#a1)
    - [Q1 Agent](#q1-agent)
    - [Q2 Environment](#q2-environment)
    - [Q3/4 MDP](#q34-mdp)
      - [State](#state)
      - [Action](#action)
      - [Reward](#reward)
      - [Transition Model](#transition-model)
    - [Q5 Experiment Design](#q5-experiment-design)
    - [Q6 Why DQN?](#q6-why-dqn)
    - [Tasks](#tasks)
    - [Installing CUDA](#installing-cuda)
  - [A2](#a2)
    - [Q1 Experience Replay](#q1-experience-replay)
    - [Q2 Epsilon-Greedy Policy](#q2-epsilon-greedy-policy)
    - [Q3 Other Policies](#q3-other-policies)
    - [Exploration Function](#exploration-function)
      - [Softmax Policy](#softmax-policy)
    - [Q4 Pedestrian/Vehicle Arrival](#q4-pedestrianvehicle-arrival)
    - [Q5 LQR or Deep RL?](#q5-lqr-or-deep-rl)
      - [Backwards recursion](#backwards-recursion)
      - [Forward recursion](#forward-recursion)
    - [Q6 Optimizers](#q6-optimizers)
    - [Tasks](#tasks-1)
    - [Misc](#misc)
  - [Policy Results](#policy-results)
  - [Future Directions](#future-directions)
    - [SUMO-RL](#sumo-rl)

## A1

First set of questions.

![RL](diagrams/RL_framework.png)

### Q1 Agent
Signal Controller

### Q2 Environment
SUMO

### Q3/4 MDP
MDP can be represented by the tuple $(S,A,R,T)$ - States, Actions, Reward, Transition Model.
Markov assumption is that the next state only depends on the current state and action.

#### State
$$
s \in S_{ped} \times S_{veh}
$$

$$
S_{ped} = S_{veh} = \{0, ..., 9\}
$$

A state is a tuple $(s_{ped}, s_{veh})$ which represents the number of peds
and vehicles respectively.
In training, one-hot encoding representation is used.

* Q: Why values $\geq 9$ lumped together?
* A: From real data, $s_{ped}, s_{veh} \geq 9$ is rare.

#### Action
2 actions - {Pedestrian Green, Vehicle Green}

* Q: Can timing of light change be included in action?
* A: Yes. This will cause the action space to grow as well.

#### Reward
Total Cummulative Wait Time (TCWT)

$$
TCWT = \sum p_{ped} + p_{veh}
$$

$$
R_t = TCWT_{t-1} - TCWT_t
$$

* $p_{ped}$: Number of waiting pedestrians
* $p_{veh}$: Number of vehicles in queue

Reward is the decrease in TCWT between time steps.
In other words, less delay means a greater reward.

#### Transition Model
Unknown, simulated by SUMO.

### Q5 Experiment Design
Hyperparameters

$$
\alpha = 0.001, \gamma = 0.95, E = 1000
$$

* Q: Were the hyperparameters tuned?
* A: Yes

Testing with 2 different environment settings
- Moderate pedestrian, low vehicle
- Moderate pedestrian, moderate vehicle

* Q: Why these settings? What about other settings?
* A: Other settings not tested yet, can be explored.

Compare with baseline controls
1. Fixed time control - Cycle between vehicle green (25s), yellow (5s), and pedestrian green (30s)
1. Actuated pedestrian control

![adaptive](diagrams/adaptive_flow.png)

### Q6 Why DQN? 

Q-value function maps the state-action pair to a value, and can be described as a "reward-to-go".

Is a Deep Q-Network (DQN) necessary given state representation? (no)

DQN is used when the state space is big, 
to generalize training on a small subset of the state space, to the entire space.

No, DQN is unnecessary, since there are only 100 possible states, with 2 possible actions.
This gives a total of 200 values in the Q-value table, 
corresponding to all state-action pairs.

Q-Learning can be used, and more actions can be added.

### Tasks
- [x] Setup SUMO
- [x] Briefly read `traci`
- [x] Read 2.2.3 Dissertation on Control Types
- [x] Implement Fixed Time Control
- [x] Report Fixed Time Control results
- [x] Replicate DQN result
  - [x] Implement GPU with CUDA
- [x] Increase flow settings
  - [x] e.g. high pedestrian, high vehicle
- [x] Report results of designs with figures

* Q: How is the vehicle arrival rate configured?
* A: In `pedcrossing.rou.xml`.

### Installing CUDA

Prerequisite: `conda`, NVIDIA GPU

Follow [installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

1. Create conda environment.
    ```bash
    conda create --name sumo python=3.9
    conda activate sumo
    ```
1. Install `cudatoolkit`
    ```bash
    conda install -c conda-forge cudatoolkit cudnn
    ```
1. Install `tensorflow`
    ```bash
    pip install tensorflow tensorflow-gpu
    ```
    
## A2

### Q1 Experience Replay

Used to stabilise training, with more efficient use of previous experience.
Reduces noise and variance in training, converges better.

### Q2 Epsilon-Greedy Policy

Covers the exploration-exploitation trade-off in RL.
Choose the greedy action with probability $1-\epsilon$,
and a random action with probability $\epsilon$.
Greedy action is chosen in the final model.

$\epsilon\in[0,1]$, higher values correspond to more explorative behaviour, lower values correspond to more exploitative behaviour. Can be lowered over time like in $\epsilon=1/t$ to eventually be optimal in greedy behaviour.

In the paper, $\epsilon$ is linearly annealed over time as follows.

$$
\epsilon = 1 - \frac{e}{E},
$$

where e is the current episode, and E is the total number of episodes trained.

### Q3 Other Policies

### Exploration Function
Use exploration function $r^+$ with bonus $\mathcal{B}$
where $N(s)$ is the number of times state s is visited in the episode.

$$
r^+(s, a) = r(s,a) + \mathcal{B}(N(s))
$$ 

and take 

$$
a=\arg\max_a r^+(s,a)
$$

UCB sets

$$
\mathcal{B}(N(s))=\sqrt{\frac{2\ln n}{N(s)}}
$$

#### Softmax Policy
Use softmax policy, given the Q-value function $Q_\theta(s,a)$

$$
\pi_\theta(s,a)=\frac{\exp Q_\theta(s,a)}{\sum_{a'}\exp Q_\theta(s,a')}
$$

Sample action $a\sim\pi_\theta(s,a)$.

### Q4 Pedestrian/Vehicle Arrival

Number of pedestrians at each time step modelled as binomial distribution, 
with a maximum number n, and probability p of each pedestrian appearing.

Number of vehicles at each time step modelled as Poisson distribution, 
with arrival rate $\lambda$.

Moderate pedestrians setting has $n=5,p=1/6$.
Moderate vehicles setting has $\lambda=0.35/s$

### Q5 LQR or Deep RL?

LQR is model-based RL. 
It requires the dynamics of the environment to be known.

Deep RL is typically model-free, where the transition model of the environment is unknown.

#### Backwards recursion

Set $V_{T+1}=0, v_{T+1}=0$.

For $t=T..1$

$$
\begin{align}
Q_t &= C_t + F_t^T V_{t+1} F_t\\  
q_t &= F_t^T V_{t+1} f_t + F_t^T v_{t+1}\\  
K_t &= -Q_{u_t, u_t}^{-1} Q_{u_t, x_t}\\
k_t &= -Q_{u_t, u_t}^{-1} q_{u_t}\\  
V_t &= Q_{x_t, x_t} + Q_{x_t, u_t} K_t + K_t^T Q_{u_t, x_t} + K_t^T Q_{u_t, u_t} K_t\\  
v_t &= q_{x_t} + Q_{x_t, u_t} k_t + K_t^T Q_{u_t} + K_t^T Q_{u_t, u_t} k_t
\end{align}
$$


#### Forward recursion

For $t=1..T$

$$
\begin{align}
u_t &= K_t x_t + k_t\\  
x_{t+1} &= f(x_t, u_t)
\end{align}
$$

### Q6 Optimizers

Optimizers and their hyperparameters
* SGD - Learning Rate, Momentum, Nesterov
* Adam - Learning Rate, $\beta_1, \beta_2$

Default SGD sets `momentum=0, nesterov=False`.
AdaGrad is Adam with $\beta_1=\beta_2=0$.

### Tasks

- [x] Replicate results & figures
- [x] Smart Control Strategy for Traffic Signal
- [x] Prepare 20 slides - <20 words per slide
  - [x] Problem statement
- [x] Future Directionss
- [x] LQR Analytical Solution

### Misc

* [nuPlan Challenge](https://www.nuscenes.org/nuplan#challenge)
* [Jieping Ye](https://scholar.google.com/citations?user=T9AzhwcAAAAJ&hl=en) - Ride-Hailing RL
* [Bayesian Data Analyis](http://www.stat.columbia.edu/~gelman/book/)

## Policy Results

The plot below shows the TCWT over iterations from Fixed, Adaptive, and RL-trained Controls. Both tabular and Deep RL were investigated, and made use of the 3 types of policy choices in training.

![delay_dqn](diagrams/combined_TCWT_DQN.png)

![delay_tbl](diagrams/combined_TCWT_Qtable.png)


It can be seen that using the softmax and exploration policy choices 
converge more quickly than $\epsilon$-greedy.
This could be attributed to the linear annealing being done on $\epsilon$
slowing convergence to the more optimal greedy policy.

Q-Table also performs much faster than DQN, taking 0.06s per iteration compared to 15s. 
This is as expected, since the neural network takes computation time, 
as opposed to the TD equation used for tabular learning.

Although tabular RL is much faster in this simple scenario, 
it is not as scalable as Deep RL.
With a significantly larger state space (e.g. modeling an entire city),
the deep neural network can be a better option to model the Q-value function.
Policy approximating algorithms, as opposed to the value approximating DQN used here, 
can also be considered.

## Future Directions

A few ideas could be used in a future continuation of this project.

| Property      | Description |
| ----------- | ----------- |
| Complex Crosswalks      | Implementing more sophisticated crosswalks       |
| State Space   | Using a more informative state space e.g. types of pedestrians        |
| Action Space   | Using more actions for the different traffic signals        |
| Reward   | Expressing other desirable properties in the reward system        |
| Multi-Agent   | Complementing complex crosswalks by treating the system as having multiple traffic controllers as agents        |

### SUMO-RL

Projects can also be built open the open-source [SUMO-RL](https://github.com/LucasAlegre/sumo-rl).
The current implementation does not include pedestrians which can be added.