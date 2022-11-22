# A1

## Q1 Agent
Signal Controller

## Q2 Environment
SUMO

## Q3/4 MDP
MDP can be represented by the tuple $(S,A,R,T)$ - States, Actions, Reward, Transition Model

### State
$$
s \in S_{ped} \times S_{veh}\\
S_{ped} = S_{veh} = \{0, ..., 9\}
$$

A state is a tuple $(s_{ped}, s_{veh})$ which represents the number of peds
and vehicles respectively.
In training, one-hot encoding representation is used.

* Q: Why values $\geq 9$ lumped together?
* A: From real data, $s_{ped}, s_{veh} \geq 9$ is rare.

### Action
2 actions - {Pedestrian Green, Vehicle Green}

* Q: Can timing of light change be included in action?
* A: Yes. This will cause the action space to grow as well.

### Reward
Total Cummulative Wait Time (TCWT)

$$
TCWT = \sum p_{ped} + p_{veh}\\
R_t = TCWT_{t-1} - TCWT_t
$$

Reward is the decrease in TCWT between time steps.s

### Transition Model
Unknown, simulated by SUMO.

## Q5 Experiment Design
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
1. Fixed time control
2. Actuated pedestrian control

## Q6 Why DQN? 
Is DQN necessary given state representation? (no)

DQN is used when the state space is big, 
to generalize training on a small subset of the state space, to the entire space.

No, DQN is unnecessary, since there are only 100 possible states, with 2 possible actions.
This gives a total of 200 values in the Q-value table, 
corresponding to all state-action pairs.

Q-Learning can be used, and more actions can be added.

## Tasks
- [x] Setup SUMO
- [x] Briefly read `traci`
- [x] Read 2.2.3 Dissertation on Control Types
- [x] Implement Fixed Time Control
- [ ] Report Fixed Time Control results
- [ ] Replicate DQN result
- [ ] Increase flow settings
  - [ ] e.g. high pedestrian, high vehicle
- [ ] Report results of designs with figures

# A2

## Q1 Experience Replay

Used to stabilise training, with more efficient use of previous experience.
Reduces noise and variance in training, converges better.

## Q2 $\epsilon$-greedy Policy

Covers the exploration-exploitation trade-off in RL.
Choose the greedy action with probability $1-\epsilon$,
and a random action with probability $\epsilon$.
Greedy action is chosen in the final model.

$\epsilon\in[0,1]$, higher values correspond to more explorative behaviour, lower values correspond to more exploitative behaviour. Can be lowered over time like in $\epsilon=1/t$ to eventually be optimal in greedy behaviour.

## Q3 Other Policies

1. Use exploration function $r^+$ with bonus $\mathcal{B}$
    $$
    r^+(s, a) = r(s,a) + \mathcal{B}(N(s))
    $$ 

    and take 
    $$
    a=\argmax_a r^+(s,a)
    $$

    UCB sets
    $$
    \mathcal{B}(N(s))=\sqrt{\frac{2\ln n}{N(s)}}
    $$
1. Use softmax policy, given the Q-value function $Q_\theta(s,a)$
    $$
    \pi_\theta(s,a)=\frac{\exp Q_\theta(s,a)}{\sum_{a'}\exp Q_\theta(s,a')}
    $$
    Sample action $a\sim\pi_\theta(s,a)$.
