## Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium

### Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.


### Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

### Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

### Environment Description
FrozenLake-v1 is a 4×4 grid-world environment. The agent starts from the starting tile and must reach the goal tile while avoiding holes. The agent can move Left, Down, Right, or Up. A reward of 1 is given when the agent reaches the goal, while other moves generally provide a reward of 0.

### Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

### Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$


### Algorithm
1. Initialize the Q-table with zeros.
2. Initialize the exploration rate ε.
3. Generate a complete episode using an ε-greedy policy.
4. Calculate the return G for each state-action pair by processing the episode backwards.
5. Update the Q-value using the incremental Monte Carlo update rule.
6. Store the total reward obtained in the episode.
7. Reduce ε gradually to decrease exploration and increase exploitation.
8. Repeat the process for the specified number of episodes.
9. Extract the greedy policy by selecting the action with the highest Q-value for each state.
10. Display the final Q-table, state-value function, learned policy, average reward, and learning curve.

### Python Program
#### Monte Carlo Control
```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create Environment
# -------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
n_states = env.observation_space.n
n_actions = env.action_space.n

num_episodes = 20000
gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))
episode_rewards = []


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state])

# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------

def generate_episode(epsilon):
    """
    Generates one episode using the current epsilon-greedy policy.
    Returns a list of (state, action, reward).
    """

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):
        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode

# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start

for episode_num in range(num_episodes):

    episode = generate_episode(epsilon)

    G = 0

    for state, action, reward in reversed(episode):

        G = reward + gamma * G

        Q[state, action] += alpha * (G - Q[state, action])

    episode_rewards.append(sum(x[2] for x in episode))

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------

optimal_policy = np.argmax(Q, axis=1)
state_values = np.max(Q, axis=1)

# -------------------------------------------------
# Display Results
# -------------------------------------------------

def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)
    print("Name: NITHYA D")
    print("Register Number: 212223240110")
    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(optimal_policy)

success_rate = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", success_rate)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500
moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Monte Carlo Control Learning Curve")
plt.grid(True)
plt.show()

env.close()
```

### Output
#### For Episodes=20000
<img width="377" height="545" alt="image" src="https://github.com/user-attachments/assets/9f474802-0f67-4e6f-896b-6ea9a9e4a097" />
<img width="671" height="430" alt="image" src="https://github.com/user-attachments/assets/e2867433-85fc-4177-a115-46329aae48b2" />


#### For Episodes=4000
<img width="362" height="540" alt="image" src="https://github.com/user-attachments/assets/e3eb521b-bd7f-429a-9f33-6407b2a075e5" />
<img width="667" height="431" alt="image" src="https://github.com/user-attachments/assets/71d78e59-f25e-4e23-a101-9fa70dd3f2d1" />

### Result
The On-Policy Monte Carlo Control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned the action-value function Q(s,a) from complete episodes using an epsilon-greedy policy. The learned Q-values were used to obtain an improved policy for reaching the goal while avoiding the holes.

### Inference 
The performance of Monte Carlo Control is affected by the values of the hyperparameters. Increasing α (learning rate) makes the Q-values change faster, while a smaller α makes learning slower but more stable. Increasing γ (discount factor) makes the agent give more importance to future rewards. A higher ε increases exploration and helps discover different actions, while a lower ε increases exploitation of the learned policy. Increasing the number of episodes generally improves the learned policy because the agent gets more experience. Therefore, suitable values of α, γ, ε, and the number of episodes are important for achieving better learning and a higher average reward.
