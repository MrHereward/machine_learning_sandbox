from collections import defaultdict

import gym
import matplotlib.pyplot as plt
import torch

# def run_episode(env, policy):
#     state, _ = env.reset()
#     total_reward = 0
#     is_done = False
#     while not is_done:
#         action = policy[state].item()
#         state, reward, is_done, info, _ = env.step(action)
#         total_reward += reward
#         if is_done:
#             break
#     return total_reward

def value_iteration(env, gamma, threshold):
    n_state = env.observation_space.n
    n_action = env.action_space.n
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.empty(n_state)
        for state in range(n_state):
            v_actions = torch.zeros(n_action)
            for action in range(n_action):
                for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                    v_actions[action] += trans_prob * (reward + gamma * V[new_state])
            V_temp[state] = torch.max(v_actions)
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if max_delta <= threshold:
            break
    return V

def extract_optimal_policy(env, V_optimal, gamma):
    n_state = env.observation_space.n
    n_action = env.action_space.n
    optimal_policy = torch.zeros(n_state)
    for state in range(n_state):
        v_actions = torch.zeros(n_action)
        for action in range(n_action):
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                v_actions[action] += trans_prob * (reward + gamma * V_optimal[new_state])
        optimal_policy[state] = torch.argmax(v_actions)
    return optimal_policy

def policy_evaluation(env, policy, gamma, threshold):
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(n_state):
            action = policy[state].item()
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                V_temp[state] += trans_prob * (reward + gamma * V[new_state])
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if max_delta <= threshold:
            break
    return V

def policy_improvement(env, V, gamma):
    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.zeros(n_state)
    for state in range(n_state):
        v_actions = torch.zeros(n_action)
        for action in range(n_action):
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                v_actions[action] += trans_prob * (reward + gamma * V[new_state])
        policy[state] = torch.argmax(v_actions)
    return policy

def policy_iteration(env, gamma, threshold):
    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.randint(high=n_action, size=(n_state,)).float()
    while True:
        V = policy_evaluation(env, policy, gamma, threshold)
        policy_improved = policy_improvement(env, V, gamma)
        if torch.equal(policy_improved, policy):
            return V, policy_improved
        policy = policy_improved

# def run_episode(env, hold_score):
#     state, _ = env.reset()
#     rewards = []
#     states = [state]
#     while True:
#         action = 1 if state[0] < hold_score else 0
#         state, reward, is_done, info, _ = env.step(action)
#         states.append(state)
#         rewards.append(reward)
#         if is_done:
#             break
#     return states, rewards

def mc_prediction_first_visit(env, hold_score, gamma, n_episode):
    V = defaultdict(float)
    N = defaultdict(int)
    for episode in range(n_episode):
        states_t, rewards_t = run_episode(env, hold_score)
        return_t = 0
        G = {}
        for state_t, reward_t in zip(states_t[1::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[state_t] = return_t
        for state, return_t in G.items():
            if state[0] <= 21:
                V[state] += return_t
                N[state] += 1
    for state in V:
        V[state] = V[state] / N[state]
    return V

def run_episode(env, Q, n_action):
    state, _ = env.reset()
    rewards = []
    actions = []
    states = []
    action = torch.randint(0, n_action, [1]).item()
    while True:
        actions.append(action)
        states.append(state)
        state, reward, is_done, info, _ = env.step(action)
        rewards.append(reward)
        if is_done:
            break
        action = torch.argmax(Q[state]).item()
    return states, actions, rewards

def mc_control_on_policy(env, gamma, n_episode):
    G_sum = defaultdict(float)
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.empty(env.action_space.n))
    for episode in range(n_episode):
        states_t, actions_t, rewards_t = run_episode(env, Q, env.action_space.n)
        return_t = 0
        G = {}
        for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[(state_t, action_t)] = return_t
        for state_action, return_t in G.items():
            state, action = state_action
            if state[0] <= 21:
                G_sum[state_action] += return_t
                N[state_action] += 1
                Q[state][action] = G_sum[state_action] / N[state_action]
    policy = {}
    for state, actions, in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy

def simulate_hold_episode(env, hold_score):
    state, _ = env.reset()
    while True:
        action = 1 if state[0] < hold_score else 0
        state, reward, is_done, _, _ = env.step(action)
        if is_done:
            return reward

def simulate_episode(env, policy):
    state, _ = env.reset()
    while True:
        action = policy[state]
        state, reward, is_done, _, _ = env.step(action)
        if is_done:
            return reward

def get_epsilon_greedy_policy(n_action, epsilon):
    def policy_function(state, Q):
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function

def q_learning(env, gamma, n_episode, alpha, length_episode, total_reward_episode):
    n_action = env.action_space.n
    Q = defaultdict(lambda: torch.zeros(n_action))
    for episode in range(n_episode):
        state, _ = env.reset()
        is_done = False
        while not is_done:
            action = epsilon_greedy_policy(state, Q)
            next_state, reward, is_done, info, _ = env.step(action)
            delta = reward + gamma * torch.max(Q[next_state]) - Q[state][action]
            Q[state][action] += alpha * delta
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            if is_done:
                break;
            state = next_state
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy

# Frozen Lake
# env = gym.make('FrozenLake-v1')
# n_state = env.observation_space.n
# print(n_state)
# n_action = env.action_space.n
# print(n_action)
# env.reset()
# env.render()
# new_state, reward, is_done, info = env.step(2)
# print(new_state)
# print(reward)
# print(is_done)
# print(info)
# env.render()

# n_episode = 100
# total_rewards = []
# for episode in range(n_episode):
#     random_policy = torch.randint(high=n_action, size=(n_state,))
#     total_reward = run_episode(env, random_policy)
#     total_rewards.append(total_reward)
# print(f'Średnia sumaryczna nagroda w losowej polityce: {sum(total_rewards)/n_episode}')
# gamma = 0.99
# threshold = 0.0001
# V_optimal = value_iteration(env, gamma, threshold)
# print('Optymalne wartości:\n', V_optimal)
# optimal_policy = extract_optimal_policy(env, V_optimal, gamma)
# print('Optymalna polityka:\n', optimal_policy)
# n_episode = 100
# total_rewards = []
# for episode in range(n_episode):
#     total_reward = run_episode(env, optimal_policy)
#     total_rewards.append(total_reward)
# print(f'Średnia sumaryczna nagroda po zastosowaniu optymalnej polityki: {sum(total_rewards)/n_episode}')
# V_optimal, optimal_policy = policy_iteration(env, gamma, threshold)
# print('Optymalne wartości:\n', V_optimal)
# print('Optymalna polityka:\n', optimal_policy)
# Black Jack
# env = gym.make('Blackjack-v1')
# hold_score = 18
# gamma = 1
# n_episode = 500000
# # value = mc_prediction_first_visit(env, hold_score, gamma, n_episode)
# # print(value)
# # print('Liczba stanów: ', len(value))
# optimal_Q, optimal_policy = mc_control_on_policy(env, gamma, n_episode)
# print(optimal_policy)
# n_episode = 100000
# hold_score = 18
# n_win_opt = 0
# n_win_hold = 0
# for _ in range(n_episode):
#     reward = simulate_episode(env, optimal_policy)
#     if reward == 1:
#         n_win_opt += 1
#     reward = simulate_hold_episode(env, hold_score)
#     if reward == 1:
#         n_win_hold += 1
# print(f'Prawdopodobieństwo wygranej\nProsta polityka: {n_win_hold/n_episode}\nOptymalna polityka: {n_win_opt/n_episode}')
# Taxi
env = gym.make('Taxi-v3')
n_state = env.observation_space.n
print(n_state)
n_action = env.action_space.n
print(n_action)
env.reset()
epsilon = 0.1
epsilon_greedy_policy = get_epsilon_greedy_policy(env.action_space.n, epsilon)
n_episode = 1000
length_episode = [0] * n_episode
total_reward_episode = [0] * n_episode
gamma = 1
alpha = 0.4
optimal_Q, optimal_policy = q_learning(env, gamma, n_episode, alpha, length_episode, total_reward_episode)

plt.plot(total_reward_episode)
plt.title('Sumaryczne nagrody w kolejnych epizodach')
plt.xlabel('Epizod')
plt.ylabel('Sumaryczna nagroda')
plt.ylim([-200, 20])
plt.show()

plt.plot(length_episode)
plt.title('Długość kolejnych epizodów')
plt.xlabel('Epizod')
plt.ylabel('Długość')
plt.show()