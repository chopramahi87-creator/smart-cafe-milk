"""agent.py - Q-learning agent (the BRAIN)."""
import numpy as np, random


def train(env, episodes=15000, alpha=0.15, gamma=0.95,
          eps_start=1.0, eps_min=0.05, eps_decay=0.9998):
    n_states = env.max_stock + 1
    n_actions = env.max_stock + 1
    Q = np.zeros((n_states, n_actions))
    eps = eps_start
    reward_curve = []
    for episode in range(episodes):
        state = env.reset(); done = False; total = 0
        while not done:
            if random.random() < eps:
                action = random.randrange(n_actions)          # explore
            else:
                action = int(np.argmax(Q[state]))             # exploit
            reward, next_state, done, _ = env.step(action)
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])
            state = next_state; total += reward
        reward_curve.append(total)
        eps = max(eps_min, eps * eps_decay)
    return Q, reward_curve


def policy_from_Q(Q):
    return np.argmax(Q, axis=1)
