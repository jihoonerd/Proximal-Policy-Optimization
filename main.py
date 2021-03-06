import gym
import pybullet_envs
import numpy as np

from ppo.agent import Agent


if __name__ == '__main__':
    env = gym.make('AntBulletEnv-v0')

    learn_interval = 100
    batch_size = 5000
    n_epochs = 1000
    learning_rate = 0.0003
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    agent = Agent(n_actions=action_space, batch_size=batch_size,
                  learning_rate=learning_rate, n_epochs=n_epochs, input_dims=observation_space)
    n_games = 300

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.push(observation, action, prob, val, reward, done)
            if n_steps % learn_interval == 0:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        print(f'Episode: {i} / Score: {score} / AVG Score (100): {avg_score}')