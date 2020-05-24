import gym
from A2C import Agent
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    agent = Agent(alpha=0.00001, beta=0.00005)
    env = gym.make('LunarLander-v2')
    score_history = []
    num_episodes = 1000
    agent.load_weights('actor.h5', 'critic.h5')
    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            env.render()
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, action, reward, observation_, done)
            observation = observation_
            score += reward
        print("Episode number: {}   reward: {}".format(i+1,score) )
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
    # Plot the scores in each iteration:
    plt.plot(score_history)
    plt.title("Score over {} episodes".format(num_episodes))
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.show()
    # Save weights:
    agent.save_weights()
