# Test the trained network:
import gym
from A2C import Agent
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    agent = Agent(alpha=0.00001, beta=0.00005)
    agent.load_weights(filename1='actor_more1000.h5', filename2='critic_more1000.h5')
    env = gym.make('LunarLander-v2')
    score_history = []
    num_episodes = 100
    for i in range(num_episodes):
        done = False
        score = 0
        win100_count = 0
        win200_count = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            env.render()
            observation_, reward, done, info = env.step(action)
            # agent.learn(observation, action, reward, observation_, done)
            observation = observation_
            score += reward
        print("Episode number: {}   reward: {}".format(i + 1, score))
        score_history.append(score)
        avg_score = np.mean(score_history)
        if 100 <= score < 200:
            win100_count += 1
        if score >= 200:
            win200_count += 1
    # Plot the scores in each iteration:
    plt.plot(score_history)
    plt.title("Score over {} episodes".format(num_episodes))
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.show()
    # Show results:
    print("Results of {} episodes: \nScore above 100: {}\nScore above 200: {}\nAverage score: {}".format(num_episodes, win100_count, win200_count, avg_score))
