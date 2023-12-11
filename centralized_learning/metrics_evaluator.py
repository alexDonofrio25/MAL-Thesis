import numpy as np
import matplotlib.pyplot as plt

class Evaluator():

    def __init__(self, rewards_per_episode,explo_expo):
        self.rewards = rewards_per_episode
        self.policy_track = explo_expo
        self.episodes = len(rewards_per_episode)

    def cumulated_reward(self):
        rewards_summation = np.zeros((self.episodes))
        i = 0
        for r in self.rewards:
            rewards_summation[i] = np.sum(r)
        fig, ax = plt.subplots()
        ax.plot(rewards_summation)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Cululated Reward per Episode')
        return fig, ax

    def average_reward(self):
        rewards_avg = np.zeros((self.episodes))
        i = 0
        for r in self.rewards:
            rewards_avg[i] = np.mean(r)
        fig, ax = plt.subplots()
        ax.plot(rewards_avg)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Average Reward per Episode')
        return fig, ax

    def std_reward(self):
        rewards_std = np.zeros((self.episodes))
        i = 0
        for r in self.rewards:
            rewards_std[i] = np.std(r)
        fig, ax = plt.subplots()
        ax.plot(rewards_std)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Standard Deviation of Reward per Episode')
        return fig, ax

    def episode_reward(self, episode):
        r = self.rewards[episode]
        fig, ax = plt.subplots()
        ax.plot(r)
        ax.set_xlabel('Action')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Reward')
        return fig, ax

    def cumulated_reward_ma(self, n_agents):
        reward_summation = np.zeros((n_agents,self.episodes))
        for i in range(0,n_agents):
            j = 0
            for r in self.rewards[i,:]:
                reward_summation[i,j] = np.sum(r)
                j += 1
        return reward_summation

    def average_reward_ma(self, n_agents):
        reward_average = np.zeros((n_agents,self.episodes))
        for i in range(0,n_agents):
            j = 0
            for r in self.rewards[i,:]:
                reward_average[i,j] = np.mean(r)
                j += 1
        return reward_average

    def std_reward_ma(self, n_agents):
        reward_std = np.zeros((n_agents,self.episodes))
        for i in range(0,n_agents):
            j = 0
            for r in self.rewards[i,:]:
                reward_std[i,j] = np.std(r)
                j += 1
        return reward_std

    def sensibility_analysis(self,
                             ma,
                             qlearning,
                             epochs = False,
                             episode = False,
                             alpha = False,
                             epo_range = range(0,1),
                             ep_range = range(0,1),
                             alpha_range = range(0,1)):
        rewardsE = []
        if epochs:
            for e in epo_range:
                if ma:
                    Q1,Q2, r = qlearning(epochs = e,ep_length = 7, gamma = 0.9)
                    last_reward = np.sum(r[e-1,:])
                    rewardsE.append(last_reward)
                else:
                    Q, r = qlearning(epochs = e,ep_length = 7, gamma = 0.9)
                    last_reward = np.sum(r[e-1,:])
                    rewardsE.append(last_reward)

        rewardsEP = []
        if episode:
            for e in ep_range:
                if ma:
                    Q1,Q2, r = qlearning(epochs = 200,ep_length = e, gamma = 0.9)
                    last_reward = np.sum(r[199,:])
                    rewardsEP.append(last_reward)
                else:
                    Q, r = qlearning(epochs = 200,ep_length = e, gamma = 0.9)
                    last_reward = np.sum(r[199,:])
                    rewardsEP.append(last_reward)

        rewardsA = []
        if alpha:
            for a in alpha_range:
                if ma:
                    Q1,Q2, r = qlearning(epochs = 200,ep_length = 7, gamma = a)
                    last_reward = np.sum(r[199,:])
                    rewardsA.append(last_reward)
                else:
                    Q, r = qlearning(epochs = 200,ep_length = 7, gamma = a)
                    last_reward = np.sum(r[199,:])
                    rewardsA.append(last_reward)
        return rewardsEP, rewardsE, rewardsA


