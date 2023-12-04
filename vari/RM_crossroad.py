import numpy as np
import pandas as pd
from random import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Crossroad:
    actions = ['MOVE', 'WAIT']
    n_actions = 2
    # design the utility matrix for player 1
    utility_matrix = pd.DataFrame([
        [-1, 1],
        [0.5, -1]
    ], columns=actions, index=actions)

class Player():
    def __init__(self,name):
        self.strategy, self.avg_strategy,\
        self.strategy_sum, self.regret_sum = np.zeros((4, Crossroad.n_actions))
        self.name = name

    def update_strategy(self):
        #copy the regret summation into the strategy in order to have it proportional to the regret
        #print('Strategy before new regret:' + str(self.strategy))
        self.strategy = np.copy(self.regret_sum)
        #print('Strategy with new regret:' + str(self.strategy))
        self.strategy[self.strategy<0] = 0 # set negative probability value obtained from the regret to zero
        summation = sum(self.strategy) #compute the all regret summation,
        #print('Summation:' + str(summation))
        if summation > 0:
            self.strategy /= summation #normalize
        else:
            self.strategy = np.repeat(1/Crossroad.n_actions, Crossroad.n_actions)

        self.strategy_sum += self.strategy
        #print('New strategy:' + str(self.strategy))
        #print('New strategy sum:' + str(self.strategy_sum))

    def regret(self, my_action, opp_action):
        #compute the regret as the difference between the reward (utility) of an action minus the reward of the real taken action
        result = Crossroad.utility_matrix.loc[my_action,opp_action] #the reward obtained from taking my_action
        facts = Crossroad.utility_matrix.loc[:,opp_action].values #
        regret = facts - result
        self.regret_sum += regret #sum the regret obtained to the one already cumulated

    def action(self, use_avg=False):
        """
        select an action according to strategy probabilities
        """
        strategy = self.avg_strategy if use_avg else self.strategy
        return np.random.choice(Crossroad.actions, p=strategy)

    def learn_avg_strategy(self):
        # averaged strategy converges to Nash Equilibrium
        summation = sum(self.strategy_sum)
        if summation > 0:
            self.avg_strategy = self.strategy_sum / summation
        else:
            self.avg_strategy = np.repeat(1/Crossroad.n_actions, Crossroad.n_actions)

class TimeToCross():
    def __init__(self, max_game):
        self.p1 = Player('Blue')
        self.p2 = Player('Red')
        self.positive_results = 0
        self.max_game = max_game
        self.p1_chosen_probabilities = []
        self.p2_chosen_probabilities = []

    def __repr__(self):
        return 'There are a ' + self.p1.__repr__ + 'and a ' + self.p2.__repr__ + ' in an intersection, which will cross? Will they crash?'

    def semaphor(self,a1,a2):
        # the semaphor check if the two robot have crashed or not
        if a1 != a2:
            return 1 # not crashed
        else:
            return 0 # crashed

    def reset(self):
        return 'Reset'
        #this function is used to reset the robot position on the intersection, now is useless but has to be implemented

    def play(self, avg_regret_matching=False):
        def play_regret_matching():
            for i in range(0, self.max_game):
                self.p1.update_strategy()
                self.p2.update_strategy()
                a1 = self.p1.action()
                a2 = self.p2.action()
                self.p1.regret(a1, a2)
                self.p2.regret(a2, a1)

                s = self.semaphor(a1,a2)
                if s==1: self.positive_results += 1
                self.plotProbabilities(a1,a2)

        def play_avg_regret_matching():
            for i in range(0, self.max_game):
                a1 = self.p1.action(use_avg=True)
                a2 = self.p2.action(use_avg=True)
                s = self.semaphor(a1,a2)
                if s==1: self.positive_results += 1

        play_regret_matching() if not avg_regret_matching else play_avg_regret_matching()
        print('Percentuale di successo:' + str(self.positive_results/self.max_game*100) + '%')

    def conclude(self):
        """
        let two players conclude the average strategy from the previous strategy stats
        """
        self.p1.learn_avg_strategy()
        self.p2.learn_avg_strategy()

    def plotProbabilities(self, a1, a2):
        if a1=='MOVE':
            self.p1_chosen_probabilities.append(self.p1.strategy[0])
            x = self.p1_chosen_probabilities
        else:
            self.p1_chosen_probabilities.append(self.p1.strategy[1])
            x = self.p1_chosen_probabilities
        if a2=='MOVE':
            self.p2_chosen_probabilities.append(self.p2.strategy[0])
            y = self.p2_chosen_probabilities
        else:
            self.p2_chosen_probabilities.append(self.p2.strategy[1])
            y = self.p2_chosen_probabilities
        plt.xlim(0, 1.3)
        plt.ylim(0, 1.3)
        plt.plot(x, y, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="white")
        plt.show()
        plt.close()



if __name__ == '__main__':
    ttc50 = TimeToCross(max_game=50)
    print('=== Use simple regret-matching strategy with 50 ephocs=== ')
    ttc50.play()
    #ttc500 = TimeToCross(max_game=500)
    print('=== Use simple regret-matching strategy with 500 ephocs=== ')
    #ttc500 = ttc500.play()
    #ttc1000 = TimeToCross(max_game=1000)
    print('=== Use simple regret-matching strategy with 1000 ephocs=== ')
    #ttc1000.play()
    #ttc10000 = TimeToCross(max_game=10000)
    print('=== Use simple regret-matching strategy with 10000 ephocs=== ')
    #ttc10000.play()

