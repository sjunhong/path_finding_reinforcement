import math
from pprint import pprint

import numpy as np

BOARD_ROWS = 7
BOARD_COLS = 7
WIN_STATE = (5, 5)
LOSE_STATE = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
              (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
              (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
              (1, 6), (2, 6), (3, 6), (4, 6), (5, 6),
              (1, 2), (2, 2), (3, 2), (4, 2),
              (2, 4), (3, 4), (4, 4), (5, 4)]
START = (1, 1)
WALLS = []
DETERMINISTIC = False
GAMMA_DECAYS = [0.9]
EXPLORATION_RATE = [0.3]
EPOCHS = [10000000]



class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        for tile in WALLS:
            self.board[tile] = -1
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self):
        if self.state == WIN_STATE:
            return 1000
        elif self.state in LOSE_STATE:
            return -100
        else:
            return 0

    def isEndFunc(self):
        if (self.state == WIN_STATE) or (self.state in LOSE_STATE):
            self.isEnd = True

    def _chooseActionProb(self, action):
        if action == "up":
            return np.random.choice(["up", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "down":
            return np.random.choice(["down", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "left":
            return np.random.choice(["left", "up", "down"], p=[0.8, 0.1, 0.1])
        if action == "right":
            return np.random.choice(["right", "up", "down"], p=[0.8, 0.1, 0.1])

    def nxtPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position on board
        """
        if self.determine:
            if action == "up":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            self.determine = False
        else:
            # non-deterministic
            action = self._chooseActionProb(action)
            self.determine = True
            nxtState, action = self.nxtPosition(action)

        # if next state is legal
        if self.isValidAction(nxtState):
            return nxtState, action
        return self.state, action

    def isValidAction(self, nxtState):
        if (nxtState[0] >= 0) and (nxtState[0] <= BOARD_ROWS-1):
            if (nxtState[1] >= 0) and (nxtState[1] <= BOARD_COLS-1 ):
                if nxtState not in WALLS:
                    return True
        return False

    def showBoard(self):
        self.board[self.state] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')


class Agent:

    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.isEnd = self.State.isEnd
        self.lr = 0.5
        self.exp_rate = EXPLORATION_RATE[0]
        self.decay_gamma = GAMMA_DECAYS[0]

        # initial Q values
        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = -math.inf
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                current_position = self.State.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            # print("current pos: {}, greedy aciton: {}".format(self.State.state, action))
        return action

    def takeAction(self, action):
        position, a = self.State.nxtPosition(action)
        #print(position, a)
        '''if position == self.State.state:
            #print(self.Q_values[position][a])
            self.Q_values[position][a] -= 10000
            #print(self.Q_values[position][a])'''
        # update State
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()
        self.isEnd = self.State.isEnd

    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                for a in self.actions:
                    self.Q_values[self.State.state][a] = reward
                print(i, "Game End Reward", reward)
                for s in reversed(self.states):
                    current_q_value = self.Q_values[s[0]][s[1]]
                    reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                self.reset()
                i += 1
            else:
                action = self.chooseAction()
                # append trace
                self.states.append([(self.State.state), action])
                #print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc()
                #print("nxt state", self.State.state)
                #print("---------------------")
                self.isEnd = self.State.isEnd

    def printOptPath(self):
        path = [START]

        curr = START
        check = set()
        i = 1
        while curr != WIN_STATE and curr not in check and curr not in LOSE_STATE:
            check.add(curr)

            dir = max(self.Q_values[curr], key=self.Q_values[curr].get)
            #print(f'{i}. curr: {curr} & move {dir}')
            if dir == 'down':
                curr = (curr[0] + 1, curr[1])
            elif dir == 'up':
                curr = (curr[0] - 1, curr[1])
            elif dir == 'right':
                curr = (curr[0], curr[1] + 1)
            elif dir == 'left':
                curr = (curr[0], curr[1] - 1)

            i += 1
            path.append(curr)

        if path[-1] == WIN_STATE:
            print(f'path found: {path}')
        else:
            print('path not found')


if __name__ == "__main__":
    for decay in GAMMA_DECAYS:
        for rate in EXPLORATION_RATE:
            for num in EPOCHS:
                ag = Agent()
                ag.exp_rate = rate
                ag.decay_gamma = decay
                #print("initial Q-values ... \n")
                #pprint(ag.Q_values)

                ag.play(num)
                #print("latest Q-values ... \n")
                #pprint(ag.Q_values)

                ag.printOptPath()