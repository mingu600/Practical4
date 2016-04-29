# Imports.
import numpy as np
import numpy.random as npr
import math

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.Q = np.random.rand(400,2)
        self.alpha = 0.5
        self.gamma = 0.5
        self.flag = 0
        self.gravity = 1
        self.time = 5
        self.tree_gap = 0
        self.epsilon = 1 / (self.time)

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def find_index(self, state):
        # total = self.Q.shape[0]
        # state_data = [self.gravity, state['monkey']['top'], state['tree']['top'], state['monkey']['bot'], state['tree']['bot'], state['tree']['dist']]
        # monkey_pos = (state_data[1] + state_data[3])/2
        #
        # if monkey_pos - state_data[2] > - 1.25 * self.tree_gap / 2.0:
        #     m = 0
        # elif monkey_pos - state_data[2] > - 2 * self.tree_gap / 2.0:
        #     m = 1
        # else:
        #     m = 2
        # g = state_data[0]
        # n = math.floor(-g/2)
        # if n > 9:
        #     n = 9
        #
        # index = int(n + 10 * m)

        state_data = [self.gravity, state['monkey']['top'], state['tree']['top'], state['monkey']['bot'], state['tree']['bot'], state['tree']['dist'],state['monkey']['vel']]
        monkey_pos = (state_data[1] + state_data[3]) / 2.0

        if monkey_pos - state_data[2] > - 1.25 * self.tree_gap / 2.0:
            m = 0
        elif monkey_pos - state_data[2] > - 2 * self.tree_gap / 2.0:
            m = 1
        else:
            m = 2
        # 2 bins
        d = math.floor(state_data[5] / 200.0)
        # 2 bins
        g = math.floor(-self.gravity / 2.0)
        index = int(100 * g + 10 * d + m)

        #index = int(18 * math.floor((-1 * state_data[2]) / 4) + 1 * math.floor((state_data[1] - state_data[0] + 200)/30))
        #index = int(100 * math.floor((-1 * state_data[4]) / 4) + 20 * math.floor((state_data[0] - 240) / 40) + 10 * math.floor((state_data[1] + 40) / 40)

        return index

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        new_state  = state

        # Get gravity here
        if self.last_state == None:
            self.flag = 1
            self.last_action = 0
            self.last_state = new_state
            return 0
        if self.flag == 1:
            self.gravity = state['monkey']['vel']
            self.tree_gap = state['tree']['top'] - state['tree']['bot']
            self.flag = 0

        if np.random.random_sample() > self.epsilon:
            index = self.find_index(state)

            old_action = self.Q[self.find_index(self.last_state)][self.last_action]
            #print self.find_index(self.last_state) ,self.last_action

            #changed from self.Q[index] = ...
            #also debugged equation -- need to double-check
            self.Q[self.find_index(self.last_state)][self.last_action] = old_action + self.alpha * (self.last_reward + self.gamma * np.argmax(self.Q[index]) - old_action)
            self.last_action = np.argmax(self.Q[index])
        else:
            self.last_action = np.random.randint(0, 2)
        #print self.last_action
        self.last_state  = new_state
        #print [self.Q[index][0], self.Q[index][1]]
        self.time += 0.1
        self.epsilon = 1 / self.time
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward

def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            learner.last_state = swing.get_state()
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    print max(hist)
    print float(sum(hist))/len(hist)
    print hist
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games.
	run_games(agent, hist, 100, 1)
	# Save history.
	np.save('hist',np.array(hist))
