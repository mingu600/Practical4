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
        self.Q = np.random.rand(40,2)
        self.alpha = 0.5
        self.gamma = 0.1
        self.flag = 0
        self.gravity = 1

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def find_index(self, state):
        state_data = [self.gravity, state['monkey']['top'], state['tree']['top'], state['monkey']['bot'], state['tree']['bot']]
        
        buffer = (state_data[2] - state_data[4])/2
        
        if state_data[1] - state_data[2] > -buffer:
            m = 0
        elif state_data[3] - state_data[4] < buffer:
            m = 1
        elif (state_data[1] + state_data[3])/2 - (state_data[2] + state_data[4])/2 > 0:
            m = 2
        else:
            m = 3
        
        g = state_data[0]
        n = math.floor(-g/2)
        if n > 9:
            n = 9
        
        index = int(n + m)
        
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
            self.flag = 0
        index = self.find_index(state)
        
        old_action = self.Q[self.find_index(self.last_state)][self.last_action]
        #print self.find_index(self.last_state) ,self.last_action
        
        #changed from self.Q[index] = ...
        #also debugged equation -- need to double-check
        self.Q[self.find_index(self.last_state)][self.last_action] = old_action + self.alpha * (self.last_reward + self.gamma * np.argmax(self.Q[index]) - old_action)
        
        self.last_action = np.argmax(self.Q[index])
        #print self.last_action
        self.last_state  = new_state
        #print [self.Q[index][0], self.Q[index][1]]
        
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
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games.
	run_games(agent, hist, 20, 1)

	# Save history.
	np.save('hist',np.array(hist))
