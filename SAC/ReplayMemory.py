import numpy as np


class ReplayMemory:

    def __init__(self, capacity, state_dim, action_dim):

        self.states = np.empty(shape=(capacity, state_dim), dtype=np.float32)
        self.goals = np.empty(shape=(capacity, 3), dtype=np.float32)
        self.actions = np.empty(shape=(capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty(shape=(capacity,), dtype=np.float32)
        self.next_states = np.empty(shape=(capacity, state_dim), dtype=np.float32)
        self.dones = np.empty(shape=(capacity,), dtype=np.float32)
        

        self.idx = 0
        self.capacity = capacity

        self.was_full = False

    def add(self, state, goal, action, reward, next_state, done):

        self.states[self.idx] = state
        self.goals[self.idx] = goal 
        self.actions[self.idx] = action 
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done


        self.idx += 1

        if self.capacity <= self.idx:
            self.idx = 0
            self.was_full = True



    
    def sample(self, batch_size):

        size = self.idx
        if self.was_full:
            size = self.capacity
        

        idxs = np.random.choice(a=size, size=batch_size)

        states = self.states[idxs, :]
        goals = self.goals[idxs, :]
        actions = self.actions[idxs, :] 
        rewards = self.rewards[idxs]
        next_states = self.next_states[idxs, :]
        dones = self.dones[idxs]

        return states, goals, actions, rewards, next_states, dones