import numpy as np

class RolloutBuffer:

    def __init__(self, capacity, state_dim, action_dim):

        self.states = np.zeros(shape=(capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(shape=(capacity, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(shape=(capacity,), dtype=np.float32)
        self.advantages = np.zeros(shape=(capacity,), dtype=np.float32)
        self.values = np.zeros(shape=(capacity,), dtype=np.float32)

        self.idx = 0


    def add(self, state, action, log_prob, advantage, value):

        self.states[self.idx] = state 
        self.actions[self.idx] = action 
        self.log_probs[self.idx] = log_prob
        self.advantages[self.idx] = advantage
        self.values[self.idx] = value

        self.idx += 1

    def reset(self):
        self.idx = 0


    def get_batches(self, batch_size):
        size = self.idx
        idxs = np.random.permutation(size)

        num_batches = size // batch_size

        num_samples = num_batches * batch_size

        idxs = idxs[:num_samples]

        idxs_batched = np.reshape(idxs, (num_batches, batch_size))


        states_batch_list = []
        actions_batch_list = []
        log_probs_batch_list = []
        advantages_batch_list = []
        values_batch_list = []


        for idxs in idxs_batched:
            states = self.states[idxs, :]
            actions = self.actions[idxs, :] 
            log_probs = self.log_probs[idxs]
            advantages = self.advantages[idxs]
            values = self.values[idxs]


            log_probs = np.expand_dims(log_probs, axis=1)
            advantages = np.expand_dims(advantages, axis=1)
            values = np.expand_dims(values, axis=1)

            states_batch_list.append(states)
            actions_batch_list.append(actions)
            log_probs_batch_list.append(log_probs)
            advantages_batch_list.append(advantages)
            values_batch_list.append(values)


        return states_batch_list, actions_batch_list, log_probs_batch_list, advantages_batch_list, values_batch_list
