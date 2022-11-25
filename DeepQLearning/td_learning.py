import numpy as np
import os

class TDLearning:
    def __init__(self, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._q_table = np.zeros((input_dim, output_dim))


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        return self._q_table[state]

    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._q_table[states]


    def train_batch(self, states, q_sa):
        """
        Train td learning using the updated q-values
        """ 
        # TD Update
        td_delta = q_sa - self.predict_batch(states)
        self._q_table[states] += self._learning_rate * td_delta


    def save_model(self, path):
        """
        Save the current model in the folder as npy file
        """
        np.save(os.path.join(path, 'q_table'), self._q_table)

    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size