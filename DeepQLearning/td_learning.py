import numpy as np
import os
import sys

class TDLearning:
    def __init__(self, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._q_table = np.zeros((input_dim, output_dim))

    def _get_state_idx(self, state):
        lane_group, lane_cell = state
        return 10 * lane_group + lane_cell

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state_idx = self._get_state_idx(state)
        return self._q_table[state_idx]

    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        state_idxs = np.array([self._get_state_idx(state) for state in states])
        return self._q_table[state_idxs]


    def train_batch(self, states, q_sa):
        """
        Train td learning using the updated q-values
        """ 
        # TD Update
        state_idxs = np.array([self._get_state_idx(state) for state in states])
        td_delta = q_sa - self._q_table[state_idxs]
        self._q_table[state_idxs] += self._learning_rate * td_delta


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

class TDLearningTest(TDLearning):
    def __init__(self, input_dim, model_path):
        super().__init__(1, 0, input_dim, 1)
        self._q_table = self._load_my_model(model_path)

    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'q_table.npy')
        
        if os.path.isfile(model_file_path):
            loaded_model = np.load(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")
