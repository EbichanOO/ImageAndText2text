import numpy as np
class understand:
    def __init__(self, s_vec_size, i_vec_size):
        self.s_vec = np.zeros(s_vec_size)
        self.i_vec = np.zeros(i_vec_size)
    
    def reward(self, input):
        