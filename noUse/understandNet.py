import numpy as np
class understand:
    def __init__(self):
        #self.max_size = max(s_vec_size, i_vec_size)

        try:
            self.vector_space = np.load('imageText_vector.npz')['space']
            self.maxs = self.vector_space.shape
        except:
            self.vector_space = [0]
            self.maxs = [0, 0]
    
    def update(self, s_vec, i_vec):
        if np.max(s_vec)+1>self.maxs[0] or np.max(i_vec)+1>self.maxs[1]:
            s_max =  np.max(s_vec)+1 if np.max(s_vec)>self.maxs[0] else self.maxs[0]
            i_max = np.max(i_vec)+1 if np.max(i_vec)>self.maxs[1] else self.maxs[1]
            
            new_space = np.zeros((s_max, i_max))
            new_space[:self.maxs[0], :self.maxs[1]] = self.vector_space
            self.vector_space = np.copy(new_space)
            self.maxs = [s_max, i_max]
        for s, i in zip(s_vec, i_vec):
            self.vector_space[s][i] += 1

    def sentence2image(self, x):
        temp = np.zeros(x.size)
        j=0
        for i in x:
            try:
                temp[j] = np.argmax(self.vector_space[i])
            except:
                pass
            j+=1
        return temp
    
    def image2sentence(self, x):
        temp = np.zeros(x.size)
        j=0
        for i in x:
            try:
                temp[j] = np.argmax(self.vector_space.T[i])
            except:
                pass
            j+=1
        return temp

    def saveSpace(self):
        np.savez_compressed('imageText_vector.npz', space=self.vector_space)

def cos_sim_matrix(v1, v2):
    d = v1 @ v2.T
    norm1 = (v1*v1).sum(0, keepdims=True) **.5
    norm2 = (v2*v2).sum(0, keepdims=True) **.5
    return d/(norm1 * norm2)