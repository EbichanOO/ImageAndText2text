from main import cos_sim_matrix
import numpy as np
a = np.array([i for i in range(15)])
b = np.ones(15)
print(cos_sim_matrix(a, b))