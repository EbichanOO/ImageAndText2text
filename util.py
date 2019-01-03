def cos_sim_matrix(v1, v2):
    d = v1 @ v2.T
    norm1 = (v1*v1).sum(0, keepdims=True) **.5
    norm2 = (v2*v2).sum(0, keepdims=True) **.5
    return d/(norm1 * norm2)