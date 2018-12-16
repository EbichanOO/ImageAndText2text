import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
class NN(chainer.Chain):
    def __init__(self, inputSize, outSize, roomSize):
        super(NN, self).__init__()
        self.inputSize = inputSize
        self.outSize = outSize
        self.roomSize = roomSize
        self.state = np.zeros(self.roomSize)

        with self.init_scope():
            self.l1 = L.Linear(self.inputSize, self.roomSize)
            self.l2 = L.Linear(self.roomSize, self.roomSize)
            self.l3 = L.Linear(self.roomSize, self.outSize)
    
    def __call__(self, x):
        temp = F.relu(self.l1(x)+self.state)
        self.state = F.Dropout(F.relu(self.l2(temp)))
        return self.l3(self.state)