from chainer import Chain
import chainer
import chainer.links as L
import chainer.functions as F

class VGG13(Chain):
    def __init__(self):
        super(VGG13, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3)
            self.conv1_2 = L.Convolution2D(64, 64, 3)
            self.conv2_1 = L.Convolution2D(64, 128, 3)
            self.conv2_2 = L.Convolution2D(128, 128, 3)
            self.conv3_1 = L.Convolution2D(128, 256, 3)
            self.conv3_2 = L.Convolution2D(256, 256, 3)
            self.conv4_1 = L.Convolution2D(256, 512, 3)
            self.conv4_2 = L.Convolution2D(512, 512, 3)
            self.conv5_1 = L.Convolution2D(512, 512, 3)
            self.conv5_2 = L.Convolution2D(512, 512, 3)
            self.fc6 = L.Linear(512*7*7, 4096)
    
    def __call__(self, x):
        out = F.relu(self.conv1_1(x))
        out = F.relu(self.conv1_2(out))
        out = F.max_pooling_2d(out, 2)
        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = F.max_pooling_2d(out, 2)
        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.max_pooling_2d(out, 2)
        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.max_pooling_2d(out, 2)
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_2(out))
        return F.dropout(F.relu(self.fc6(out)))