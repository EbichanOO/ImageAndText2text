from chainer import Chain
import chainer
import chainer.links as L
import chainer.functions as F

from bertChainer.modeling import BertModel, BertConfig
class ImNet(Chain):
    def __init__(self):
        super(ImNet, self).__init__()
        config = BertConfig(vocab_size=32000, hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=1024)
        with self.init_scope():
            self.bert = BertModel(config=config)
            self.vgg = VGG13()
    
    def __call__(self, x):
        return self.bert(self.vgg(x))

def configMaker():
    import json
    Bconfig = BertConfig(vocab_size=255)
    config_json = Bconfig.to_json_string()
    with open('bert_config.json', 'w') as fw:
        json.dump(config_json, fw)

class VGG13(Chain):
    def __init__(self):
        super(VGG13, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.fc6 = L.Linear(73728, 4096)
            #self.fc6 = L.Linear(1843200, 4096)
    
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

