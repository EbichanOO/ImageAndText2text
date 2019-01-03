import chainer
from bertChainer.modeling import BertConfig, BertModel
import librosa

class SoundNet(chainer.Chain):
    def __init__(self):
        super(SoundNet, self).__init__()
        config = BertConfig(vocab_size=32000, hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=1024)
        with self.init_scope():
            self.bert = BertModel(config)
    
    def __call__(self, x):
        out = librosa.feature.mfcc(x, 44100).T
        out = self.bert(out)
        return out

def change_feature_v(sound, fs=44100):
    return librosa.feature.mfcc(sound, fs).T

#x, fs = librosa.load('./kizuna_sample00.wav', sr=44100)