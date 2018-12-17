import chainer
from bertChainer import modeling
import librosa

class SoundNet(chainer.Chain):
    def __init__(self):
        super(SoundNet, self).__init__()
        with self.init_scope():
            self.bert = modeling.BertModel(255)
    
    def __call__(self, x):
        out = librosa.feature.mfcc(x, 44100).T
        out = self.bert(out)
        return out

def change_feature_v(sound, fs=44100):
    return librosa.feature.mfcc(sound, fs).T

#x, fs = librosa.load('./kizuna_sample00.wav', sr=44100)