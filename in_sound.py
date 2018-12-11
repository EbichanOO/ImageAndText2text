import librosa
def change_feature_v(sound, fs=44100):
    return librosa.feature.mfcc(sound, fs).T

x, fs = librosa.load('./kizuna_sample00.wav', sr=44100)
print(change_feature_v(x, fs))