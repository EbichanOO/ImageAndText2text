import numpy as np
class DataMaker:
    def __init__(self):
        import os
        self.file_names = os.listdir(path='./datas/')

    def videoData(self):
        import cv2
        for name in self.file_names:
            i=0
            video = []
            cap = cv2.VideoCapture('./datas/'+name)
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    tmp = np.array(frame)
                    video.append(tmp.T)
                else:
                    break

                if len(video)%3==0:
                    yield np.array(video[i:i+3], dtype=np.float32)
                    i+=1
            cap.release()

    def soundData(self):
        for name in self.file_names:
            from pydub import AudioSegment
            sound = AudioSegment.from_file('./datas/'+name, "mp4").get_array_of_samples()
            sound = np.array(sound, dtype=np.float32)
            for i in range(0, len(sound)-100, 100):
                yield sound[i:i+100]

'''
これで使う
DM = DataMaker()
for video in DM.videoData():
for sound in DM.soundData():
'''