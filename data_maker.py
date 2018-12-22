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
                    video.append(frame)
                else:
                    break

                if len(video)%30==0:
                    yield video[i:i+30]
                    i+=1
            cap.release()

    def soundData(self):
        for name in self.file_names:
            from pydub import AudioSegment
            self.sound = AudioSegment.from_file('./datas/'+name, "mp4")
            for i in range(0, len(self.sound)-1000, 1000):
                yield self.sound[i:i+1000]