import os
file_names = os.listdir(path='./datas/')
print(file_names)

import cv2
for name in file_names:
    video = []
    cap = cv2.VideoCapture('./datas/'+name)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    print(video[0][0][0])