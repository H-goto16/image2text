import cv2
from image2text import i2t

i2t = i2t()

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

text = i2t.image2text(frame)

for text in i2t.realtime_i2t():
    print(text)
