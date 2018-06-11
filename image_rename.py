import os
import cv2
import numpy as np

direc = 'driving_dataset2'
count = 0
for img in os.listdir(direc):
    print(img)
    f1 = cv2.imread('driving_dataset2/'+img)
    cv2.imwrite("test1/%d.jpg"%count,f1)
    count +=1
