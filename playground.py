import time

from gui import GUI
import cv2 as cv

import numpy as np

for x in range(10):
    print(x, end='\r')
    time.sleep(3)
print()

blank = np.zeros((1000,1000,3), dtype='uint8')



steep = -1
ycut = 500

prev_steep = 2
prev_cut = 5
x1 = 0
x2 = 1500

pt1=(x1, x1*steep+ycut)
pt2=(x2, x2*steep+ycut)


intersect_x = (ycut - prev_cut) / (prev_steep - steep)
print(intersect_x)

print("Pt1: ", pt1, " Pt2 ", pt2)

cv.line(blank, pt1, pt2, color=(255,0,0), thickness=5)
GUI.imShow(blank)