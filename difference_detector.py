import cv2
import numpy as np


#print(cv2.__version__)
from gui import GUI


class Difference_Detector:

    #original
    # ENV = {
    #     'BLUR' : (5,5),
    #     'BINARY_THRESHOLD_MIN' : 75,
    #     'BINARY_THRESHOLD_MAX' : 255,
    #     'CLAHE_CLIP_LIMIT' : 5,
    #     'CLAHE_TILE_SIZE' : (10,10),
    #
    #     'ARROW_BLUR' : (5,5),
    #     'ARROW_BINARY_THRESHOLD_MIN' : 50,
    #     'ARROW_BINARY_THRESHOLD_MAX' : 255,
    #     'ARROW_CLAHE_CLIP_LIMIT' : 20,
    #     'ARROW_CLAHE_TILE_SIZE' : (10,10)
    # }
    #def __init__(self):
    ENV = {
        'BLUR' : (11,11),
        'BINARY_THRESHOLD_MIN' : 75,
        'BINARY_THRESHOLD_MAX' : 255,
        'CLAHE_CLIP_LIMIT' : 5,
        'CLAHE_TILE_SIZE' : (10,10),

        'ARROW_BLUR' : (5,5),
        'ARROW_BINARY_THRESHOLD_MIN' : 25,
        'ARROW_BINARY_THRESHOLD_MAX' : 255,
        'ARROW_CLAHE_CLIP_LIMIT' : 20,
        'ARROW_CLAHE_TILE_SIZE' : (5,5)
    }


    def computeDifference(self,grey1,grey2):
        # blur
        #GUI.im_compare([grey2, grey1])
        blur = Difference_Detector.ENV['BLUR']
        grey2 = cv2.blur(grey2,blur)
        grey1 = cv2.blur(grey1,blur)
        #normalize
        grey1 = cv2.equalizeHist(grey1)
        grey2 = cv2.equalizeHist(grey2)
        clahe = cv2.createCLAHE(Difference_Detector.ENV['CLAHE_CLIP_LIMIT'], Difference_Detector.ENV['CLAHE_TILE_SIZE'])

        #clahe
        grey1 = clahe.apply(grey1)
        grey2 = clahe.apply(grey2)
        #diff
        diff = cv2.subtract(grey2,grey1) + cv2.subtract(grey1,grey2)
        #GUI.im_compare([diff, grey2])
        ret2,dif_thred = cv2.threshold(diff,Difference_Detector.ENV['BINARY_THRESHOLD_MIN'],Difference_Detector.ENV['BINARY_THRESHOLD_MAX'],cv2.THRESH_BINARY)
        ret3, arrow = cv2.threshold(diff, 40,Difference_Detector.ENV['BINARY_THRESHOLD_MAX'], cv2.THRESH_BINARY)

        #GUI.imShow(arrow)
        return dif_thred,grey1,grey2,diff,arrow


    def computeDifferenceHighRes(self,grey1,grey2):
        # blur
        blur = Difference_Detector.ENV['BLUR']
        grey2 = cv2.blur(grey2,blur)
        grey1 = cv2.blur(grey1,blur)
        #normalize
        grey1 = cv2.equalizeHist(grey1)
        grey2 = cv2.equalizeHist(grey2)
        clahe = cv2.createCLAHE(Difference_Detector.ENV['ARROW_CLAHE_CLIP_LIMIT'], Difference_Detector.ENV['ARROW_CLAHE_TILE_SIZE'])
        #clahe
        grey1 = clahe.apply(grey1)
        grey2 = clahe.apply(grey2)
        #diff
        diff = cv2.subtract(grey2,grey1) + cv2.subtract(grey1,grey2)
        ret2,dif_thred = cv2.threshold(diff,Difference_Detector.ENV['ARROW_BINARY_THRESHOLD_MIN'],Difference_Detector.ENV['ARROW_BINARY_THRESHOLD_MAX'],cv2.THRESH_BINARY)
        return dif_thred